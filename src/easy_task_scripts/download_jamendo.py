import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

API = "https://api.jamendo.com/v3.0/tracks"


def jamendo_get(url: str) -> dict:
    """
    Jamendo responses include a 'headers' object with 'code' and possibly 'error_message'.
    If code != 0, the request failed (e.g., invalid client_id).
    """
    with urllib.request.urlopen(url) as r:
        data = json.loads(r.read().decode("utf-8"))

    headers = data.get("headers", {})
    code = headers.get("code", None)

    # Jamendo uses code=0 for success. Anything else is an error.
    if code != 0:
        raise RuntimeError(
            f"Jamendo API error.\n"
            f"URL: {url}\n"
            f"headers: {headers}\n"
            f"(Tip: set JAMENDO_CLIENT_ID to a valid client id.)"
        )
    return data


def fetch_tracks(client_id: str, *, lang: str, limit: int):
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": str(limit),
        "audioformat": "mp32",
        "vocalinstrumental": "vocal",
        # Jamendo supports lang[] for lyrics language filtering
        "lang[]": lang,  # "en", "bn", etc.
    }
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    data = jamendo_get(url)
    return data.get("results_easy", [])


def fetch_tracks_fallback(client_id: str, *, query: str, limit: int):
    params = {
        "client_id": client_id,
        "format": "json",
        "limit": str(limit),
        "audioformat": "mp32",
        "vocalinstrumental": "vocal",
        # fallback search term if lang[] gives nothing
        "search": query,
    }
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    data = jamendo_get(url)
    return data.get("results_easy", [])


def pick_audio_url(track: dict) -> Optional[str]:

    """
    Prefer audiodownload if present; otherwise use audio preview URL.
    Some tracks may not allow downloads; some may have missing fields.
    """
    # Try download first (if allowed/provided)
    url = track.get("audiodownload")
    if url:
        return url
    # Otherwise fall back to "audio" (often a stream/preview)
    url = track.get("audio")
    if url:
        return url
    return None


def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())


def sanity_check(client_id: str):
    """Quick check that your client_id works at all."""
    params = {"client_id": client_id, "format": "json", "limit": "1"}
    url = API + "/?" + urllib.parse.urlencode(params, doseq=True)
    data = jamendo_get(url)
    headers = data.get("headers", {})
    print("Jamendo OK. headers:", headers)


def main():
    # IMPORTANT: If this is invalid, the script will now ERROR with a clear message.
    client_id = os.environ.get("JAMENDO_CLIENT_ID", "709fa152")

    n_each = int(os.environ.get("N_EACH", "30"))

    project_root = Path(__file__).resolve().parents[2]
    out_en = project_root / "data" / "audio" / "english"
    out_bn = project_root / "data" / "audio" / "bangla"

    # 0) sanity check (will raise if client_id is invalid)
    sanity_check(client_id)

    # 1) fetch tracks
    print("Fetching English tracks…")
    en = fetch_tracks(client_id, lang="en", limit=n_each * 3)
    if not en:
        print("No EN via lang[]. Trying fallback search='english' …")
        en = fetch_tracks_fallback(client_id, query="english", limit=n_each * 3)

    print("Fetching Bengali/Bangla tracks…")
    bn = fetch_tracks(client_id, lang="bn", limit=n_each * 3)
    if not bn:
        print("No BN via lang[]. Trying fallback search='bengali' …")
        bn = fetch_tracks_fallback(client_id, query="bengali", limit=n_each * 3)

    print(f"Fetched candidates: EN={len(en)}  BN={len(bn)}")

    # 2) download up to n_each per language (skipping tracks without audio urls)
    en_done = 0
    for t in en:
        if en_done >= n_each:
            break
        audio_url = pick_audio_url(t)
        if not audio_url:
            continue

        tid = t.get("id", f"en_{en_done+1}")
        out = out_en / f"{tid}.mp3"
        download(audio_url, out)
        en_done += 1
        print("EN", en_done, "->", out.name)
        time.sleep(0.25)

    bn_done = 0
    for t in bn:
        if bn_done >= n_each:
            break
        audio_url = pick_audio_url(t)
        if not audio_url:
            continue

        tid = t.get("id", f"bn_{bn_done+1}")
        out = out_bn / f"{tid}.mp3"
        download(audio_url, out)
        bn_done += 1
        print("BN", bn_done, "->", out.name)
        time.sleep(0.25)

    print(f"Done. Downloaded EN={en_done} BN={bn_done}")
    print("Check:", out_en)
    print("Check:", out_bn)

    if en_done == 0 and bn_done == 0:
        print("\nNothing downloaded.")
        print("Most likely: your JAMENDO_CLIENT_ID is invalid, or downloads are not available for returned tracks.")
        print("If sanity_check passed but still 0, switch to Wikimedia Commons approach.")


if __name__ == "__main__":
    main()
