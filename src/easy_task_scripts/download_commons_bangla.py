import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from urllib.error import HTTPError

API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_EXTS = {".mp3", ".ogg", ".flac", ".wav", ".m4a", ".oga"}

USER_AGENT = "VAE-Hybrid-Music-Downloader/1.0 (educational; python-urllib)"
SLEEP_SEC = float(os.environ.get("SLEEP_SEC", "2.0"))     # slow down downloads
N_EACH = int(os.environ.get("N_EACH", "10"))              # keep small to avoid rate limit
ONLY = os.environ.get("ONLY", "both").strip().lower()     # "en", "bn", or "both"


class RateLimited(Exception):
    def __init__(self, retry_after: Optional[int], msg: str):
        super().__init__(msg)
        self.retry_after = retry_after


def request_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.read()
    except HTTPError as e:
        if e.code == 429:
            ra = e.headers.get("Retry-After")
            retry_after = int(ra) if (ra and ra.isdigit()) else None
            raise RateLimited(retry_after, f"HTTP 429 Too Many Requests for URL: {url}")
        raise


def api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    raw = request_bytes(url)
    return json.loads(raw.decode("utf-8"))


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def category_titles(category: str, limit: int) -> List[str]:
    out: List[str] = []
    cont = None
    while len(out) < limit:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:" + category,
            "cmtype": "file",
            "cmlimit": "50",
            "format": "json",
        }
        if cont:
            params.update(cont)

        data = api_get(params)
        out.extend([m["title"] for m in data["query"]["categorymembers"]])

        cont = data.get("continue")
        if not cont:
            break

        time.sleep(0.2)
    return out[:limit]


def urls_for_titles(titles: List[str]) -> Dict[str, str]:
    """
    Batch query: one API call can return URLs for many File: titles.
    """
    if not titles:
        return {}

    joined = "|".join(titles)
    data = api_get({
        "action": "query",
        "titles": joined,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    })

    out: Dict[str, str] = {}
    for _, page in data["query"]["pages"].items():
        title = page.get("title")
        ii = page.get("imageinfo")
        if title and ii and isinstance(ii, list) and ii[0].get("url"):
            out[title] = ii[0]["url"]
    return out


def download_file(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(req) as r, open(out_path, "wb") as f:
            f.write(r.read())
    except HTTPError as e:
        if e.code == 429:
            ra = e.headers.get("Retry-After")
            retry_after = int(ra) if (ra and ra.isdigit()) else None
            raise RateLimited(retry_after, f"HTTP 429 on download: {url}")
        raise


def grab(category: str, out_dir: Path, n_each: int, tag: str) -> int:
    # oversample titles; we’ll filter by extension
    titles = category_titles(category, n_each * 8)

    filtered: List[str] = []
    for t in titles:
        ext = Path(t).suffix.lower()
        if ext in ALLOWED_EXTS:
            filtered.append(t)

    # Batch URL lookup in chunks (reduces API spam)
    got = 0
    for i in range(0, len(filtered), 25):
        chunk = filtered[i:i+25]
        url_map = urls_for_titles(chunk)

        for title in chunk:
            if got >= n_each:
                return got

            url = url_map.get(title)
            if not url:
                continue

            fname = safe_name(title.replace("File:", ""))
            try:
                download_file(url, out_dir / fname)
            except RateLimited as rl:
                print(f"{tag} RATE LIMITED (429).")
                if rl.retry_after is not None:
                    print(f"{tag} Server says Retry-After: {rl.retry_after} seconds.")
                print(f"{tag} Stop now and re-run later with SAME command (don’t spam retries).")
                return got

            got += 1
            print(f"{tag} {got} -> {fname}")
            time.sleep(SLEEP_SEC)

    return got


def main():
    project_root = Path(__file__).resolve().parents[2]
    out_en = project_root / "data" / "audio" / "english"
    out_bn = project_root / "data" / "audio" / "bangla"

    # Categories
    cat_en = "Audio files of songs in English"
    cat_bn = "Audio files of Bengali music"

    en_got = bn_got = 0

    try:
        if ONLY in ("both", "en", "english"):
            print("Downloading English from:", cat_en)
            en_got = grab(cat_en, out_en, N_EACH, "EN")

        if ONLY in ("both", "bn", "bangla", "bengali"):
            print("Downloading Bangla/Bengali from:", cat_bn)
            bn_got = grab(cat_bn, out_bn, N_EACH, "BN")

    except RateLimited as rl:
        # Should already be handled inside grab(), but keep as safety net
        print("RATE LIMITED (429). Stop and re-run later.")

    print(f"Done. Downloaded EN={en_got} BN={bn_got}")
    print("Folders:")
    print(" -", out_en)
    print(" -", out_bn)


if __name__ == "__main__":
    main()
