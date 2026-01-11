import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict

API = "https://commons.wikimedia.org/w/api.php"
ALLOWED_EXTS = {".mp3", ".ogg", ".flac", ".wav", ".m4a", ".oga"}

# IMPORTANT: Wikimedia may return 403 if User-Agent is missing/too generic.
USER_AGENT = "VAE-Hybrid-Music-Downloader/1.0 (contact: local-project; python-urllib)"


def http_get(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
        },
        method="GET",
    )
    with urllib.request.urlopen(req) as r:
        return r.read()


def api_get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params, doseq=True)
    raw = http_get(url)
    return json.loads(raw.decode("utf-8"))


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def category_files(category: str, limit: int) -> List[Dict]:
    out = []
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
        out.extend(data["query"]["categorymembers"])
        cont = data.get("continue")
        if not cont:
            break

        time.sleep(0.1)

    return out[:limit]


def file_url(file_title: str) -> Optional[str]:
    data = api_get({
        "action": "query",
        "titles": file_title,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json",
    })
    pages = data["query"]["pages"]
    page = next(iter(pages.values()))
    ii = page.get("imageinfo")
    if not ii:
        return None
    return ii[0].get("url")


def download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return

    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT},
        method="GET",
    )
    with urllib.request.urlopen(req) as r, open(out_path, "wb") as f:
        f.write(r.read())


def grab(category: str, out_dir: Path, n_each: int, tag: str) -> int:
    candidates = category_files(category, n_each * 6)  # oversample
    got = 0

    for item in candidates:
        title = item["title"]  # "File:Something.mp3"
        ext = Path(title).suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue

        url = file_url(title)
        if not url:
            continue

        fname = safe_name(title.replace("File:", ""))
        try:
            download(url, out_dir / fname)
        except Exception as e:
            print(f"{tag} skip (download error): {fname} -> {e}")
            continue

        got += 1
        print(f"{tag} {got} -> {fname}")

        if got >= n_each:
            break

        time.sleep(0.2)

    return got


def main():
    n_each = int(os.environ.get("N_EACH", "30"))

    project_root = Path(__file__).resolve().parents[2]
    out_en = project_root / "data" / "audio" / "english"
    out_bn = project_root / "data" / "audio" / "bangla"

    # Commons categories
    cat_en = "Audio files of songs in English"
    cat_bn = "Audio files of Bengali music"

    print("Downloading English from Commons category:", cat_en)
    en_got = grab(cat_en, out_en, n_each, "EN")

    print("Downloading Bangla/Bengali from Commons category:", cat_bn)
    bn_got = grab(cat_bn, out_bn, n_each, "BN")

    print(f"Done. Downloaded EN={en_got} BN={bn_got}")
    print("Folders:")
    print(" -", out_en)
    print(" -", out_bn)


if __name__ == "__main__":
    main()
