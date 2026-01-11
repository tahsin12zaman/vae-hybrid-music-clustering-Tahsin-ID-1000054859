#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def find_project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent
        if (parent / "requirements.txt").exists():
            return parent
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find project root.")


def main():
    ap = argparse.ArgumentParser(description="Create/validate metadata file with genre for hard task.")
    ap.add_argument("--lyrics_csv", type=str, default="data/lyrics/lyrics.csv",
                    help="Input lyrics CSV with filename/language/lyrics columns.")
    ap.add_argument("--genre_csv", type=str, default="",
                    help="Optional CSV containing at least filename,genre (will be merged).")
    ap.add_argument("--out_csv", type=str, default="data/meta_with_genre.csv",
                    help="Output metadata CSV (filename,language,genre).")
    ap.add_argument("--overwrite_lyrics", action="store_true",
                    help="If set, also writes genre column back into lyrics_csv file.")
    args = ap.parse_args()

    project_root = find_project_root()
    lyrics_path = (project_root / args.lyrics_csv).resolve()
    out_path = (project_root / args.out_csv).resolve()

    if not lyrics_path.exists():
        raise FileNotFoundError(f"Missing: {lyrics_path}")

    df = pd.read_csv(lyrics_path)

    # Ensure required columns exist
    if "filename" not in df.columns or "language" not in df.columns:
        raise RuntimeError("lyrics.csv must contain 'filename' and 'language' columns.")

    if "genre" not in df.columns:
        # create empty genre column after language if possible
        insert_at = list(df.columns).index("language") + 1
        df.insert(insert_at, "genre", "")

    # Optional: merge external genre CSV by filename
    if args.genre_csv.strip():
        genre_path = (project_root / args.genre_csv).resolve()
        if not genre_path.exists():
            raise FileNotFoundError(f"Missing: {genre_path}")

        g = pd.read_csv(genre_path)
        if "filename" not in g.columns or "genre" not in g.columns:
            raise RuntimeError("--genre_csv must contain columns: filename, genre")

        # merge and prefer genre_csv values when present
        df = df.merge(g[["filename", "genre"]].rename(columns={"genre": "genre_from_file"}),
                      on="filename", how="left")
        df["genre_from_file"] = df["genre_from_file"].fillna("").astype(str).str.strip()

        # If external has genre, overwrite blank genres
        cur = df["genre"].fillna("").astype(str).str.strip()
        df["genre"] = cur.where(cur != "", df["genre_from_file"])
        df = df.drop(columns=["genre_from_file"])

    # Build meta output
    meta = df[["filename", "language", "genre"]].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_path, index=False)

    # Optionally update lyrics CSV in place (adds genre column)
    if args.overwrite_lyrics:
        df.to_csv(lyrics_path, index=False)
        print(f"Updated lyrics CSV with genre column: {lyrics_path}")

    # Summary
    genre_series = meta["genre"].fillna("").astype(str).str.strip()
    missing = int((genre_series == "").sum())
    uniq = sorted([g for g in genre_series.unique().tolist() if g])

    print("\nDone.")
    print("Input lyrics:", lyrics_path)
    if args.genre_csv.strip():
        print("Merged genre:", (project_root / args.genre_csv).resolve())
    print("Output meta :", out_path)
    print(f"Rows       : {len(meta)}")
    print(f"Missing genre rows: {missing}")
    print(f"Unique genres ({len(uniq)}): {uniq}")

    if missing > 0:
        print("\nNext step: fill the 'genre' column.")
        print(f"Edit either:\n- {out_path}\n(or run again later with --overwrite_lyrics to embed genre into lyrics.csv)")


if __name__ == "__main__":
    main()
