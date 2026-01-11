#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

AUDIO_EXTS = {".ogg", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".opus"}

def safe_stem(name: str) -> str:
    s = Path(name).stem
    s = re.sub(r"[^\w\-\. ]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:180] if len(s) > 180 else s

def looks_like_filename(s: str) -> bool:
    s = (s or "").strip().lower()
    return any(s.endswith(ext) for ext in AUDIO_EXTS)

def find_audio(filename: str, audio_roots: list[Path]) -> Optional[Path]:
    for root in audio_roots:
        cand = root / filename
        if cand.exists() and cand.is_file():
            return cand
    for root in audio_roots:
        if root.exists():
            hits = list(root.rglob(filename))
            hits = [h for h in hits if h.is_file()]
            if hits:
                return hits[0]
    return None

def transcribe_whisper(audio_path: Path, model_name: str, language: Optional[str]) -> str:
    import whisper  # from openai-whisper
    model = whisper.load_model(model_name)
    kwargs = {}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(audio_path), **kwargs)
    return (result.get("text") or "").strip()

def main():
    ap = argparse.ArgumentParser(description="Build lyrics.csv by transcribing audio files with Whisper.")
    ap.add_argument("--in_csv", type=str, default="data/lyrics/lyrics.csv")
    ap.add_argument("--out_csv", type=str, default=None, help="Default: overwrite input.")
    ap.add_argument("--audio_dir", action="append", default=["data/audio"],
                    help="Root folder(s) to search for audio. Can pass multiple.")
    ap.add_argument("--transcript_dir", type=str, default="data/lyrics/asr",
                    help="Where to save per-file transcripts as .txt")
    ap.add_argument("--model", type=str, default="small",
                    help="Whisper model: tiny/base/small/medium/large")
    ap.add_argument("--force", action="store_true",
                    help="Re-transcribe even if lyrics already filled (and not filename-like).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional limit for debugging (0 = no limit)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    in_csv = (project_root / args.in_csv).resolve()
    out_csv = (project_root / (args.out_csv or args.in_csv)).resolve()

    audio_roots = [(project_root / p).resolve() for p in args.audio_dir]
    transcript_dir = (project_root / args.transcript_dir).resolve()
    transcript_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "filename" not in df.columns:
        raise RuntimeError("Input CSV must have a 'filename' column.")
    if "lyrics" not in df.columns:
        df["lyrics"] = ""
    if "language" not in df.columns:
        df["language"] = ""

    done = 0
    skipped = 0
    failed = 0

    for idx, row in df.iterrows():
        if args.limit and done >= args.limit:
            break

        filename = str(row["filename"]).strip()
        cur = str(row.get("lyrics", "") or "")
        lang = str(row.get("language", "") or "").strip().lower()

        needs = args.force or (not cur.strip()) or looks_like_filename(cur)
        if not needs:
            skipped += 1
            continue

        audio_path = find_audio(filename, audio_roots)
        if not audio_path:
            print(f"[MISS] audio not found for: {filename}")
            failed += 1
            df.at[idx, "lyrics"] = ""
            continue

        whisper_lang = "en" if lang in {"english", "en"} else None

        print(f"[ASR] {filename}  ->  {audio_path}")
        try:
            text = transcribe_whisper(audio_path, args.model, whisper_lang)
            if not text:
                print(f"[WARN] empty transcript: {filename}")
                failed += 1
                df.at[idx, "lyrics"] = ""
                continue

            df.at[idx, "lyrics"] = text
            (transcript_dir / f"{safe_stem(filename)}.txt").write_text(text, encoding="utf-8")
            done += 1
        except Exception as e:
            print(f"[FAIL] {filename}: {e}")
            failed += 1
            df.at[idx, "lyrics"] = ""

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    missing = (df["lyrics"].fillna("").astype(str).str.strip() == "").sum()

    print("\nDone.")
    print(f"Input:  {in_csv}")
    print(f"Output: {out_csv}")
    print(f"Transcribed: {done}")
    print(f"Skipped (already had lyrics): {skipped}")
    print(f"Failed/missing: {failed}")
    print(f"Still missing lyrics rows: {missing}")
    print(f"Saved transcripts to: {transcript_dir}")

if __name__ == "__main__":
    main()
