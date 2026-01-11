#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import whisper  # pip install openai-whisper


def find_audio_file(audio_root: Path, filename: str) -> Optional[Path]:
    """Find filename under audio_root (direct or recursive)."""
    cand = audio_root / filename
    if cand.exists() and cand.is_file():
        return cand

    hits = list(audio_root.rglob(filename))
    hits = [h for h in hits if h.is_file()]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(
        description="Medium task: extract fixed-size log-mel spectrogram tensors for Conv-VAE"
    )
    ap.add_argument(
        "--meta_csv",
        type=str,
        default="results_medium/embeddings/lyrics_meta.csv",
        help="CSV with filename, language (uses this ordering).",
    )
    ap.add_argument(
        "--audio_root",
        type=str,
        default="data/audio",
        help="Root folder containing audio files (subfolders allowed).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="results_medium/audio_features",
        help="Output directory.",
    )
    ap.add_argument(
        "--seconds",
        type=float,
        default=30.0,
        help="Fixed duration per track in seconds (trim/pad).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they already exist.",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    meta_csv = (project_root / args.meta_csv).resolve()
    audio_root = (project_root / args.audio_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_x = out_dir / "X_aud_logmel.npy"
    out_meta = out_dir / "audio_meta.csv"

    if (out_x.exists() or out_meta.exists()) and not args.overwrite:
        raise RuntimeError(
            f"Outputs already exist:\n  {out_x}\n  {out_meta}\n"
            "Run again with --overwrite to regenerate."
        )

    if not meta_csv.exists():
        raise RuntimeError(
            f"Meta CSV not found: {meta_csv}\n"
            "Run `python3 src/lyrics_embed.py` first."
        )

    df = pd.read_csv(meta_csv)
    if "filename" not in df.columns:
        raise RuntimeError("meta_csv must contain a 'filename' column.")
    if "language" not in df.columns:
        df["language"] = ""

    filenames = df["filename"].astype(str).tolist()
    languages = df["language"].astype(str).tolist()

    # Whisper uses 16kHz internally
    sr = 16000
    target_samples = int(sr * float(args.seconds))

    feats = []
    rows = []

    for i, (fn, lang) in enumerate(zip(filenames, languages), start=1):
        p = find_audio_file(audio_root, fn)
        if not p:
            raise RuntimeError(f"[MISS] Could not find audio for '{fn}' under {audio_root}")

        print(f"[{i}/{len(filenames)}] {fn} -> {p}")

        # Robust decoding via ffmpeg inside whisper
        audio = whisper.audio.load_audio(str(p))               # np.float32, 16kHz
        audio = whisper.audio.pad_or_trim(audio, target_samples)

        # log-mel spectrogram (80 mel bins)
        mel = whisper.log_mel_spectrogram(audio)               # torch.Tensor (80, frames)
        mel_np = mel.cpu().numpy().astype(np.float32)          # (80, T)

        feats.append(mel_np)
        rows.append(
            {
                "filename": fn,
                "language": lang,
                "audio_path": str(p),
                "sr": sr,
                "seconds": float(args.seconds),
                "mel_bins": int(mel_np.shape[0]),
                "frames": int(mel_np.shape[1]),
            }
        )

    X = np.stack(feats, axis=0)     # (N, 80, T)
    X = X[:, None, :, :]            # (N, 1, 80, T)  -> CNN-friendly (B,C,H,W)

    np.save(out_x, X)
    pd.DataFrame(rows).to_csv(out_meta, index=False)

    print("\nSaved:", out_x, "shape:", X.shape)
    print("Saved:", out_meta, "rows:", len(rows))


if __name__ == "__main__":
    main()
