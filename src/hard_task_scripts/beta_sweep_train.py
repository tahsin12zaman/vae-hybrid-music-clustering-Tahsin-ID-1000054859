#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


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
    ap = argparse.ArgumentParser("Beta sweep runner (calls the conv trainer repeatedly)")
    ap.add_argument("--trainer", type=str, default="src/medium_task_scripts/train_conv_vae_audio_medium_task.py")
    ap.add_argument("--betas", type=str, default="0,0.1,1,4,10")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl_warmup", type=int, default=100)
    ap.add_argument("--out_root", type=str, default="results_hard/beta_sweep")
    args = ap.parse_args()

    root = find_project_root()
    trainer = (root / args.trainer).resolve()
    if not trainer.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer}")

    out_root = (root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    betas = [b.strip() for b in args.betas.split(",") if b.strip()]
    for b in betas:
        out_dir = out_root / f"beta{b}"
        cmd = [
            "python3", str(trainer),
            "--beta", str(b),
            "--epochs", str(args.epochs),
            "--latent_dim", str(args.latent_dim),
            "--lr", str(args.lr),
            "--kl_warmup", str(args.kl_warmup),
            "--out_dir", str(out_dir),
        ]
        print("\n>>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nDone. Outputs in:", out_root)


if __name__ == "__main__":
    main()
