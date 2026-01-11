from pathlib import Path
import numpy as np
import pandas as pd
import librosa

SR = 22050
N_MFCC = 20
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

def mfcc_stats(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)

def collect_files(audio_root: Path):
    rows = []
    for lang in ["english", "bangla"]:
        folder = audio_root / lang
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
                rows.append((p, lang))
    return rows

def main():
    project_root = Path(__file__).resolve().parents[2]
    audio_root = project_root / "data" / "audio"
    out_dir = project_root / "results_easy" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(audio_root)
    if not files:
        raise RuntimeError(f"No audio files found under: {audio_root}")

    X, meta = [], []
    for path, lang in files:
        y, sr = librosa.load(path, sr=SR, mono=True)
        X.append(mfcc_stats(y, sr))
        meta.append({"file": str(path.relative_to(project_root)), "language": lang})

    X = np.vstack(X).astype(np.float32)  # (N, 40)
    np.save(out_dir / "X.npy", X)
    pd.DataFrame(meta).to_csv(out_dir / "meta.csv", index=False)

    print("Saved X:", out_dir / "X.npy", "shape:", X.shape)
    print("Saved meta:", out_dir / "meta.csv", "rows:", len(meta))

if __name__ == "__main__":
    main()
