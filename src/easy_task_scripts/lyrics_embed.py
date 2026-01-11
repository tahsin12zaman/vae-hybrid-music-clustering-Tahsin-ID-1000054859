from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

SEED = 42
LYR_DIM = 64  # output lyric embedding dimension

def main():
    project_root = Path(__file__).resolve().parents[2]
    lyr_path = project_root / "data" / "lyrics" / "lyrics.csv"
    out_dir = project_root / "results_medium" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(lyr_path)
    if "lyrics" not in df.columns:
        raise RuntimeError("lyrics.csv must have a 'lyrics' column")

    #texts = df["lyrics"].fillna("").astype(str).tolist()
    texts = df["lyrics"].fillna("").astype(str).tolist()
    meta = df[["filename", "language"]]
    if all(len(t.strip()) == 0 for t in texts):
        raise RuntimeError("All lyrics are empty. Fill the 'lyrics' column first.")

    # Multilingual-friendly: character ngrams
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    X = vec.fit_transform(texts)

    # Reduce to dense vectors
    dim = min(LYR_DIM, X.shape[1] - 1) if X.shape[1] > 1 else 1
    svd = TruncatedSVD(n_components=dim, random_state=SEED)
    Z = svd.fit_transform(X).astype(np.float32)

    np.save(out_dir / "Z_lyr.npy", Z)
    df[["filename", "language"]].to_csv(out_dir / "lyrics_meta.csv", index=False)

    print("Saved:", out_dir / "Z_lyr.npy", "shape:", Z.shape)
    print("Saved:", out_dir / "lyrics_meta.csv", "rows:", len(df))

if __name__ == "__main__":
    main()
