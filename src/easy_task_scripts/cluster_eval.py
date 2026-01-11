from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score

SEED = 42

def eval_clustering(X, labels):
    # Silhouette needs at least 2 clusters and less than n_samples clusters
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return sil, ch

def main():
    project_root = Path(__file__).resolve().parents[2]
    emb_dir = project_root / "results_easy" / "embeddings"
    out_dir = project_root / "results_easy" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = np.load(emb_dir / "Z.npy").astype(np.float32)         # (N, z_dim)
    X = np.load(emb_dir / "X.npy").astype(np.float32)         # (N, 40)
    meta = pd.read_csv(emb_dir / "meta.csv")

    # Standardize X again for fair PCA baseline (same as training)
    mean = np.load(emb_dir / "X_mean.npy")
    std = np.load(emb_dir / "X_std.npy")
    Xn = (X - mean) / std

    results = []

    # Choose a simple k. Since you have 2 languages, k=2 is the natural choice.
    # (You can also try k=3..5 and report best, but easy task is fine with k=2.)
    for k in [2]:
        # --- VAE latent + KMeans ---
        km_z = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        lab_z = km_z.fit_predict(Z)
        sil_z, ch_z = eval_clustering(Z, lab_z)
        results.append({
            "method": "VAE+KMeans",
            "dim": Z.shape[1],
            "k": k,
            "silhouette": sil_z,
            "calinski_harabasz": ch_z
        })

        # --- PCA baseline + KMeans ---
        pca = PCA(n_components=Z.shape[1], random_state=SEED)
        Xp = pca.fit_transform(Xn)
        km_p = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        lab_p = km_p.fit_predict(Xp)
        sil_p, ch_p = eval_clustering(Xp, lab_p)
        results.append({
            "method": "PCA+KMeans",
            "dim": Z.shape[1],
            "k": k,
            "silhouette": sil_p,
            "calinski_harabasz": ch_p
        })

        # Save cluster labels for later plotting
        meta_out = meta.copy()
        meta_out["cluster_vae"] = lab_z
        meta_out["cluster_pca"] = lab_p
        meta_out.to_csv(out_dir / "clusters.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "metrics.csv", index=False)

    print(df.to_string(index=False))
    print("Saved:", out_dir / "metrics.csv")
    print("Saved:", out_dir / "clusters.csv")

if __name__ == "__main__":
    main()
