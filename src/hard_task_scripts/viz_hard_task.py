#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


# -------------------------
# Path / alignment helpers
# -------------------------
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


def align_by_filename(
    target_filenames: List[str],
    source_filenames: List[str],
    X: np.ndarray,
    name: str,
) -> np.ndarray:
    tgt = [str(x).strip() for x in target_filenames]
    src = [str(x).strip() for x in source_filenames]

    index = {}
    for i, fn in enumerate(src):
        if fn in index:
            raise RuntimeError(f"Duplicate filename in {name}: {fn!r}")
        index[fn] = i

    missing = [fn for fn in tgt if fn not in index]
    if missing:
        ex = "\n".join(missing[:10])
        raise RuntimeError(
            f"[{name}] Missing {len(missing)} filenames when aligning.\n"
            f"First missing examples:\n{ex}"
        )

    order = [index[fn] for fn in tgt]
    return X[order]


# -------------------------
# Plot helpers
# -------------------------
def _scatter_by_category(
    X2: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    max_legend: int = 30,
):
    plt.figure(figsize=(8, 6))
    labels = [str(x) for x in labels]
    uniq = sorted(set(labels))

    # map categories to integers
    m = {u: i for i, u in enumerate(uniq)}
    y = np.array([m[x] for x in labels], dtype=int)

    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=45)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")

    # legend (limited)
    if len(uniq) <= max_legend:
        handles = []
        for u in uniq:
            idx = m[u]
            handles.append(plt.Line2D([], [], marker="o", linestyle="", label=u))
        plt.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _heatmap_counts(
    counts: pd.DataFrame,
    title: str,
    out_path: Path,
):
    plt.figure(figsize=(8, 5))
    plt.imshow(counts.values, aspect="auto", origin="lower")
    plt.title(title)
    plt.xticks(range(len(counts.columns)), counts.columns, rotation=45, ha="right")
    plt.yticks(range(len(counts.index)), counts.index)
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _show_mel(mel_2d: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_2d, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("mel bins")
    plt.colorbar(label="value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -------------------------
# Clustering (for viz)
# -------------------------
def parse_best_methods(metrics_csv: Path) -> Dict[str, str]:
    """
    Pick best method per rep by (ARI_genre, NMI_genre).
    """
    df = pd.read_csv(metrics_csv)
    df["ARI_genre"] = pd.to_numeric(df.get("ARI_genre"), errors="coerce")
    df["NMI_genre"] = pd.to_numeric(df.get("NMI_genre"), errors="coerce")

    best: Dict[str, str] = {}
    for rep in df["rep"].unique():
        sub = df[df["rep"] == rep].copy()
        sub = sub.sort_values(["ARI_genre", "NMI_genre"], ascending=False)
        best[rep] = str(sub.iloc[0]["method"])
    return best


def cluster_assignments(
    X: np.ndarray,
    method: str,
    k: int,
    seed: int,
    dbscan_min_samples: int,
) -> np.ndarray:
    """
    Re-run clustering to get y_pred for plots.
    method is one of: kmeans, agglomerative_ward, dbscan_eps0.5, etc.
    """
    if method == "kmeans":
        return KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X)
    if method == "agglomerative_ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)

    if method.startswith("dbscan_eps"):
        # method like "dbscan_eps0.5"
        eps_str = method.replace("dbscan_eps", "")
        eps = float(eps_str)
        return DBSCAN(eps=eps, min_samples=dbscan_min_samples).fit_predict(X)

    # fallback
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


# -------------------------
# 2D embedding
# -------------------------
def embed_2d(X: np.ndarray, seed: int, mode: str = "tsne") -> np.ndarray:
    """
    Return 2D embedding. For tiny datasets, PCA->tSNE is stable.
    """
    Xs = StandardScaler().fit_transform(X)

    if mode == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(Xs)

    # TSNE: pick a safe perplexity based on N
    n = Xs.shape[0]
    perplexity = max(2, min(10, (n - 1) // 3))
    Xp = PCA(n_components=min(10, Xs.shape[1], n - 1), random_state=seed).fit_transform(Xs)

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(Xp)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Hard task visualizations (latent plots + cluster distributions + recon examples)")

    ap.add_argument("--meta_csv", type=str, default="data/lyrics/lyrics.csv")
    ap.add_argument("--audio_meta_csv", type=str, default="results_medium/audio_features/audio_meta.csv")
    ap.add_argument("--lyrics_meta_csv", type=str, default="results_medium/embeddings/lyrics_meta.csv")

    ap.add_argument("--x_mel_path", type=str, default="results_medium/audio_features/X_aud_logmel.npy")
    ap.add_argument("--z_beta_path", type=str, default="results_medium/conv_vae_audio/z_audio_mu.npy")
    ap.add_argument("--z_ae_path", type=str, default="results_hard/ae_audio/z_audio_mu.npy")
    ap.add_argument("--z_lyr_path", type=str, default="results_medium/embeddings/Z_lyr.npy")

    ap.add_argument("--metrics_csv", type=str, default="results_hard/clustering/metrics_hard.csv")
    ap.add_argument("--out_dir", type=str, default="results_hard/plots")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embed_mode", type=str, default="tsne", choices=["tsne", "pca"])

    ap.add_argument("--alpha", type=float, default=1.0, help="Hybrid weight: [alpha*z_audio, Z_lyr].")
    ap.add_argument("--dbscan_min_samples", type=int, default=3)

    args = ap.parse_args()
    root = find_project_root()

    meta_path = (root / args.meta_csv).resolve()
    audio_meta_path = (root / args.audio_meta_csv).resolve()
    lyrics_meta_path = (root / args.lyrics_meta_csv).resolve()

    x_mel_path = (root / args.x_mel_path).resolve()
    z_beta_path = (root / args.z_beta_path).resolve()
    z_ae_path = (root / args.z_ae_path).resolve()
    z_lyr_path = (root / args.z_lyr_path).resolve()

    metrics_path = (root / args.metrics_csv).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [meta_path, audio_meta_path, lyrics_meta_path, x_mel_path, z_beta_path, z_ae_path, z_lyr_path, metrics_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    meta = pd.read_csv(meta_path)
    for col in ["filename", "language", "genre"]:
        if col not in meta.columns:
            raise RuntimeError(f"{meta_path} must contain columns filename, language, genre")

    target_filenames = meta["filename"].astype(str).str.strip().tolist()
    language = meta["language"].astype(str).str.strip().tolist()
    genre = meta["genre"].astype(str).str.strip().tolist()

    # k for cluster-distribution plots (same default as clustering script)
    n_genres = len(sorted(set(genre)))
    k = max(2, n_genres)

    audio_meta = pd.read_csv(audio_meta_path)
    lyrics_meta = pd.read_csv(lyrics_meta_path)
    if "filename" not in audio_meta.columns or "filename" not in lyrics_meta.columns:
        raise RuntimeError("audio_meta.csv and lyrics_meta.csv must contain filename column")

    audio_filenames = audio_meta["filename"].astype(str).str.strip().tolist()
    lyr_filenames = lyrics_meta["filename"].astype(str).str.strip().tolist()

    X_mel_raw = np.load(x_mel_path).astype(np.float32)     # (N,1,80,T)
    Z_beta_raw = np.load(z_beta_path).astype(np.float32)   # (N,d)
    Z_ae_raw = np.load(z_ae_path).astype(np.float32)       # (N,d)
    Z_lyr_raw = np.load(z_lyr_path).astype(np.float32)     # (N,d)

    # align everything to meta order
    X_mel = align_by_filename(target_filenames, audio_filenames, X_mel_raw, "X_mel")
    Z_beta = align_by_filename(target_filenames, audio_filenames, Z_beta_raw, "Z_beta")
    Z_ae = align_by_filename(target_filenames, audio_filenames, Z_ae_raw, "Z_ae")
    Z_lyr = align_by_filename(target_filenames, lyr_filenames, Z_lyr_raw, "Z_lyr")

    # representations to plot
    reps: Dict[str, np.ndarray] = {
        "audio_betaVAE_latent": Z_beta,
        "audio_autoencoder_latent": Z_ae,
        "lyrics_embed": Z_lyr,
        f"hybrid_alpha{args.alpha:g}_audio+lyrics": np.concatenate([args.alpha * Z_beta, Z_lyr], axis=1),
    }

    # pick best clustering methods from metrics CSV
    best_methods = parse_best_methods(metrics_path)

    # 1) 2D latent plots (colored by genre/language/cluster)
    for rep_name, X in reps.items():
        X2 = embed_2d(X, seed=args.seed, mode=args.embed_mode)

        _scatter_by_category(
            X2, genre,
            title=f"{rep_name} ({args.embed_mode}) colored by genre",
            out_path=out_dir / f"{rep_name}_{args.embed_mode}_by_genre.png",
        )

        _scatter_by_category(
            X2, language,
            title=f"{rep_name} ({args.embed_mode}) colored by language",
            out_path=out_dir / f"{rep_name}_{args.embed_mode}_by_language.png",
        )

        # cluster assignment using best method from metrics (fallback to agglo)
        method = best_methods.get(rep_name, "agglomerative_ward")
        Xs = StandardScaler().fit_transform(X)
        y_pred = cluster_assignments(Xs, method=method, k=k, seed=args.seed, dbscan_min_samples=args.dbscan_min_samples)

        # nicer cluster labels (keep noise as -1)
        y_str = [f"c{c}" if c != -1 else "noise" for c in y_pred.tolist()]
        _scatter_by_category(
            X2, y_str,
            title=f"{rep_name} ({args.embed_mode}) colored by clusters ({method})",
            out_path=out_dir / f"{rep_name}_{args.embed_mode}_by_cluster_{method}.png",
            max_legend=60,
        )

        # 2) Cluster distribution heatmaps
        dfc = pd.DataFrame({
            "cluster": y_str,
            "genre": genre,
            "language": language,
        })

        c_genre = pd.crosstab(dfc["cluster"], dfc["genre"])
        c_lang = pd.crosstab(dfc["cluster"], dfc["language"])

        _heatmap_counts(
            c_genre,
            title=f"{rep_name}: cluster x genre ({method}) counts",
            out_path=out_dir / f"{rep_name}_cluster_vs_genre_{method}.png",
        )
        _heatmap_counts(
            c_lang,
            title=f"{rep_name}: cluster x language ({method}) counts",
            out_path=out_dir / f"{rep_name}_cluster_vs_language_{method}.png",
        )

    # 3) Reconstruction examples (uses recon_sample.npy files you already have)
    # We’ll compare mel[0] with recon_sample from betaVAE + AE if they exist.
    orig0 = X_mel[0, 0]  # (80,T)
    _show_mel(orig0, "Original log-mel (sample 0)", out_dir / "orig_logmel_sample0.png")

    # betaVAE recon sample (from results_medium)
    beta_recon_path = (root / "results_medium/conv_vae_audio/recon_sample.npy").resolve()
    if beta_recon_path.exists():
        beta_recon = np.load(beta_recon_path).astype(np.float32)[0, 0]
        _show_mel(beta_recon, "Beta-VAE recon_sample (sample 0)", out_dir / "recon_betaVAE_sample0.png")

    # AE recon sample (from results_hard/ae_audio)
    ae_recon_path = (root / "results_hard/ae_audio/recon_sample.npy").resolve()
    if ae_recon_path.exists():
        ae_recon = np.load(ae_recon_path).astype(np.float32)[0, 0]
        _show_mel(ae_recon, "Autoencoder recon_sample (sample 0)", out_dir / "recon_AE_sample0.png")

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()
