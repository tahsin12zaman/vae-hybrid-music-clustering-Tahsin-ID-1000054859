#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# -------------------------
# Utilities
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


def encode_labels(series: pd.Series) -> np.ndarray:
    vals = series.fillna("").astype(str).str.strip().tolist()
    uniq = sorted(set(vals))
    m = {v: i for i, v in enumerate(uniq)}
    return np.array([m[v] for v in vals], dtype=int)


def one_hot(series: pd.Series) -> np.ndarray:
    vals = series.fillna("").astype(str).str.strip()
    uniq = sorted(vals.unique().tolist())
    m = {v: i for i, v in enumerate(uniq)}
    Y = np.zeros((len(vals), len(uniq)), dtype=np.float32)
    for i, v in enumerate(vals.tolist()):
        Y[i, m[v]] = 1.0
    return Y


def align_by_filename(
    target_filenames: List[str],
    source_filenames: List[str],
    X: np.ndarray,
    name: str,
) -> np.ndarray:
    # normalize filenames (strip only; do NOT change case because your filenames may be non-ascii)
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


def purity_score(y_true: np.ndarray, y_pred: np.ndarray, drop_noise: bool = False) -> Optional[float]:
    if drop_noise:
        mask = y_pred != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]

    if y_true.size == 0:
        return None

    # if only 1 cluster, purity is the max class fraction
    clusters = np.unique(y_pred)
    total = 0
    for c in clusters:
        idx = np.where(y_pred == c)[0]
        if idx.size == 0:
            continue
        true_labels = y_true[idx]
        counts = np.bincount(true_labels)
        total += counts.max() if counts.size else 0
    return float(total / len(y_true))


def safe_internal_scores(X: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Silhouette + Davies-Bouldin.
    For DBSCAN we drop noise points (-1) for these scores.
    """
    out = {"silhouette": None, "davies_bouldin": None}

    mask = y_pred != -1
    X2 = X[mask]
    y2 = y_pred[mask]

    uniq = np.unique(y2)
    if len(uniq) < 2:
        return out
    if X2.shape[0] <= len(uniq):
        return out

    try:
        out["silhouette"] = float(silhouette_score(X2, y2))
    except Exception:
        out["silhouette"] = None

    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X2, y2))
    except Exception:
        out["davies_bouldin"] = None

    return out


def label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drop_noise: bool = False,
) -> Dict[str, Optional[float]]:
    if drop_noise:
        mask = y_pred != -1
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_true.size == 0:
            return {"ARI": None, "NMI": None, "purity": None}

    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "purity": purity_score(y_true, y_pred, drop_noise=False),
    }


def run_clusterers(
    X: np.ndarray,
    k: int,
    seed: int,
    dbscan_eps_list: List[float],
    dbscan_min_samples: int,
) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    preds["kmeans"] = km.fit_predict(X)

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    preds["agglomerative_ward"] = agg.fit_predict(X)

    for eps in dbscan_eps_list:
        db = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
        preds[f"dbscan_eps{eps:g}"] = db.fit_predict(X)

    return preds


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Hard task: multimodal clustering + metrics (Sil/DB/ARI/NMI/Purity)")
    ap.add_argument("--seed", type=int, default=42)

    # inputs
    ap.add_argument("--meta_csv", type=str, default="data/lyrics/lyrics.csv",
                    help="Must contain filename, language, genre columns.")
    ap.add_argument("--results_medium_dir", type=str, default="results_medium")

    ap.add_argument("--z_audio_path", type=str, default="results_medium/conv_vae_audio/z_audio_mu.npy")
    ap.add_argument("--z_lyr_path", type=str, default="results_medium/embeddings/Z_lyr.npy")
    ap.add_argument("--x_mel_path", type=str, default="results_medium/audio_features/X_aud_logmel.npy")

    ap.add_argument("--audio_meta_csv", type=str, default="results_medium/audio_features/audio_meta.csv")
    ap.add_argument("--lyrics_meta_csv", type=str, default="results_medium/embeddings/lyrics_meta.csv")

    # optional AE baseline latent (train a conv autoencoder by running your VAE trainer with beta=0 and saving z)
    ap.add_argument("--ae_latent_path", type=str, default="",
                    help="Optional path to AE latent z_mu.npy (same N order as audio_meta.csv).")

    # clustering config
    ap.add_argument("--k", type=int, default=0,
                    help="k for kmeans/agglo. 0 = use #genres (recommended for hard task).")
    ap.add_argument("--dbscan_eps", type=str, default="0.5,1.0,1.5")
    ap.add_argument("--dbscan_min_samples", type=int, default=3)

    # multimodal controls
    ap.add_argument("--alphas", type=str, default="1,0.3,0.1,0.03,0.01",
                    help="Comma-separated audio weights for hybrid concat: [alpha*z_audio, Z_lyr].")
    ap.add_argument("--include_genre_feature", action="store_true",
                    help="If set, also build X = [alpha*z_audio, Z_lyr, gamma*onehot(genre)].")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="Weight for one-hot genre feature when --include_genre_feature is used.")

    # outputs
    ap.add_argument("--out_dir", type=str, default="results_hard/clustering")

    args = ap.parse_args()
    np.random.seed(args.seed)

    root = find_project_root()

    meta_path = (root / args.meta_csv).resolve()
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    meta = pd.read_csv(meta_path)
    for col in ["filename", "language", "genre"]:
        if col not in meta.columns:
            raise RuntimeError(f"{meta_path} must contain column: {col}")

    target_filenames = meta["filename"].astype(str).str.strip().tolist()
    y_lang = encode_labels(meta["language"])
    y_genre = encode_labels(meta["genre"])
    genre_oh = one_hot(meta["genre"])  # for optional feature use

    # k default: number of genres
    n_genres = len(meta["genre"].astype(str).str.strip().unique())
    k = args.k if args.k > 0 else max(2, n_genres)

    dbscan_eps_list = [float(x.strip()) for x in args.dbscan_eps.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    # Load arrays + their metas for alignment
    z_audio_path = (root / args.z_audio_path).resolve()
    z_lyr_path = (root / args.z_lyr_path).resolve()
    x_mel_path = (root / args.x_mel_path).resolve()

    audio_meta_path = (root / args.audio_meta_csv).resolve()
    lyrics_meta_path = (root / args.lyrics_meta_csv).resolve()

    for p in [z_audio_path, z_lyr_path, x_mel_path, audio_meta_path, lyrics_meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    Z_audio_raw = np.load(z_audio_path).astype(np.float32)  # (N, d)
    Z_lyr_raw = np.load(z_lyr_path).astype(np.float32)      # (N, d)
    X_mel_raw = np.load(x_mel_path).astype(np.float32)      # (N,1,80,T)

    audio_meta = pd.read_csv(audio_meta_path)
    lyrics_meta = pd.read_csv(lyrics_meta_path)
    if "filename" not in audio_meta.columns:
        raise RuntimeError("audio_meta.csv must contain filename column")
    if "filename" not in lyrics_meta.columns:
        raise RuntimeError("lyrics_meta.csv must contain filename column")

    # Align everything to meta.csv filename order
    audio_filenames = audio_meta["filename"].astype(str).str.strip().tolist()
    lyr_filenames = lyrics_meta["filename"].astype(str).str.strip().tolist()

    Z_audio = align_by_filename(target_filenames, audio_filenames, Z_audio_raw, "z_audio")
    X_mel = align_by_filename(target_filenames, audio_filenames, X_mel_raw, "x_mel")
    Z_lyr = align_by_filename(target_filenames, lyr_filenames, Z_lyr_raw, "z_lyr")

    # A) Direct spectral baseline: mean over time -> (N,80)
    X_meanmel = X_mel.mean(axis=-1).squeeze(1)  # (N, 80)

    # B) PCA + KMeans baseline (on meanmel; keep dims small due to N=29)
    pca_dim = min(16, X_meanmel.shape[0] - 1, X_meanmel.shape[1])
    X_pca = PCA(n_components=pca_dim, random_state=args.seed).fit_transform(StandardScaler().fit_transform(X_meanmel))

    # C) VAE/Beta-VAE latent
    X_audio_lat = Z_audio  # (N, d_lat)

    # D) Lyrics embedding
    X_lyrics = Z_lyr

    # Optional AE latent
    X_ae_lat = None
    if args.ae_latent_path.strip():
        ae_path = (root / args.ae_latent_path).resolve()
        if not ae_path.exists():
            raise FileNotFoundError(f"ae_latent_path not found: {ae_path}")
        AE_raw = np.load(ae_path).astype(np.float32)
        X_ae_lat = align_by_filename(target_filenames, audio_filenames, AE_raw, "ae_latent")

    # Build representations
    reps: Dict[str, np.ndarray] = {
        "spectral_meanmel": X_meanmel,
        f"spectral_pca{pca_dim}": X_pca,
        "lyrics_embed": X_lyrics,
        "audio_betaVAE_latent": X_audio_lat,
    }
    if X_ae_lat is not None:
        reps["audio_autoencoder_latent"] = X_ae_lat

    # Hybrid: [alpha * audio_lat, lyrics]
    for a in alphas:
        reps[f"hybrid_alpha{a:g}_audio+lyrics"] = np.concatenate([a * X_audio_lat, X_lyrics], axis=1)

        if args.include_genre_feature:
            reps[f"multimodal_alpha{a:g}_audio+lyrics+genreOH_gamma{args.gamma:g}"] = np.concatenate(
                [a * X_audio_lat, X_lyrics, args.gamma * genre_oh],
                axis=1
            )

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict] = {}

    for rep_name, X in reps.items():
        # standardize features per representation
        Xs = StandardScaler().fit_transform(X)

        preds = run_clusterers(
            X=Xs,
            k=k,
            seed=args.seed,
            dbscan_eps_list=dbscan_eps_list,
            dbscan_min_samples=args.dbscan_min_samples,
        )

        rep_res: Dict[str, Dict] = {}
        for method, y_pred in preds.items():
            internal = safe_internal_scores(Xs, y_pred)

            # label metrics for language + genre (and also a "drop_noise" version for DBSCAN)
            m_lang = label_metrics(y_lang, y_pred, drop_noise=False)
            m_genre = label_metrics(y_genre, y_pred, drop_noise=False)

            # also compute drop-noise variants (meaningful mainly for DBSCAN)
            m_lang_dn = label_metrics(y_lang, y_pred, drop_noise=True)
            m_genre_dn = label_metrics(y_genre, y_pred, drop_noise=True)

            n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
            rep_res[method] = {
                "k": int(k) if "kmeans" in method or "agglomerative" in method else None,
                "n_clusters_found": int(n_clusters),
                "n_noise": int(np.sum(y_pred == -1)),

                # internal
                **internal,

                # language label metrics
                "ARI_language": m_lang["ARI"],
                "NMI_language": m_lang["NMI"],
                "purity_language": m_lang["purity"],
                "ARI_language_dropNoise": m_lang_dn["ARI"],
                "NMI_language_dropNoise": m_lang_dn["NMI"],
                "purity_language_dropNoise": m_lang_dn["purity"],

                # genre label metrics
                "ARI_genre": m_genre["ARI"],
                "NMI_genre": m_genre["NMI"],
                "purity_genre": m_genre["purity"],
                "ARI_genre_dropNoise": m_genre_dn["ARI"],
                "NMI_genre_dropNoise": m_genre_dn["NMI"],
                "purity_genre_dropNoise": m_genre_dn["purity"],
            }

        all_results[rep_name] = rep_res

    # Save JSON
    json_path = out_dir / "metrics_hard.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("Saved:", json_path)

    # Save CSV
    rows = []
    for rep, methods in all_results.items():
        for method, m in methods.items():
            rows.append({"rep": rep, "method": method, **m})
    df = pd.DataFrame(rows)
    csv_path = out_dir / "metrics_hard.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # Quick “best” summaries (by ARI_genre then NMI_genre, since genre is hard-task focus)
    print("\n=== Quick best per representation (prefers ARI_genre, then NMI_genre) ===")
    df2 = df.copy()
    df2["ARI_genre"] = pd.to_numeric(df2["ARI_genre"], errors="coerce")
    df2["NMI_genre"] = pd.to_numeric(df2["NMI_genre"], errors="coerce")
    for rep in reps.keys():
        sub = df2[df2["rep"] == rep].copy()
        sub = sub.sort_values(["ARI_genre", "NMI_genre"], ascending=False)
        top = sub.iloc[0].to_dict()
        print(
            f"- {rep}: {top['method']} | "
            f"ARI_genre={top.get('ARI_genre')} NMI_genre={top.get('NMI_genre')} "
            f"sil={top.get('silhouette')} db={top.get('davies_bouldin')}"
        )


if __name__ == "__main__":
    main()
