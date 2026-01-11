#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score


def _safe_scores(X: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Compute Silhouette + Davies-Bouldin safely.
    For DBSCAN, we ignore noise (-1) for these metrics.
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


def _encode_labels(labels: pd.Series) -> np.ndarray:
    uniq = sorted(labels.astype(str).unique().tolist())
    m = {l: i for i, l in enumerate(uniq)}
    return np.array([m[str(x)] for x in labels.astype(str).tolist()], dtype=int)


def run_all_clusterers(
    X: np.ndarray,
    y_true: np.ndarray,
    k: int,
    seed: int,
    dbscan_eps_list: list[float],
    dbscan_min_samples: int,
) -> Dict[str, Dict]:
    results = {}

    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    y = km.fit_predict(X)
    results["kmeans"] = {
        "k": int(k),
        "ARI": float(adjusted_rand_score(y_true, y)),
        **_safe_scores(X, y),
        "n_clusters_found": int(len(np.unique(y))),
    }

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    y = agg.fit_predict(X)
    results["agglomerative_ward"] = {
        "k": int(k),
        "ARI": float(adjusted_rand_score(y_true, y)),
        **_safe_scores(X, y),
        "n_clusters_found": int(len(np.unique(y))),
    }

    for eps in dbscan_eps_list:
        db = DBSCAN(eps=eps, min_samples=dbscan_min_samples)
        y = db.fit_predict(X)
        n_clusters = len(set(y)) - (1 if -1 in y else 0)
        results[f"dbscan_eps{eps:g}"] = {
            "eps": float(eps),
            "min_samples": int(dbscan_min_samples),
            "ARI": float(adjusted_rand_score(y_true, y)),
            **_safe_scores(X, y),
            "n_clusters_found": int(n_clusters),
            "n_noise": int(np.sum(y == -1)),
        }

    return results


def main():
    ap = argparse.ArgumentParser("Medium task: hybrid clustering + evaluation (with proper hybrid weighting)")
    ap.add_argument("--results_dir", type=str, default="results_medium")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=0, help="Clusters for kmeans/agglo. 0 = auto from #languages.")
    ap.add_argument("--dbscan_eps", type=str, default="0.5,1.0,1.5", help="Comma-separated eps list.")
    ap.add_argument("--dbscan_min_samples", type=int, default=3)

    # Sweep alphas for hybrid weighting
    ap.add_argument("--hybrid_alphas", type=str, default="1,0.3,0.1,0.03,0.01,0.003,0.001",
                    help="Comma-separated audio weights (alpha) AFTER per-modality standardization.")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    results_dir = (project_root / args.results_dir).resolve()

    z_audio_path = results_dir / "conv_vae_audio" / "z_audio_mu.npy"
    z_lyr_path = results_dir / "embeddings" / "Z_lyr.npy"
    meta_path = results_dir / "embeddings" / "lyrics_meta.csv"
    x_mel_path = results_dir / "audio_features" / "X_aud_logmel.npy"

    for p in [z_audio_path, z_lyr_path, meta_path, x_mel_path]:
        if not p.exists():
            raise RuntimeError(f"Missing: {p}")

    Z_audio = np.load(z_audio_path).astype(np.float32)   # (N,16)
    Z_lyr = np.load(z_lyr_path).astype(np.float32)       # (N,29)
    meta = pd.read_csv(meta_path)

    assert Z_audio.shape[0] == Z_lyr.shape[0] == len(meta), (
        f"Row mismatch: Z_audio={Z_audio.shape[0]} Z_lyr={Z_lyr.shape[0]} meta={len(meta)}"
    )

    if "language" not in meta.columns:
        raise RuntimeError("lyrics_meta.csv must contain a 'language' column for ARI.")
    y_true = _encode_labels(meta["language"])

    n_lang = len(meta["language"].astype(str).unique())
    k = args.k if args.k and args.k > 0 else max(2, n_lang)

    dbscan_eps_list = [float(x.strip()) for x in args.dbscan_eps.split(",") if x.strip()]
    hybrid_alphas = [float(x.strip()) for x in args.hybrid_alphas.split(",") if x.strip()]

    # A) Audio baseline: time-mean log-mel -> (N,80)
    X_mel = np.load(x_mel_path).astype(np.float32)   # (N,1,80,3000)
    X_aud_mean = X_mel.mean(axis=-1).squeeze(1)      # (N,80)

    # B) Audio VAE latent
    X_aud_vae = Z_audio                               # (N,16)

    # C) Lyrics embedding
    X_lyr = Z_lyr                                     # (N,29)

    # ===== IMPORTANT: Proper hybrid weighting =====
    # 1) Standardize each modality separately
    Z_audio_s = StandardScaler().fit_transform(Z_audio)
    Z_lyr_s = StandardScaler().fit_transform(Z_lyr)

    # Representations dictionary:
    # value is (X, already_scaled_bool)
    reps: Dict[str, Tuple[np.ndarray, bool]] = {
        # We'll scale these normally in the loop
        "audio_baseline_meanmel": (X_aud_mean, False),
        "audio_vae_latent": (X_aud_vae, False),
        "lyrics_svd": (X_lyr, False),
    }

    # 2) Apply alpha AFTER per-modality scaling and DO NOT re-scale the concat
    for a in hybrid_alphas:
        X_h = np.concatenate([a * Z_audio_s, Z_lyr_s], axis=1)
        reps[f"hybrid_alpha{a:g}_audioVAE_plus_lyrics"] = (X_h, True)  # already scaled

    out_dir = results_dir / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for name, (X, already_scaled) in reps.items():
        if already_scaled:
            Xs = X
        else:
            Xs = StandardScaler().fit_transform(X)

        res = run_all_clusterers(
            X=Xs,
            y_true=y_true,
            k=k,
            seed=args.seed,
            dbscan_eps_list=dbscan_eps_list,
            dbscan_min_samples=args.dbscan_min_samples,
        )
        all_results[name] = res

    metrics_path = out_dir / "metrics_medium.json"
    metrics_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print("Saved:", metrics_path)

    rows = []
    for rep, methods in all_results.items():
        for method, m in methods.items():
            rows.append({"rep": rep, "method": method, **m})
    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "metrics_medium.csv"
    df_out.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    print("\n=== Quick best per representation (prefers silhouette, then ARI) ===")
    for rep in reps.keys():
        sub = df_out[df_out["rep"] == rep].copy()
        sub["sil_ok"] = sub["silhouette"].notna().astype(int)
        sub = sub.sort_values(["sil_ok", "silhouette", "ARI"], ascending=False)
        top = sub.iloc[0].to_dict()
        print(
            f"- {rep}: {top['method']} | silhouette={top.get('silhouette')} "
            f"db={top.get('davies_bouldin')} ARI={top.get('ARI')}"
        )


if __name__ == "__main__":
    main()
