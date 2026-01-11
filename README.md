# VAE Hybrid Music Clustering (Easy + Medium + Hard)

**Student:** Tahsin (ID: 1000054859)  
**Repo:** vae-hybrid-music-clustering-Tahsin-ID-1000054859  
**GitHub:** https://github.com/tahsin12zaman/vae-hybrid-music-clustering-Tahsin-ID-1000054859

This project performs unsupervised clustering on a small multilingual music dataset using VAE-based representations and hybrid multi-modal features (audio + lyrics + genre).

---

## What’s Implemented

### Easy Task
- Basic **VAE** embeddings → **K-Means**
- Baseline: **PCA + K-Means**
- Visualization: **UMAP**

### Medium Task
- **Conv-VAE** on **log-mel spectrograms**
- Hybrid representation: **audio latent + lyrics embeddings**
- Clustering: **K-Means, Agglomerative (Ward), DBSCAN**
- Metrics: **Silhouette, Davies–Bouldin, ARI** (language used as partial label)

### Hard Task
- **Beta-VAE** (and AE baseline via `beta=0`)
- Multi-modal clustering: **audio + lyrics** (and optional **genre one-hot** feature)
- Metrics: **Silhouette, ARI, NMI, Cluster Purity**
- Visualizations: **t-SNE latent plots**, **cluster distributions**, **reconstructions**

---

## Repository Structure (high level)

- `data/`
  - `lyrics/lyrics.csv` (filename, language, genre, lyrics)
  - `genre_map.csv`, `meta_with_genre.csv`
  - *(raw audio is intentionally not tracked in git)*
- `src/`
  - `easy_task_scripts/`
  - `medium_task_scripts/`
  - `hard_task_scripts/`
- `results_easy/` : Easy task outputs (metrics + plots)
- `results_medium/` : Medium task outputs (embeddings/features + clustering metrics)
- `results_hard/` : Hard task outputs (clustering metrics + plots)
- `REPORT.pdf` : NeurIPS-style report (in repo root)

---

## Setup

Recommended (Ubuntu / Python 3.8+):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
