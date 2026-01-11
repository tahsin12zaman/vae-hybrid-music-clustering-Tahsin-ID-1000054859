from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

SEED = 42

def scatter(points, labels, title, out_path):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    project_root = Path(__file__).resolve().parents[2]
    emb_dir = project_root / "results_easy" / "embeddings"
    met_dir = project_root / "results_easy" / "metrics"
    out_dir = project_root / "results_easy" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    Z = np.load(emb_dir / "Z.npy").astype(np.float32)
    clusters = pd.read_csv(met_dir / "clusters.csv")

    reducer = umap.UMAP(n_components=2, random_state=SEED)
    Z2 = reducer.fit_transform(Z)

    scatter(Z2, clusters["cluster_vae"].to_numpy(), "UMAP of VAE latent (colored by KMeans cluster)", out_dir / "umap_vae_clusters.png")

    # Optional: color by language to see if clusters align with English/Bangla
    # map english/bangla -> 0/1
    lang_map = {"english": 0, "bangla": 1}
    lang = clusters["language"].map(lang_map).fillna(-1).to_numpy()
    scatter(Z2, lang, "UMAP of VAE latent (colored by language: english=0, bangla=1)", out_dir / "umap_language.png")

    print("Saved plots in:", out_dir)

if __name__ == "__main__":
    main()
