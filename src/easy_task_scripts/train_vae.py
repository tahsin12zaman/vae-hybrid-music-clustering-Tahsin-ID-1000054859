from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- config ---
Z_DIM = 8
H1, H2 = 128, 64
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-3
SEED = 42


class VAE(nn.Module):
    def __init__(self, x_dim: int, z_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, H1), nn.ReLU(),
            nn.Linear(H1, H2), nn.ReLU(),
        )
        self.mu = nn.Linear(H2, z_dim)
        self.logvar = nn.Linear(H2, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim, H2), nn.ReLU(),
            nn.Linear(H2, H1), nn.ReLU(),
            nn.Linear(H1, x_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar


def kl_div(mu, logvar):
    # KL(N(mu, sigma) || N(0,1))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    project_root = Path(__file__).resolve().parents[2]
    in_path = project_root / "results_easy" / "embeddings" / "X.npy"
    out_dir = project_root / "results_easy" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(in_path).astype(np.float32)  # (N, 40)

    # Standardize (important for stable training)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mean) / std

    np.save(out_dir / "X_mean.npy", mean)
    np.save(out_dir / "X_std.npy", std)

    x_tensor = torch.from_numpy(Xn)
    ds = TensorDataset(x_tensor)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = VAE(x_dim=X.shape[1], z_dim=Z_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for (xb,) in dl:
            xhat, mu, logvar = model(xb)

            recon = nn.functional.mse_loss(xhat, xb, reduction="mean")
            kl = kl_div(mu, logvar)
            loss = recon + 0.1 * kl  # small KL weight helps on tiny datasets

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        if epoch % 50 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | loss {total/len(dl):.6f}")

    # Extract latent means as embeddings
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(x_tensor)
        Z = mu.numpy().astype(np.float32)

    np.save(out_dir / "Z.npy", Z)
    torch.save(model.state_dict(), out_dir / "vae.pt")

    print("Saved Z:", out_dir / "Z.npy", "shape:", Z.shape)
    print("Saved model:", out_dir / "vae.pt")


if __name__ == "__main__":
    main()
