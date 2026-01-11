#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Conv-VAE for inputs shaped (B,1,80,T).

    IMPORTANT: This model builds some layers lazily after seeing the first input
    so it can adapt to different T. That means you MUST run one forward pass
    before creating the optimizer (done in main()).
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> (16, 40,  T/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> (32, 20,  T/4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 10,  T/8)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 5, T/16)
            nn.ReLU(),
        )

        # Lazy-built after first encode()
        self._enc_shape = None  # (C,H,W)
        self._flat_dim = None

        self.mu = None
        self.logvar = None
        self.dec_fc = None
        self.dec = None

    def _build_heads(self, enc_out: torch.Tensor):
        # enc_out: (B,C,H,W)
        _, C, H, W = enc_out.shape
        self._enc_shape = (C, H, W)
        self._flat_dim = C * H * W

        self.mu = nn.Linear(self._flat_dim, self.latent_dim)
        self.logvar = nn.Linear(self._flat_dim, self.latent_dim)
        self.dec_fc = nn.Linear(self.latent_dim, self._flat_dim)

        # Decoder mirrors encoder strides; can overshoot in time dimension -> we crop
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        if self._flat_dim is None:
            self._build_heads(h)
            # move newly created layers to correct device
            self.mu.to(x.device)
            self.logvar.to(x.device)
            self.dec_fc.to(x.device)
            self.dec.to(x.device)

        h = h.flatten(1)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        assert self._enc_shape is not None
        C, H, W = self._enc_shape
        # Decoder expects C==128 because we hard-coded ConvTranspose2d(128,...)
        if C != 128:
            raise RuntimeError(f"Unexpected encoder channels: {C} (expected 128)")
        h = self.dec_fc(z).view(-1, C, H, W)
        xhat = self.dec(h)
        return xhat

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


def vae_loss(x: torch.Tensor, xhat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
    # crop recon to match original (decoder may overshoot)
    _, _, H, W = x.shape
    xhat = xhat[:, :, :H, :W]

    recon = F.mse_loss(xhat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, recon.detach().item(), kl.detach().item(), xhat


def main():
    ap = argparse.ArgumentParser("Train Conv-VAE on log-mel spectrogram tensors (medium task)")
    ap.add_argument("--x_path", type=str, default="results_medium/audio_features/X_aud_logmel.npy")
    ap.add_argument("--out_dir", type=str, default="results_medium/conv_vae_audio")
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=1.0, help="Final KL weight (beta).")
    ap.add_argument("--kl_warmup", type=int, default=100, help="Linearly ramp beta over this many epochs (0 disables).")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=5.0, help="0 disables.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    project_root = Path(__file__).resolve().parents[2]
    x_path = (project_root / args.x_path).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(x_path).astype(np.float32)  # (N,1,80,T)

    # Dataset-level normalization (simple + stable)
    X_mean = float(X.mean())
    X_std = float(X.std() + 1e-8)
    Xn = (X - X_mean) / X_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(Xn).to(device)

    model = ConvVAE(latent_dim=args.latent_dim).to(device)

    # ✅ CRITICAL FIX: build lazy layers BEFORE creating optimizer
    model.eval()
    with torch.no_grad():
        _ = model(xt[:1])
    model.train()

    # Now optimizer sees ALL parameters (including mu/logvar/dec_fc/dec)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    N = xt.shape[0]
    idx = np.arange(N)

    for ep in range(1, args.epochs + 1):
        np.random.shuffle(idx)
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        seen = 0

        # KL warmup schedule
        if args.kl_warmup and args.kl_warmup > 0:
            beta_eff = args.beta * min(1.0, ep / float(args.kl_warmup))
        else:
            beta_eff = args.beta

        for i in range(0, N, args.batch):
            b = idx[i:i + args.batch]
            xb = xt[b]

            opt.zero_grad()
            xhat, mu, logvar = model(xb)
            loss, recon, kl, _ = vae_loss(xb, xhat, mu, logvar, beta=beta_eff)
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            bs = len(b)
            total_loss += loss.item() * bs
            total_recon += recon * bs
            total_kl += kl * bs
            seen += bs

        if ep == 1 or ep % 25 == 0 or ep == args.epochs:
            print(
                f"[ep {ep:4d}] beta={beta_eff:.3f} "
                f"loss={total_loss/seen:.4f} recon={total_recon/seen:.4f} kl={total_kl/seen:.4f}"
            )

    # Save latent means for all samples + one recon sample
    model.eval()
    with torch.no_grad():
        mu_all, _ = model.encode(xt)
        Z_mu = mu_all.cpu().numpy().astype(np.float32)

        xhat, _, _ = model(xt[:1])
        xhat = xhat[:, :, :xt.shape[2], :xt.shape[3]].cpu().numpy().astype(np.float32)

    np.save(out_dir / "z_audio_mu.npy", Z_mu)
    np.save(out_dir / "x_mean.npy", np.array([X_mean], dtype=np.float32))
    np.save(out_dir / "x_std.npy", np.array([X_std], dtype=np.float32))
    torch.save(model.state_dict(), out_dir / "conv_vae_state.pt")
    np.save(out_dir / "recon_sample.npy", xhat)

    print("Saved:", out_dir / "z_audio_mu.npy", "shape:", Z_mu.shape)
    print("Saved:", out_dir / "recon_sample.npy", "shape:", xhat.shape)


if __name__ == "__main__":
    main()
