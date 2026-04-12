"""Auxiliary heads for self-supervised representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear projection used for CURL / contrastive InfoNCE loss."""

    type_name = "projection"

    def __init__(self, input_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ConvDecoderHead(nn.Module):
    """Convolutional decoder for pixel reconstruction (AE auxiliary loss).

    Decodes a flat latent vector back to an image using transposed convolutions.
    """

    type_name = "decoder"

    def __init__(
        self,
        latent_dim: int,
        output_shape: tuple[int, int, int] = (3, 84, 84),
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        C, H, W = output_shape
        # Compute stem size: assume 4× upsampling → stem_h = H // 4
        sh, sw = H // 8, W // 8
        self.stem_h, self.stem_w = sh, sw
        self.stem_channels = base_channels * 4

        self.fc = nn.Linear(latent_dim, self.stem_channels * sh * sw)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.stem_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, C, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        x = self.fc(z).view(B, self.stem_channels, self.stem_h, self.stem_w)
        return self.deconv(x)


class VAEHead(nn.Module):
    """Variational reparameterization head for VAE auxiliary loss.

    Takes a CNN encoder output and maps it to a (mu, log_var) pair.
    The caller is responsible for:
    1. Sampling ``z = mu + eps * std``  (use :func:`srl.algorithms.sac._reparameterize`)
    2. Passing ``z`` through a :class:`ConvDecoderHead` for reconstruction.
    3. Computing :func:`srl.losses.aux_losses.vae_loss`.

    Shape::

        z_det  (B, latent_dim)  →  mu     (B, vae_dim)
                                    log_var (B, vae_dim)
    """

    type_name = "vae"

    def __init__(self, input_dim: int, vae_dim: int | None = None) -> None:
        super().__init__()
        out_dim = vae_dim or input_dim
        self.fc_mu = nn.Linear(input_dim, out_dim)
        self.fc_log_var = nn.Linear(input_dim, out_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mu, log_var)`` from deterministic encoder output ``z``."""
        return self.fc_mu(z), self.fc_log_var(z)


class LatentTransitionModel(nn.Module):
    """Lightweight MLP forward model for SPR (Self-Predictive Representations).

    Predicts ``z_{t+1}`` from the current latent ``z_t`` and the action taken.
    Used by :func:`srl.losses.aux_losses.spr_loss` to provide a self-supervised
    latent prediction signal without requiring next-observation rendering.

    Shape::

        (z_t: (B, latent_dim), action: (B, action_dim))  →  z_{t+1}: (B, latent_dim)
    """

    type_name = "latent_transition"

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        h = hidden_dim or latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, h),
            nn.LayerNorm(h),
            nn.ReLU(),
            nn.Linear(h, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(0).expand(z.size(0), -1)
        x = torch.cat([z, action], dim=-1)
        return self.net(x)
