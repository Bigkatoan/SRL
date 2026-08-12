"""Augmentation pipeline for visual observations (GPU-accelerated)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_crop(obs: torch.Tensor, crop_size: int | None = None) -> torch.Tensor:
    """Random crop with zero-padding (DrQ / CURL default augmentation).

    Parameters
    ----------
    obs:
        ``[B, C, H, W]`` float tensor, values in [0, 1].
    crop_size:
        Output spatial size.  Defaults to 84% of original H.
    """
    B, C, H, W = obs.shape
    if crop_size is None:
        crop_size = int(H * 0.9)
    pad = (H - crop_size) // 2
    obs_pad = F.pad(obs, [pad] * 4, mode="replicate")
    h0 = torch.randint(0, 2 * pad + 1, (B,))
    w0 = torch.randint(0, 2 * pad + 1, (B,))
    out = torch.stack(
        [obs_pad[b, :, h0[b] : h0[b] + crop_size, w0[b] : w0[b] + crop_size] for b in range(B)]
    )
    return out


def random_translate(obs: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
    """Random integer translation with border replication."""
    B, C, H, W = obs.shape
    shifts_h = torch.randint(-max_shift, max_shift + 1, (B,))
    shifts_w = torch.randint(-max_shift, max_shift + 1, (B,))
    out = torch.stack(
        [torch.roll(obs[b], (int(shifts_h[b]), int(shifts_w[b])), dims=(-2, -1)) for b in range(B)]
    )
    return out


def color_jitter(
    obs: torch.Tensor,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> torch.Tensor:
    """Per-sample brightness and contrast jitter."""
    B = obs.shape[0]
    b_jitter = 1.0 + torch.empty(B, 1, 1, 1, device=obs.device).uniform_(-brightness, brightness)
    c_jitter = 1.0 + torch.empty(B, 1, 1, 1, device=obs.device).uniform_(-contrast, contrast)
    mean = obs.mean(dim=(-3, -2, -1), keepdim=True)
    out = (obs * b_jitter - mean) * c_jitter + mean
    return out.clamp(0.0, 1.0)


def cutout(obs: torch.Tensor, prob: float = 0.5, size_pct: float = 0.1) -> torch.Tensor:
    """Random rectangular cutout."""
    B, C, H, W = obs.shape
    out = obs.clone()
    cut_h, cut_w = int(H * size_pct), int(W * size_pct)
    for b in range(B):
        if torch.rand(1).item() < prob:
            i = torch.randint(0, H - cut_h + 1, (1,)).item()
            j = torch.randint(0, W - cut_w + 1, (1,)).item()
            out[b, :, i : i + cut_h, j : j + cut_w] = 0.0
    return out


def augment(obs: torch.Tensor, mode: str = "curl") -> torch.Tensor:
    """Apply augmentation pipeline.

    Parameters
    ----------
    obs:
        ``[B, C, H, W]`` tensor.
    mode:
        ``'drq'``       — random crop only
        ``'curl'``      — random crop + color jitter
        ``'aggressive'``— random crop + translate + color jitter + cutout
    """
    if mode in ("drq", "curl", "aggressive"):
        obs = random_crop(obs)
    if mode in ("curl", "aggressive"):
        obs = color_jitter(obs)
    if mode == "aggressive":
        obs = random_translate(obs)
        obs = cutout(obs)
    return obs
