"""Normalisation layer factory — supports 1D (MLP) and 2D (CNN) contexts."""

from __future__ import annotations

import torch.nn as nn


def get_norm(name: str, num_features: int, dim: int = 1, **kwargs) -> nn.Module:
    """Return a normalisation module.

    Parameters
    ----------
    name:
        One of: ``batch_norm``, ``layer_norm``, ``group_norm``,
        ``instance_norm``, ``rms_norm``, ``none``.
    num_features:
        Number of channels (CNN) or features (MLP).
    dim:
        1 for MLP layers, 2 for CNN feature maps.
    kwargs:
        Extra args, e.g. ``num_groups`` for GroupNorm.
    """
    name = name.lower()
    if name in ("none", "identity", ""):
        return nn.Identity()

    if name == "batch_norm":
        return nn.BatchNorm2d(num_features) if dim == 2 else nn.BatchNorm1d(num_features)

    if name == "layer_norm":
        return nn.LayerNorm(num_features)

    if name == "group_norm":
        num_groups = kwargs.get("num_groups", 8)
        return nn.GroupNorm(num_groups, num_features)

    if name == "instance_norm":
        return nn.InstanceNorm2d(num_features) if dim == 2 else nn.InstanceNorm1d(num_features)

    if name == "rms_norm":
        return _RMSNorm(num_features)

    raise ValueError(
        f"Unknown norm '{name}'. Options: batch_norm, layer_norm, group_norm, "
        "instance_norm, rms_norm, none"
    )


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        import torch

        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):

        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
