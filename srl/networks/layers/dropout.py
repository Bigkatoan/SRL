"""Dropout layer factory — standard, spatial (2D), and DropPath."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_dropout(rate: float, dropout_type: str = "auto", dim: int = 1) -> nn.Module:
    """Return a dropout module.

    Parameters
    ----------
    rate:
        Dropout probability.  0.0 returns ``nn.Identity()``.
    dropout_type:
        ``'dropout'``, ``'dropout2d'``, ``'droppath'``, or ``'auto'``.
        Auto selects ``dropout2d`` when *dim* == 2, else ``dropout``.
    dim:
        Context dimension (1 = MLP, 2 = CNN).  Used only when *dropout_type* == ``'auto'``.
    """
    if rate <= 0.0:
        return nn.Identity()

    t = dropout_type.lower()
    if t == "auto":
        t = "dropout2d" if dim == 2 else "dropout"

    if t == "dropout":
        return nn.Dropout(p=rate)
    if t == "dropout2d":
        return nn.Dropout2d(p=rate)
    if t == "droppath":
        return _DropPath(p=rate)

    raise ValueError(
        f"Unknown dropout_type '{dropout_type}'. Options: dropout, dropout2d, droppath, auto"
    )


class _DropPath(nn.Module):
    """Stochastic Depth (drop entire residual branch per sample)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep
