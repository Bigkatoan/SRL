"""MomentumEncoder — EMA copy of an online encoder (used by CURL)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class MomentumEncoder(nn.Module):
    """Wraps any encoder with a slow-moving (EMA) target copy.

    Parameters
    ----------
    encoder:
        The *online* encoder whose weights are updated by back-propagation.
    tau:
        EMA coefficient for target weights (default 0.99).
    """

    def __init__(self, encoder: nn.Module, tau: float = 0.99) -> None:
        super().__init__()
        self.online = encoder
        self.target = copy.deepcopy(encoder)
        self.tau = tau

        # Target is never trained via gradients
        for p in self.target.parameters():
            p.requires_grad_(False)

    @property
    def latent_dim(self) -> int:
        return self.online.latent_dim

    def forward(self, obs: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        if use_target:
            with torch.no_grad():
                return self.target(obs)
        return self.online(obs)

    @torch.no_grad()
    def update_target(self) -> None:
        """Perform one EMA step: target = τ·target + (1−τ)·online."""
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters(), strict=True):
            p_t.data.mul_(self.tau).add_((1.0 - self.tau) * p_o.data)
