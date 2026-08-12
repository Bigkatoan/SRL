"""Weight initialisation helpers."""

from __future__ import annotations

import torch.nn as nn


def apply_weight_init(module: nn.Module, scheme: str) -> nn.Module:
    """Apply *scheme* to all Linear and Conv layers in *module* recursively."""
    scheme = scheme.lower()
    if scheme in ("none", "default", ""):
        return module

    def _init(m: nn.Module) -> None:
        if not isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
            return
        if scheme == "xavier_uniform":
            nn.init.xavier_uniform_(m.weight)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(m.weight)
        elif scheme == "kaiming_normal":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif scheme == "orthogonal":
            nn.init.orthogonal_(m.weight)
        elif scheme == "zeros":
            nn.init.zeros_(m.weight)
        elif scheme == "ones":
            nn.init.ones_(m.weight)
        else:
            raise ValueError(
                f"Unknown weight init '{scheme}'. Options: xavier_uniform, xavier_normal, "
                "kaiming_normal, kaiming_uniform, orthogonal, zeros, ones, none"
            )
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    module.apply(_init)
    return module
