"""Weight initialisation helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable

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


def _orthogonal_(m: nn.Module, gain: float) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _init_head(head: nn.Module, output_gain: float) -> None:
    """Hidden Linear layers gain sqrt(2); the head's output Linear layer(s)
    -- the ones not inside its ``net`` hidden MLP -- *output_gain*."""
    hidden = getattr(head, "net", None)
    hidden_ids = set()
    if isinstance(hidden, nn.Module):
        for m in hidden.modules():
            _orthogonal_(m, math.sqrt(2))
            hidden_ids.add(id(m))
    for m in head.modules():
        if id(m) not in hidden_ids:
            _orthogonal_(m, output_gain)


def apply_sb3_orthogonal_init(
    encoders: Iterable[nn.Module], actor: nn.Module | None, critic: nn.Module | None
) -> None:
    """SB3 ``ActorCriticPolicy(ortho_init=True)`` / CleanRL ``layer_init``.

    Every Linear layer gets an orthogonal weight and a zero bias: gain
    ``sqrt(2)`` for encoder and hidden head layers, ``0.01`` for the actor's
    output layers (mean / log-std / action), ``1.0`` for the critic's
    output layer(s). Multi-head critics (twin Q) are initialised per Q-net.
    """
    for enc in encoders:
        for m in enc.modules():
            _orthogonal_(m, math.sqrt(2))
    if actor is not None:
        _init_head(actor, 0.01)
    if critic is not None:
        subs = [m for m in critic.children() if hasattr(m, "net")]
        for sub in subs or [critic]:
            _init_head(sub, 1.0)
