"""MLP builder — parses layer config dicts/ints into nn.Sequential blocks."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from srl.networks.layers.activations import get_activation
from srl.networks.layers.dropout import get_dropout
from srl.networks.layers.init import apply_weight_init
from srl.networks.layers.norms import get_norm

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _normalise_layer(layer: int | dict[str, Any]) -> dict[str, Any]:
    """Convert shorthand int → full dict using defaults."""
    if isinstance(layer, int):
        return {"out": layer}
    d = dict(layer)
    # Support both key names
    if "out" not in d and "out_features" in d:
        d["out"] = d["out_features"]
    return d


def _resolve(layer_cfg: dict, key: str, default: Any) -> Any:
    return layer_cfg.get(key, default)


# ──────────────────────────────────────────────────────────────────────────────
# _ResidualBlock
# ──────────────────────────────────────────────────────────────────────────────


class _ResidualBlock(nn.Module):
    def __init__(self, transform: nn.Module, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.transform = transform
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.transform(x) + self.skip(x)


# ──────────────────────────────────────────────────────────────────────────────
# build_mlp
# ──────────────────────────────────────────────────────────────────────────────


def build_mlp(
    layer_configs: list[int | dict],
    input_dim: int,
    *,
    default_activation: str = "relu",
    default_norm: str = "none",
    default_dropout: float = 0.0,
    norm_order: str = "post",  # "pre" | "post"
    weight_init: str = "none",
) -> tuple[nn.Module, int]:
    """Build an MLP from a list of layer configs.

    Parameters
    ----------
    layer_configs:
        List of ints (shorthand) or dicts (full spec).
    input_dim:
        Input feature size.
    default_*:
        Encoder-level defaults applied to every layer.
    norm_order:
        ``'post'`` — Linear → Norm → Act → Dropout
        ``'pre'``  — Norm → Linear → Act → Dropout

    Returns
    -------
    module : nn.Module
    output_dim : int
    """
    modules: list[nn.Module] = []
    current_dim = input_dim

    for raw in layer_configs:
        cfg = _normalise_layer(raw)
        out_dim: int = cfg["out"]
        activation = _resolve(cfg, "activation", default_activation)
        norm_name = _resolve(cfg, "norm", default_norm)
        dropout_rate = float(_resolve(cfg, "dropout", default_dropout))
        dropout_type = _resolve(cfg, "dropout_type", "auto")
        residual = bool(_resolve(cfg, "residual", False))

        linear = nn.Linear(current_dim, out_dim)
        act = get_activation(activation)
        norm = get_norm(norm_name, out_dim, dim=1)
        drop = get_dropout(dropout_rate, dropout_type, dim=1)

        if norm_order == "pre":
            block = nn.Sequential(
                get_norm(norm_name, current_dim, dim=1),
                linear,
                act,
                drop,
            )
        else:  # post
            block = nn.Sequential(linear, norm, act, drop)

        if residual:
            modules.append(_ResidualBlock(block, current_dim, out_dim))
        else:
            modules.append(block)

        current_dim = out_dim

    net = nn.Sequential(*modules)
    apply_weight_init(net, weight_init)
    return net, current_dim
