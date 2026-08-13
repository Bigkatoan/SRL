"""CNN builder — parses layer config dicts/lists into nn.Sequential blocks."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from srl.networks.layers.activations import get_activation
from srl.networks.layers.dropout import get_dropout
from srl.networks.layers.init import apply_weight_init
from srl.networks.layers.norms import get_norm
from srl.networks.layers.pooling import get_pooling

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_SHORTHAND_KEYS = ("out_channels", "kernel", "stride", "activation", "pooling")


def _normalise_layer(layer: list | dict[str, Any]) -> dict[str, Any]:
    """Convert shorthand list → full dict."""
    if isinstance(layer, (list, tuple)):
        # strict=False: shorthand lists may intentionally omit trailing keys.
        cfg = dict(zip(_SHORTHAND_KEYS, layer, strict=False))
        return cfg
    return dict(layer)


def _resolve(cfg: dict, key: str, default: Any) -> Any:
    return cfg.get(key, default)


def _conv_output_size(size: int, kernel: int, stride: int, padding: int | str) -> int:
    if padding == "same":
        return size
    return math.floor((size - kernel + 2 * int(padding)) / stride + 1)


# ──────────────────────────────────────────────────────────────────────────────
# _ResidualCNNBlock
# ──────────────────────────────────────────────────────────────────────────────


class _ResidualCNNBlock(nn.Module):
    def __init__(self, transform: nn.Module, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.transform = transform
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x) + self.skip(x)


# ──────────────────────────────────────────────────────────────────────────────
# Depthwise Separable Conv block
# ──────────────────────────────────────────────────────────────────────────────


def _depthwise_sep_conv(
    in_ch: int, out_ch: int, kernel: int, stride: int, padding
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_ch,
            in_ch,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=False,
        ),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
    )


# ──────────────────────────────────────────────────────────────────────────────
# build_cnn
# ──────────────────────────────────────────────────────────────────────────────


def build_cnn(
    layer_configs: list[list | dict],
    input_shape: tuple[int, int, int],  # (C, H, W)
    *,
    conv_type: str = "cnn",  # "cnn" | "depthwise_cnn"
    default_activation: str = "relu",
    default_norm: str = "none",
    default_dropout: float = 0.0,
    norm_order: str = "post",  # "pre" | "post"
    weight_init: str = "kaiming_normal",
) -> tuple[nn.Module, int]:
    """Build a CNN encoder from layer configs.

    Returns
    -------
    module : nn.Module
    flat_output_dim : int   (C_out × H_out × W_out)
    """
    modules: list[nn.Module] = []
    in_ch, h, w = input_shape

    for raw in layer_configs:
        cfg = _normalise_layer(raw)
        out_ch: int = cfg["out_channels"]
        kernel: int = cfg.get("kernel", 3)
        stride: int = cfg.get("stride", 1)
        # "same" padding is only well-defined (and only accepted by
        # nn.Conv2d) for stride=1; strided layers -- the common case for
        # shorthand-list downsampling convs like [32, 8, 4, relu] -- default
        # to "valid" (no padding) instead.
        padding = cfg.get("padding", "same" if stride == 1 else "valid")
        activation = _resolve(cfg, "activation", default_activation)
        norm_name = _resolve(cfg, "norm", default_norm)
        dropout_rate = float(_resolve(cfg, "dropout", default_dropout))
        dropout_type = _resolve(cfg, "dropout_type", "auto")
        pooling_name = str(_resolve(cfg, "pooling", "none"))
        residual = bool(_resolve(cfg, "residual", False))
        num_groups = cfg.get("norm_groups", 8)

        # Padding for Conv2d
        pad_arg = 0 if padding == "valid" else (padding if isinstance(padding, int) else padding)

        # Conv block
        if conv_type == "depthwise_cnn":
            conv = _depthwise_sep_conv(in_ch, out_ch, kernel, stride, pad_arg)
        else:
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad_arg)

        act = get_activation(activation)
        norm = get_norm(norm_name, out_ch, dim=2, num_groups=num_groups)
        drop = get_dropout(dropout_rate, dropout_type, dim=2)
        pool = get_pooling(pooling_name)

        if norm_order == "pre":
            block = nn.Sequential(
                get_norm(norm_name, in_ch, dim=2, num_groups=num_groups),
                conv,
                act,
                drop,
                pool,
            )
        else:  # post
            block = nn.Sequential(conv, norm, act, drop, pool)

        if residual:
            modules.append(_ResidualCNNBlock(block, in_ch, out_ch))
        else:
            modules.append(block)

        # Track spatial dimensions
        conv_pad = 0 if padding in ("same",) else (padding if isinstance(padding, int) else 0)
        if padding == "same":
            h_out, w_out = h, w
        else:
            h_out = _conv_output_size(h, kernel, stride, conv_pad)
            w_out = _conv_output_size(w, kernel, stride, conv_pad)

        if pooling_name not in ("none", "identity", ""):
            h_out = h_out // 2
            w_out = w_out // 2

        in_ch, h, w = out_ch, max(1, h_out), max(1, w_out)

    net = nn.Sequential(*modules)
    apply_weight_init(net, weight_init)
    flat_dim = in_ch * h * w
    return net, flat_dim
