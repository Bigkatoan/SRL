"""Activation function registry."""

from __future__ import annotations

import torch.nn as nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "elu": nn.ELU,
    "mish": nn.Mish,
    "hardswish": nn.Hardswish,
    "none": nn.Identity,
    "identity": nn.Identity,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Available: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name](**kwargs)
