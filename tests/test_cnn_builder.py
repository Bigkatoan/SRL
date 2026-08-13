"""Regression tests for build_cnn's shorthand-list layer parsing.

Every CNN example across the docs (yaml_core.md, encoders.md,
yaml/heads_flows.md, yaml/auxiliary.md, examples/encoder_examples.md) uses
the shorthand list format `[out_channels, kernel_size, stride, activation]`
-- e.g. `[32, 8, 4, relu]`. `_SHORTHAND_KEYS` previously mapped the 3rd
position to `padding` instead of `stride`, so every one of those documented
examples silently built a stride=1 (never-downsampling) network with a huge,
wrong-shaped flatten dim instead of the intended architecture. This was
never caught because no test exercised a real CNN build until now --
configs/envs/car_racing_ppo_visual.yaml (the one shipped config that uses
this format against a real env) crashed on the very first eval step as a
consequence.
"""

from __future__ import annotations

import torch

from srl.networks.layers.cnn_builder import build_cnn

_ATARI_STYLE_LAYERS = [
    [32, 8, 4, "relu"],
    [64, 4, 2, "relu"],
    [64, 3, 1, "relu"],
]


def test_shorthand_list_third_position_is_stride_not_padding() -> None:
    net, flat_dim = build_cnn(_ATARI_STYLE_LAYERS, input_shape=(3, 96, 96))

    # A real strided Atari-style stack on a 96x96 input downsamples sharply
    # (stride=4 then stride=2, "valid"-padded since "same" padding is
    # rejected by PyTorch for stride>1; the final stride=1 layer keeps its
    # historical "same" default) -- if `stride` were silently dropped
    # (defaulting to 1 for every layer, as it did with the padding/stride
    # swap), padding="same" would keep spatial dims near 96x96 throughout
    # and flat_dim would come out enormous (614656, not 6400).
    assert flat_dim == 6400

    x = torch.zeros(2, 3, 96, 96)
    out = net(x)
    assert tuple(out.shape) == (2, 64, 10, 10)


def test_shorthand_list_conv_layers_actually_have_the_declared_stride() -> None:
    net, _ = build_cnn(_ATARI_STYLE_LAYERS, input_shape=(3, 96, 96))
    strides = [block[0].stride[0] for block in net]
    assert strides == [4, 2, 1]


def test_explicit_dict_layer_config_still_supports_padding() -> None:
    # Dict-form configs (not the positional shorthand) can still set padding
    # explicitly -- this must keep working unchanged.
    layers = [{"out_channels": 16, "kernel": 3, "stride": 1, "padding": 1, "activation": "relu"}]
    net, flat_dim = build_cnn(layers, input_shape=(3, 32, 32))
    assert flat_dim == 16 * 32 * 32  # stride=1, padding=1, kernel=3 preserves spatial size

    x = torch.zeros(1, 3, 32, 32)
    out = net(x)
    assert tuple(out.shape) == (1, 16, 32, 32)
