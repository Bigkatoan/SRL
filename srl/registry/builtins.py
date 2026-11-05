"""SRL's built-in encoders, heads, algorithms and env types.

Registered through the very same :mod:`srl.registry.registry` API that user
code and entry-point plugins use. Imported lazily by the registries on first
use; every factory imports its network module inside the function, so loading
this module does not build or import any ``nn.Module`` code.
"""

from __future__ import annotations

from typing import Any

from srl.registry.registry import (
    AlgorithmSpec,
    register_algorithm,
    register_encoder,
    register_env,
    register_head,
)

_BUILTIN = "builtin"

# ──────────────────────────────────────────────────────────────────────────────
# Encoders: factory(cfg: EncoderConfig) -> nn.Module
# ──────────────────────────────────────────────────────────────────────────────


@register_encoder("mlp", requires=("input_dim",), source=_BUILTIN)
def build_mlp_encoder(cfg: Any):
    from srl.networks.encoders.mlp_encoder import MLPEncoder

    return MLPEncoder(input_dim=cfg.input_dim, layer_configs=cfg.layers, latent_dim=cfg.latent_dim)


@register_encoder("cnn", requires=("input_shape",), source=_BUILTIN)
def build_cnn_encoder(cfg: Any):
    from srl.networks.encoders.cnn_encoder import CNNEncoder

    return CNNEncoder(
        input_shape=tuple(cfg.input_shape), layer_configs=cfg.layers, latent_dim=cfg.latent_dim
    )


@register_encoder("lstm", requires=("input_dim",), source=_BUILTIN)
def build_lstm_encoder(cfg: Any):
    from srl.networks.encoders.mlp_encoder import MLPEncoder
    from srl.networks.encoders.recurrent import LSTMEncoder

    base = MLPEncoder(input_dim=cfg.input_dim, layer_configs=cfg.layers, latent_dim=cfg.latent_dim)
    return LSTMEncoder(base_encoder=base, hidden_size=cfg.lstm_hidden)


# Recurrent output size is lstm_hidden, not latent_dim (see builder).
build_lstm_encoder.srl_is_recurrent = True  # type: ignore[attr-defined]

_TEXT_KEYS = ("vocab_size", "embed_dim", "channels", "max_len")


@register_encoder("text", extra_keys=_TEXT_KEYS, source=_BUILTIN)
def build_text_encoder(cfg: Any):
    from srl.networks.encoders.text_encoder import CharCNNTextEncoder

    # These used to be silently dropped into an unread `extra` dict.
    kwargs = {k: int(cfg.extra[k]) for k in _TEXT_KEYS if k in cfg.extra}
    return CharCNNTextEncoder(latent_dim=cfg.latent_dim, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Heads: factory(cfg: HeadConfig, input_dim: int) -> nn.Module
# ──────────────────────────────────────────────────────────────────────────────


def _actor_factory(head_type: str):
    def build(cfg: Any, input_dim: int):
        from srl.networks.heads.actor_head import build_actor_head

        extra_kw = {}
        if head_type == "gaussian":
            extra_kw["state_dependent_std"] = bool(cfg.state_dependent_std)
        return build_actor_head(
            head_type=head_type,
            input_dim=input_dim,
            action_dim=cfg.action_dim,
            layer_configs=cfg.layers,
            log_std_init=cfg.log_std_init,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            **extra_kw,
        )

    build.__name__ = f"build_{head_type}_head"
    return build


def _critic_factory(head_type: str):
    def build(cfg: Any, input_dim: int):
        from srl.networks.heads.critic_head import build_critic_head

        return build_critic_head(
            head_type=head_type,
            input_dim=input_dim,
            layer_configs=cfg.layers,
            action_dim=cfg.action_dim,
        )

    build.__name__ = f"build_{head_type}_head"
    return build


for _name in ("gaussian", "squashed_gaussian", "deterministic"):
    register_head(_name, _actor_factory(_name), requires_action_dim=True, source=_BUILTIN)
register_head("value", _critic_factory("value"), source=_BUILTIN)
for _name in ("twin_q", "q", "q_function"):
    register_head(_name, _critic_factory(_name), requires_action_dim=True, source=_BUILTIN)


# ──────────────────────────────────────────────────────────────────────────────
# Algorithms
# ──────────────────────────────────────────────────────────────────────────────


def _register_algorithms() -> None:
    from srl.core import config as c

    specs = [
        AlgorithmSpec(
            "ppo",
            c.PPOConfig,
            "srl.algorithms.ppo:PPO",
            False,
            c.VisualPPOConfig,
            ("gaussian",),
            ("value",),
        ),
        AlgorithmSpec(
            "a2c", c.A2CConfig, "srl.algorithms.a2c:A2C", False, None, ("gaussian",), ("value",)
        ),
        AlgorithmSpec(
            "a3c", c.A3CConfig, "srl.algorithms.a3c:A3C", False, None, ("gaussian",), ("value",)
        ),
        AlgorithmSpec(
            "sac",
            c.SACConfig,
            "srl.algorithms.sac:SAC",
            True,
            c.VisualSACConfig,
            ("squashed_gaussian",),
            ("twin_q",),
        ),
        AlgorithmSpec(
            "ddpg",
            c.DDPGConfig,
            "srl.algorithms.ddpg:DDPG",
            True,
            None,
            ("deterministic",),
            ("q", "q_function", "twin_q"),
        ),
        AlgorithmSpec(
            "td3",
            c.TD3Config,
            "srl.algorithms.td3:TD3",
            True,
            None,
            ("deterministic",),
            ("twin_q",),
        ),
    ]
    for spec in specs:
        register_algorithm(spec.name, spec, source=_BUILTIN)


_register_algorithms()


# ──────────────────────────────────────────────────────────────────────────────
# Env types: factory(env_id, *, device="cpu", n_envs=1, render=False) -> env
# ──────────────────────────────────────────────────────────────────────────────


def _cli_env_factory(env_type: str):
    def make(env_id: str, *, device: str = "cpu", n_envs: int = 1, render: bool = False):
        from srl.cli.train import _make_cli_env

        return _make_cli_env(env_id, device, n_envs, env_type, render=render)

    make.__name__ = f"make_{env_type}_env"
    return make


for _env_type in ("flat", "goal", "racecar", "isaaclab", "mjlab"):
    register_env(_env_type, _cli_env_factory(_env_type), source=_BUILTIN)
