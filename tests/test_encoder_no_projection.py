"""``latent_dim: null`` (no encoder projection) and the MuJoCo benchmark
architectures.

The encoder schema defaults to ``latent_dim: 128``, which appends a Linear
projection after the last hidden layer. The Phase-1 benchmark configs are
meant to reproduce SB3's networks exactly, but every one of them silently
got that extra layer (TD3 actor 400-300-*128*-|A|, critic Q(W s, a) with the
state squeezed through an 11->128 linear map, PPO 256-256-*128*-out).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn
import yaml

from srl.config._validate import ConfigError
from srl.registry.builder import ModelBuilder

REPO_ROOT = Path(__file__).resolve().parents[1]
MUJOCO = REPO_ROOT / "configs" / "benchmarks" / "mujoco"


def _cfg(latent_dim, layers, enc_type="mlp"):
    enc = {"name": "enc", "type": enc_type, "latent_dim": latent_dim, "layers": layers}
    if enc_type == "cnn":
        enc["input_shape"] = [3, 16, 16]
    else:
        enc["input_dim"] = 5
    return {
        "encoders": [enc],
        "flows": ["enc -> actor", "enc -> critic"],
        "actor": {"name": "actor", "type": "deterministic", "action_dim": 2},
        "critic": {"name": "critic", "type": "twin_q", "action_dim": 2, "layers": [16]},
    }


def test_latent_dim_null_adds_no_projection() -> None:
    model = ModelBuilder.from_dict(_cfg(None, [32, 24]))
    enc = model.encoders["enc"]
    assert isinstance(enc.projection, nn.Identity)
    assert enc.latent_dim == 24
    assert model.actor.out[0].in_features == 24
    assert model.critic.q1.net[0][0].in_features == 24 + 2


def test_latent_dim_null_with_no_layers_is_identity_encoder() -> None:
    model = ModelBuilder.from_dict(_cfg(None, []))
    assert list(model.encoders["enc"].parameters()) == []
    # Q(s, a) sees the raw observation concatenated with the action.
    assert model.critic.q1.net[0][0].in_features == 5 + 2


def test_latent_dim_default_still_projects() -> None:
    # Backwards compatible: existing configs keep their architecture.
    model = ModelBuilder.from_dict(_cfg(128, [32]))
    assert isinstance(model.encoders["enc"].projection, nn.Linear)
    assert model.encoders["enc"].latent_dim == 128


def test_latent_dim_null_rejected_for_cnn() -> None:
    with pytest.raises(ConfigError, match="latent_dim"):
        ModelBuilder.from_dict(_cfg(None, [], enc_type="cnn"))


def _n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _mlp_params(sizes: list[int]) -> int:
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:], strict=False))


_DIMS = {"hopper": (11, 3), "walker2d": (17, 6), "halfcheetah": (17, 6)}


@pytest.mark.parametrize("env", sorted(_DIMS))
@pytest.mark.parametrize("algo", ["td3", "sac", "ppo"])
def test_benchmark_configs_match_sb3_architecture(env: str, algo: str) -> None:
    """Parameter counts equal SB3's MlpPolicy for the rl-zoo net_arch."""
    obs, act = _DIMS[env]
    model = ModelBuilder.from_yaml(MUJOCO / f"{env}_{algo}.yaml")
    actor_params = _n_params(model.actor) + _n_params(model.encoders["actor_state_enc"])
    critic_params = _n_params(model.critic) + _n_params(model.encoders["critic_state_enc"])
    if algo == "td3":
        assert actor_params == _mlp_params([obs, 400, 300, act])
        assert critic_params == 2 * _mlp_params([obs + act, 400, 300, 1])
    elif algo == "sac":
        # latent_pi 256-256, then mu and log_std linear layers.
        assert actor_params == _mlp_params([obs, 256, 256]) + 2 * (256 * act + act)
        assert critic_params == 2 * _mlp_params([obs + act, 256, 256, 1])
    else:
        # pi -> action_net, plus the state-independent log_std vector. The
        # rl-zoo Walker2d-v4 PPO entry has no policy_kwargs: SB3's default
        # 64-64 Tanh, orthogonal init; Hopper/HalfCheetah use 256-256 ReLU.
        hidden = [64, 64] if env == "walker2d" else [256, 256]
        assert actor_params == _mlp_params([obs, *hidden, act]) + act
        assert critic_params == _mlp_params([obs, *hidden, 1])
        raw = yaml.safe_load((MUJOCO / f"{env}_{algo}.yaml").read_text())
        assert bool(raw.get("ortho_init", False)) is (env == "walker2d")


def test_latent_dim_null_survives_save_and_load_policy(tmp_path: Path) -> None:
    import gymnasium as gym
    import numpy as np

    import srl
    from srl import TD3, BaseAgent

    cfg = {
        "env_id": "Pendulum-v1",
        "algo": "td3",
        "encoders": [
            {
                "name": "actor_state_enc",
                "type": "mlp",
                "input_dim": 3,
                "latent_dim": None,
                "layers": [16],
            },
            {
                "name": "critic_state_enc",
                "type": "mlp",
                "input_dim": 3,
                "latent_dim": None,
                "layers": [],
            },
        ],
        "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
        "actor": {"name": "actor", "type": "deterministic", "action_dim": 1},
        "critic": {"name": "critic", "type": "twin_q", "action_dim": 1, "layers": [16]},
        "train": {"learning_starts": 50, "batch_size": 16, "buffer_size": 1000},
    }
    agent = TD3.from_config(cfg, env=gym.make("Pendulum-v1"))
    agent.learn(100)
    out = agent.save(tmp_path / "agent")
    obs = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    expected, _ = agent.predict(obs, deterministic=True)
    np.testing.assert_array_equal(srl.load_policy(out).predict(obs), expected)
    again = BaseAgent.load(out, device="cpu")
    np.testing.assert_array_equal(again.predict(obs, deterministic=True)[0], expected)


# ──────────────────────────────────────────────────────────────────────────────
# ortho_init (SB3 ActorCriticPolicy ortho_init=True)
# ──────────────────────────────────────────────────────────────────────────────


def _ppo_cfg(ortho: bool) -> dict:
    return {
        "ortho_init": ortho,
        "encoders": [
            {
                "name": "a",
                "type": "mlp",
                "input_dim": 7,
                "latent_dim": None,
                "layers": [{"out_features": 64, "activation": "tanh"}] * 2,
            },
            {
                "name": "c",
                "type": "mlp",
                "input_dim": 7,
                "latent_dim": None,
                "layers": [{"out_features": 64, "activation": "tanh"}] * 2,
            },
        ],
        "flows": ["a -> actor", "c -> critic"],
        "actor": {"name": "actor", "type": "gaussian", "action_dim": 3, "log_std_init": 0.0},
        "critic": {"name": "critic", "type": "value"},
    }


def _is_orthogonal_with_gain(w, gain: float) -> bool:
    import torch

    rows, cols = w.shape
    m = w @ w.T if rows <= cols else w.T @ w
    return torch.allclose(m, gain**2 * torch.eye(min(rows, cols)), atol=1e-4)


def test_ortho_init_uses_sb3_gains() -> None:
    import math

    import torch

    torch.manual_seed(0)
    model = ModelBuilder.from_dict(_ppo_cfg(True))
    for enc in model.encoders.values():
        for lin in [m for m in enc.modules() if isinstance(m, nn.Linear)]:
            assert _is_orthogonal_with_gain(lin.weight.detach(), math.sqrt(2))
            assert torch.count_nonzero(lin.bias) == 0
    assert _is_orthogonal_with_gain(model.actor.mean_head.weight.detach(), 0.01)
    assert _is_orthogonal_with_gain(model.critic.out.weight.detach(), 1.0)
    assert torch.count_nonzero(model.critic.out.bias) == 0


def test_ortho_init_off_by_default_and_round_trips() -> None:
    from srl.registry.config_schema import AgentModelConfig

    model = ModelBuilder.from_dict(_ppo_cfg(False))
    assert not _is_orthogonal_with_gain(model.actor.mean_head.weight.detach(), 0.01)
    cfg = AgentModelConfig.from_dict(_ppo_cfg(True))
    assert cfg.ortho_init is True
    assert AgentModelConfig.from_dict(cfg.to_dict()).ortho_init is True
    assert "ortho_init" not in AgentModelConfig.from_dict(_ppo_cfg(False)).to_dict()
    with pytest.raises(ConfigError, match="ortho_init"):
        AgentModelConfig.from_dict({**_ppo_cfg(False), "ortho_init": "yes"})
