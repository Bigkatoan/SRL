"""Minimal per-algorithm regression tests: one gradient step, on fake data.

For each of the 6 algorithms, builds a tiny model via ModelBuilder.from_dict,
fills the agent's buffer with enough synthetic transitions, runs exactly one
`update()`, and asserts the returned loss metrics are finite and that model
parameters actually moved. Not a convergence test -- a "the gradient step
doesn't crash / doesn't NaN / doesn't silently no-op" test, cheap enough to
run on every CI push.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from srl.registry.builder import ModelBuilder

OBS_DIM = 3
ACTION_DIM = 2
REPO_ROOT = Path(__file__).resolve().parent.parent


def _param_sum(model: torch.nn.Module) -> float:
    return float(sum(p.detach().abs().sum().item() for p in model.parameters()))


def _assert_finite_metrics(metrics: dict[str, float]) -> None:
    assert metrics, "update() returned no metrics -- did it skip the gradient step?"
    for key, value in metrics.items():
        assert np.isfinite(value), f"metric {key!r} is not finite: {value}"


def _onpolicy_model_dict(actor_type: str, critic_type: str) -> dict:
    return {
        "encoders": [
            {
                "name": "actor_state_enc",
                "type": "mlp",
                "input_dim": OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
            },
            {
                "name": "critic_state_enc",
                "type": "mlp",
                "input_dim": OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
            },
        ],
        "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
        "actor": {
            "name": "actor",
            "type": actor_type,
            "action_dim": ACTION_DIM,
            "log_std_init": -0.5,
            "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
        },
        "critic": {
            "name": "critic",
            "type": critic_type,
            "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
        },
    }


def _offpolicy_model_dict(actor_type: str, critic_type: str) -> dict:
    return {
        "encoders": [
            {
                "name": "actor_state_enc",
                "type": "mlp",
                "input_dim": OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
            },
            {
                "name": "critic_state_enc",
                "type": "mlp",
                "input_dim": OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
            },
        ],
        "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
        "actor": {
            "name": "actor",
            "type": actor_type,
            "action_dim": ACTION_DIM,
            "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
            **(
                {"log_std_min": -5.0, "log_std_max": 2.0}
                if actor_type == "squashed_gaussian"
                else {}
            ),
        },
        "critic": {
            "name": "critic",
            "type": critic_type,
            "action_dim": ACTION_DIM,
            "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
        },
    }


def _fake_rollout(rng: np.random.Generator) -> dict:
    return {"policy": rng.standard_normal(OBS_DIM).astype(np.float32)}


def _fill_rollout_buffer(agent, n_steps: int, num_envs: int, rng: np.random.Generator) -> None:
    for _ in range(n_steps):
        obs = {"policy": rng.standard_normal((num_envs, OBS_DIM)).astype(np.float32)}
        action = rng.standard_normal((num_envs, ACTION_DIM)).astype(np.float32)
        reward = rng.standard_normal(num_envs).astype(np.float32)
        done = np.zeros(num_envs, dtype=bool)
        value = rng.standard_normal((num_envs, 1)).astype(np.float32)
        log_prob = rng.standard_normal(num_envs).astype(np.float32)
        agent.buffer.add(
            obs=obs, action=action, reward=reward, done=done, value=value, log_prob=log_prob
        )
    agent.buffer.compute_returns_and_advantages(last_value=np.zeros(num_envs, dtype=np.float32))


def _fill_replay_buffer(agent, n_transitions: int, rng: np.random.Generator) -> None:
    for _ in range(n_transitions):
        obs = {"policy": rng.standard_normal(OBS_DIM).astype(np.float32)}
        next_obs = {"policy": rng.standard_normal(OBS_DIM).astype(np.float32)}
        action = rng.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
        reward = np.array([float(rng.standard_normal())], dtype=np.float32)
        done = np.array([False])
        truncated = np.array([False])
        agent.buffer.add(
            obs=obs, action=action, reward=reward, next_obs=next_obs, done=done, truncated=truncated
        )


def test_ppo_one_gradient_step_is_finite_and_moves_params() -> None:
    from srl.algorithms.ppo import PPO
    from srl.core.config import PPOConfig

    rng = np.random.default_rng(0)
    model = ModelBuilder.from_dict(_onpolicy_model_dict("gaussian", "value"))
    agent = PPO(
        model, config=PPOConfig(n_steps=8, num_envs=2, batch_size=4, n_epochs=2), device="cpu"
    )

    before = _param_sum(agent.model)
    _fill_rollout_buffer(agent, n_steps=8, num_envs=2, rng=rng)
    metrics = agent.update()
    after = _param_sum(agent.model)

    _assert_finite_metrics(metrics)
    assert before != after


def test_a2c_one_gradient_step_is_finite_and_moves_params() -> None:
    from srl.algorithms.a2c import A2C
    from srl.core.config import A2CConfig

    rng = np.random.default_rng(0)
    model = ModelBuilder.from_dict(_onpolicy_model_dict("gaussian", "value"))
    agent = A2C(model, config=A2CConfig(n_steps=6, num_envs=2, batch_size=6), device="cpu")

    before = _param_sum(agent.model)
    _fill_rollout_buffer(agent, n_steps=6, num_envs=2, rng=rng)
    metrics = agent.update()
    after = _param_sum(agent.model)

    _assert_finite_metrics(metrics)
    assert before != after


def test_a2c_update_ignores_batch_size_and_always_takes_one_full_rollout_step() -> None:
    """Regression test for the divergence bug: a mis-set `batch_size` (not
    scaled to `n_steps * num_envs`) used to split one rollout into several
    sequential minibatch gradient updates against increasingly stale
    advantages/log-probs -- confirmed by direct repro to blow up policy loss
    within a few thousand steps at num_envs=16 with the *documented default*
    `batch_size=5`. A2C.update() must always take exactly one gradient step
    over the full rollout, regardless of `cfg.batch_size`."""
    from srl.algorithms.a2c import A2C
    from srl.core.config import A2CConfig

    rng = np.random.default_rng(0)
    model = ModelBuilder.from_dict(_onpolicy_model_dict("gaussian", "value"))
    # n_steps * num_envs = 12, but batch_size is deliberately much smaller --
    # if update() honored it, this would loop 6 times (12 / 2) instead of 1.
    agent = A2C(model, config=A2CConfig(n_steps=6, num_envs=2, batch_size=2), device="cpu")

    _fill_rollout_buffer(agent, n_steps=6, num_envs=2, rng=rng)
    assert agent._global_step == 0
    agent.update()
    assert agent._global_step == 1  # exactly one optimizer step, not six


def test_a3c_worker_gradient_step_is_finite_and_moves_shared_model() -> None:
    """Exercises `_worker_fn`'s real loss/backward path in-process (no
    multiprocessing spawn -- too slow/flaky for a unit test), with a plain
    threading.Lock standing in for mp.Lock (both are context managers with no
    process-specific behaviour used here)."""
    import threading
    from types import SimpleNamespace

    from srl.algorithms.a3c import _worker_fn
    from srl.core.config import A3CConfig

    class _FakeEnv:
        def __init__(self) -> None:
            self._rng = np.random.default_rng(0)

        def reset(self):
            return {"policy": self._rng.standard_normal(OBS_DIM).astype(np.float32)}, {}

        def step(self, action):
            obs = {"policy": self._rng.standard_normal(OBS_DIM).astype(np.float32)}
            return obs, float(self._rng.standard_normal()), False, False, {}

        def close(self) -> None:
            pass

    class _StopAfterOneIteration:
        def __init__(self) -> None:
            self._calls = 0

        def is_set(self) -> bool:
            self._calls += 1
            return self._calls > 1

    shared_model = ModelBuilder.from_dict(_onpolicy_model_dict("gaussian", "value"))
    shared_model.share_memory()
    cfg = A3CConfig(n_steps=6, batch_size=6, n_workers=1)
    optimizer = torch.optim.Adam(shared_model.parameters(), lr=cfg.lr)

    before = _param_sum(shared_model)
    _worker_fn(
        rank=0,
        shared_model=shared_model,
        optimizer=optimizer,
        env_fn=_FakeEnv,
        config=cfg,
        counter=SimpleNamespace(value=0),
        lock=threading.Lock(),
        stop_event=_StopAfterOneIteration(),
        stats_queue=None,
    )
    after = _param_sum(shared_model)

    for p in shared_model.parameters():
        assert torch.isfinite(p).all()
    assert before != after


def test_sac_one_gradient_step_is_finite_and_moves_params() -> None:
    import copy

    from srl.algorithms.sac import SAC
    from srl.core.config import SACConfig

    rng = np.random.default_rng(0)
    model = ModelBuilder.from_dict(_offpolicy_model_dict("squashed_gaussian", "twin_q"))
    target = copy.deepcopy(model)
    agent = SAC(model, target, config=SACConfig(action_dim=ACTION_DIM, batch_size=8), device="cpu")

    before = _param_sum(agent.model)
    _fill_replay_buffer(agent, n_transitions=16, rng=rng)
    metrics = agent.update()
    after = _param_sum(agent.model)

    _assert_finite_metrics(metrics)
    assert before != after


def test_ddpg_one_gradient_step_is_finite_and_moves_params() -> None:
    import copy

    from srl.algorithms.ddpg import DDPG
    from srl.core.config import DDPGConfig

    rng = np.random.default_rng(0)
    model = ModelBuilder.from_dict(_offpolicy_model_dict("deterministic", "twin_q"))
    target = copy.deepcopy(model)
    agent = DDPG(
        model, target, config=DDPGConfig(action_dim=ACTION_DIM, batch_size=8), device="cpu"
    )

    before = _param_sum(agent.model)
    _fill_replay_buffer(agent, n_transitions=16, rng=rng)
    metrics = agent.update()
    after = _param_sum(agent.model)

    _assert_finite_metrics(metrics)
    assert before != after


def test_td3_one_gradient_step_is_finite_and_moves_params() -> None:
    import copy

    from srl.algorithms.td3 import TD3
    from srl.core.config import TD3Config

    rng = np.random.default_rng(0)
    model = ModelBuilder.from_dict(_offpolicy_model_dict("deterministic", "twin_q"))
    target = copy.deepcopy(model)
    agent = TD3(
        model,
        target,
        config=TD3Config(action_dim=ACTION_DIM, batch_size=8, policy_delay=1),
        device="cpu",
    )

    before = _param_sum(agent.model)
    _fill_replay_buffer(agent, n_transitions=16, rng=rng)
    metrics = agent.update()
    after = _param_sum(agent.model)

    _assert_finite_metrics(metrics)
    assert before != after


def test_ppo_visual_aux_loss_gradient_step_is_finite_and_moves_params() -> None:
    """Regression test: PPO's __init__ and update() both called
    nn.ModuleDict.get(), which doesn't exist (ModuleDict has no .get()) --
    constructing a VisualPPOConfig agent for ANY aux-loss config (autoencoder/
    contrastive/byol) crashed immediately, before a single gradient step.
    Uses the repo's own shipped example config as the fixture, since the bug
    is specifically about the encoder/aux-module wiring a real config
    produces, not something easy to fake with a toy dict."""
    from srl.algorithms.ppo import PPO
    from srl.core.config import VisualPPOConfig

    config_path = REPO_ROOT / "configs" / "envs" / "car_racing_ppo_visual.yaml"
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f)
    input_shape = raw_cfg["encoders"][0]["input_shape"]
    action_dim = raw_cfg["actor"]["action_dim"]

    model = ModelBuilder.from_yaml(config_path)
    agent = PPO(
        model,
        config=VisualPPOConfig(n_steps=4, num_envs=1, batch_size=4, n_epochs=1),
        device="cpu",
    )
    assert agent.encoder_optimizer is not None

    rng = np.random.default_rng(0)
    for _ in range(4):
        obs = {"policy": rng.standard_normal((1, *input_shape)).astype(np.float32)}
        agent.buffer.add(
            obs=obs,
            action=rng.standard_normal((1, action_dim)).astype(np.float32),
            reward=np.array([1.0], dtype=np.float32),
            done=np.array([False]),
            value=np.array([[0.0]], dtype=np.float32),
            log_prob=np.array([0.0], dtype=np.float32),
        )
    agent.buffer.compute_returns_and_advantages(last_value=np.zeros(1, dtype=np.float32))

    before = _param_sum(agent.model)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = agent.update()
    after = _param_sum(agent.model)

    _assert_finite_metrics(metrics)
    assert before != after
