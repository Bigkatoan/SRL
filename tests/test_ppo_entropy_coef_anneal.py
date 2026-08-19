"""Tests for PPOConfig.entropy_coef_anneal_steps (entropy bonus annealing).

Motivating bug: a real 40M-step PPO run against JAVIS's mjlab balance task
(with lr_schedule="adaptive" -- see test_ppo_adaptive_kl_lr.py -- already
enabled) declined from a step-11M peak for the rest of the run.
`ppo/entropy` climbed continuously and monotonically the whole time (-2.76
at step ~100k to +1.07 by step ~27.6M, still rising) -- the OPPOSITE of the
earlier entropy-COLLAPSE failure mode, but the same underlying shape of
problem: nothing bounds a monotonic drift over a long enough run, and a
policy getting steadily noisier degrades closed-loop control on a physical
balance task. `log_std` (via `DiagonalGaussian`) is correctly clamped to
[log_std_min, log_std_max], but was nowhere near that ceiling at step 27.6M
-- the entropy bonus's constant, one-directional pull just kept winning
against the policy gradient's counter-pressure. PPO.__init__ now wires
`entropy_coef_anneal_steps` through `LossComposer`'s existing
`schedule="linear_decay"` machinery (previously implemented but unused for
the entropy term), gated off by default (`entropy_coef_anneal_steps=None`)
so every existing run's behavior is preserved exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.registry.builder import ModelBuilder

OBS_DIM = 3
ACTION_DIM = 2


def _model_dict() -> dict:
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
            "type": "gaussian",
            "action_dim": ACTION_DIM,
            "log_std_init": -0.5,
            "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
        },
        "critic": {
            "name": "critic",
            "type": "value",
            "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
        },
    }


def _make_agent(**cfg_kwargs) -> PPO:
    model = ModelBuilder.from_dict(_model_dict())
    cfg = PPOConfig(n_steps=8, num_envs=2, batch_size=4, n_epochs=2, **cfg_kwargs)
    return PPO(model, config=cfg, device="cpu")


def _fill_buffer(agent: PPO, n_steps: int, num_envs: int, rng: np.random.Generator) -> None:
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


# ──────────────────────────────────────────────────────────────────────────
# Config defaults preserve old behavior.
# ──────────────────────────────────────────────────────────────────────────


def test_ppo_config_defaults_to_no_entropy_annealing() -> None:
    cfg = PPOConfig()
    assert cfg.entropy_coef_anneal_steps is None
    assert cfg.entropy_coef_final == 0.0


def test_disabled_by_default_entropy_weight_never_moves_across_updates() -> None:
    """Default (entropy_coef_anneal_steps=None) must reproduce every
    existing run's behavior exactly: the effective entropy weight never
    changes, no matter how many real update() calls run."""
    rng = np.random.default_rng(0)
    agent = _make_agent(entropy_coef=0.01)  # entropy_coef_anneal_steps defaults to None
    assert agent.cfg.entropy_coef_anneal_steps is None

    for _ in range(5):
        _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
        metrics = agent.update()
        assert metrics["entropy_weight"] == 0.01


# ──────────────────────────────────────────────────────────────────────────
# Wired into update(): annealing actually moves the effective weight.
# ──────────────────────────────────────────────────────────────────────────


def test_entropy_weight_anneals_toward_final_value_across_real_updates() -> None:
    """The effective entropy weight (`metrics["entropy_weight"]`, i.e. what
    LossComposer actually multiplied `entropy_loss` by -- not just a config
    field that happens to exist) must decrease across real update() calls
    when annealing is enabled, not stay pinned at its initial value."""
    rng = np.random.default_rng(0)
    # n_epochs=2, batch_size=4 over an 8-step/2-env buffer (16 transitions)
    # -> 4 minibatches/epoch * 2 epochs = 8 gradient steps per update() call.
    # anneal_steps=20 spans roughly 2.5 update() calls -- long enough to
    # observe real, in-progress decay (not already saturated at the floor)
    # within the handful of calls this test makes.
    agent = _make_agent(
        entropy_coef=0.01,
        entropy_coef_final=0.001,
        entropy_coef_anneal_steps=20,
    )
    initial_weight = None
    seen_weights: list[float] = []

    for _ in range(5):
        _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
        metrics = agent.update()
        seen_weights.append(metrics["entropy_weight"])
        if initial_weight is None:
            initial_weight = metrics["entropy_weight"]

    # `metrics["entropy_weight"]` is `update()`'s own mean over every
    # minibatch in that call (see PPO.update()'s final `sum(v) / len(v)`),
    # so even the very first call already reflects some in-call decay --
    # check it's close to (not exactly) the configured start, and that
    # later calls are clearly lower.
    assert initial_weight == pytest.approx(0.01, abs=2e-3)
    assert len(set(seen_weights)) > 1, "entropy weight never moved across 5 real updates"
    # Monotonically non-increasing (linear decay, no reason to ever go back up).
    assert all(a >= b for a, b in zip(seen_weights, seen_weights[1:], strict=False)), seen_weights
    # And bounded within [entropy_coef_final, entropy_coef] throughout.
    assert all(0.001 <= w <= 0.01 for w in seen_weights), seen_weights


def test_entropy_weight_reaches_and_holds_final_value_past_anneal_horizon() -> None:
    """Once `self._global_step` (gradient steps) passes
    `entropy_coef_anneal_steps`, the effective weight must be pinned at
    `entropy_coef_final` -- not keep decaying past it, and not bounce back
    up."""
    rng = np.random.default_rng(1)
    agent = _make_agent(
        entropy_coef=0.02,
        entropy_coef_final=0.0,
        entropy_coef_anneal_steps=4,  # well under 1 update() call's worth of steps
    )

    for _ in range(6):  # far past the anneal horizon in gradient-step terms
        _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
        metrics = agent.update()

    assert agent._global_step > agent.cfg.entropy_coef_anneal_steps
    assert metrics["entropy_weight"] == 0.0
