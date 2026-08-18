"""Tests for PPOConfig.lr_schedule="adaptive" (PPO._adapt_lr).

Motivating bug: a real 20M-step PPO run against JAVIS's mjlab balance task
declined continuously for most of the run after an early peak, while
`ppo/approx_kl` stayed inside a superficially "healthy" band the whole time
-- healthy only because nothing was actually enforcing a target, since
`target_kl` (PPOConfig's *existing* knob) defaulted to None (disabled) and
even when set is only a same-epoch early stop, not a persistent bound on
future updates. The same task trains fine (converges and holds) under
mjlab's own reference PPO (rsl_rl), which applies a continuous adaptive
learning-rate schedule keyed on measured KL divergence every minibatch, for
the entire run: too far above `desired_kl` shrinks the LR, too far below
grows it back, both clamped to [min_lr, max_lr]. PPO._adapt_lr ports that
mechanism into SRL, gated behind `lr_schedule="adaptive"` (default
"fixed" -- off, preserving every existing run's behavior exactly).
"""

from __future__ import annotations

import numpy as np

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
# _adapt_lr in isolation -- exact port of rsl_rl PPO's thresholds/clamps.
# ──────────────────────────────────────────────────────────────────────────


def test_adapt_lr_shrinks_when_kl_far_above_desired() -> None:
    agent = _make_agent(lr=1e-3, lr_schedule="adaptive", desired_kl=0.01, kl_lr_factor=1.5)
    agent._adapt_lr(kl_mean=0.03)  # > desired_kl * 2.0
    assert agent._current_lr == 1e-3 / 1.5
    assert agent.optimizer.param_groups[0]["lr"] == 1e-3 / 1.5


def test_adapt_lr_grows_when_kl_far_below_desired() -> None:
    agent = _make_agent(lr=1e-3, lr_schedule="adaptive", desired_kl=0.01, kl_lr_factor=1.5)
    agent._adapt_lr(kl_mean=0.001)  # < desired_kl / 2.0
    assert agent._current_lr == 1e-3 * 1.5
    assert agent.optimizer.param_groups[0]["lr"] == 1e-3 * 1.5


def test_adapt_lr_unchanged_in_dead_zone() -> None:
    agent = _make_agent(lr=1e-3, lr_schedule="adaptive", desired_kl=0.01, kl_lr_factor=1.5)
    agent._adapt_lr(kl_mean=0.01)  # exactly at desired_kl -- neither threshold trips
    assert agent._current_lr == 1e-3


def test_adapt_lr_respects_min_lr_clamp() -> None:
    agent = _make_agent(
        lr=1e-5, lr_schedule="adaptive", desired_kl=0.01, min_lr=1e-5, kl_lr_factor=1.5
    )
    agent._adapt_lr(kl_mean=1.0)  # way above desired_kl*2 -- would shrink below min_lr
    assert agent._current_lr == 1e-5  # clamped, not 1e-5 / 1.5


def test_adapt_lr_respects_max_lr_clamp() -> None:
    agent = _make_agent(
        lr=1e-2, lr_schedule="adaptive", desired_kl=0.01, max_lr=1e-2, kl_lr_factor=1.5
    )
    agent._adapt_lr(kl_mean=1e-6)  # way below desired_kl/2 -- would grow past max_lr
    assert agent._current_lr == 1e-2  # clamped, not 1e-2 * 1.5


def test_adapt_lr_zero_kl_does_not_grow() -> None:
    """rsl_rl's own guard: `kl_mean < desired_kl / 2.0 and kl_mean > 0.0` --
    a exactly-zero KL (e.g. before any real divergence has been measured)
    must not trigger growth."""
    agent = _make_agent(lr=1e-3, lr_schedule="adaptive", desired_kl=0.01)
    agent._adapt_lr(kl_mean=0.0)
    assert agent._current_lr == 1e-3


def test_adapt_lr_repeated_shrinks_stay_within_bounds() -> None:
    agent = _make_agent(
        lr=1e-3, lr_schedule="adaptive", desired_kl=0.01, min_lr=1e-5, kl_lr_factor=1.5
    )
    for _ in range(50):
        agent._adapt_lr(kl_mean=10.0)
    assert agent._current_lr == 1e-5
    assert agent.optimizer.param_groups[0]["lr"] == 1e-5


# ──────────────────────────────────────────────────────────────────────────
# Wired into update(): fixed (default) leaves LR untouched; adaptive moves it.
# ──────────────────────────────────────────────────────────────────────────


def test_fixed_schedule_never_touches_lr_across_many_updates() -> None:
    """Default lr_schedule="fixed" must reproduce every existing run's
    behavior exactly: the optimizer LR never moves, no matter what KL comes
    out of real updates."""
    rng = np.random.default_rng(0)
    agent = _make_agent(lr=3e-4)  # lr_schedule defaults to "fixed"
    assert agent.cfg.lr_schedule == "fixed"

    for _ in range(5):
        _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
        metrics = agent.update()
        assert agent.optimizer.param_groups[0]["lr"] == 3e-4
        # ppo/lr is reported either way (useful visibility even in fixed
        # mode -- confirms it's genuinely constant), but under "fixed" it
        # must always read back exactly the configured, never-adapted lr.
        assert metrics["lr"] == 3e-4


def test_adaptive_schedule_changes_lr_across_updates() -> None:
    """With genuinely random (large-KL-inducing) minibatches against a
    freshly-initialized policy, adaptive scheduling must actually move the
    LR away from its initial value at some point across several real
    update() calls -- not just in the isolated _adapt_lr unit tests above."""
    rng = np.random.default_rng(0)
    agent = _make_agent(lr=1e-3, lr_schedule="adaptive", desired_kl=0.01, kl_lr_factor=1.5)
    initial_lr = agent._current_lr

    seen_lrs = {initial_lr}
    for _ in range(10):
        _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
        metrics = agent.update()
        seen_lrs.add(agent.optimizer.param_groups[0]["lr"])
        assert "lr" in metrics
        assert np.isfinite(metrics["lr"])
        assert agent.cfg.min_lr <= agent.optimizer.param_groups[0]["lr"] <= agent.cfg.max_lr

    assert len(seen_lrs) > 1, "adaptive schedule never moved the LR across 10 real updates"


# ──────────────────────────────────────────────────────────────────────────
# Backward compatibility: config defaults preserve old behavior.
# ──────────────────────────────────────────────────────────────────────────


def test_ppo_config_defaults_to_fixed_schedule() -> None:
    cfg = PPOConfig()
    assert cfg.lr_schedule == "fixed"
    assert cfg.desired_kl == 0.01
    assert cfg.min_lr == 1e-5
    assert cfg.max_lr == 1e-2
    assert cfg.kl_lr_factor == 1.5


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint round-trip of the adapted LR (--resume correctness).
# ──────────────────────────────────────────────────────────────────────────


def test_current_lr_round_trips_through_checkpoint_payload() -> None:
    agent = _make_agent(lr=1e-3, lr_schedule="adaptive", desired_kl=0.01, kl_lr_factor=1.5)
    agent._adapt_lr(kl_mean=1.0)  # drives _current_lr away from the initial 1e-3
    adapted_lr = agent._current_lr
    assert adapted_lr != 1e-3

    payload = agent.checkpoint_payload()
    assert payload["current_lr"] == adapted_lr

    fresh_agent = _make_agent(lr=1e-3, lr_schedule="adaptive", desired_kl=0.01, kl_lr_factor=1.5)
    assert fresh_agent._current_lr == 1e-3  # sanity: starts at cfg.lr, not yet adapted
    fresh_agent.load_checkpoint_payload(payload)
    assert fresh_agent._current_lr == adapted_lr

    # And the *next* adaptation on the resumed agent must continue from the
    # restored value, not silently reset to cfg.lr=1e-3.
    fresh_agent._adapt_lr(kl_mean=1.0)
    assert fresh_agent._current_lr == adapted_lr / 1.5


def test_checkpoint_payload_without_current_lr_key_is_backward_compatible() -> None:
    """Checkpoints saved before this feature existed have no "current_lr"
    key -- loading one must not raise, and must leave _current_lr at
    whatever the fresh agent already initialized it to."""
    agent = _make_agent(lr=5e-4)
    old_style_payload = {
        "model_state": agent.model.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "algo_step": 100,
    }
    agent.load_checkpoint_payload(old_style_payload)  # must not raise
    assert agent._current_lr == 5e-4
    assert agent._global_step == 100
