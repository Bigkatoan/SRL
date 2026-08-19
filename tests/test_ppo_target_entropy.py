"""Tests for PPOConfig.target_entropy (adaptive, self-correcting entropy
coefficient).

Background / correction this feature is built on: `entropy_coef_anneal_
steps`/`entropy_coef_final` (see test_ppo_entropy_coef_anneal.py) were
originally motivated by real runs where `ppo/entropy` appeared to climb
without bound. That appearance was a sign-reading error, not a real
runaway: `ppo/entropy` (via `entropy_loss(ent) = -ent.mean()`, logged as-is
with no further sign change) is the NEGATIVE of the true mean entropy.
Confirmed directly: this task's actor (log_std_init=0.0, action_dim=2) has
a theoretical initial entropy of ~2.838, and every real run's first logged
`ppo/entropy` value is ~-2.76 -- matching -2.838, not +2.838. The real
phenomenon across every run tested is entropy COLLAPSE (declining from
~+2.8 toward the log_std_min-derived floor), and a fixed/annealed
`entropy_coef` is fundamentally a one-directional dial: it can push true
entropy up, never pull an over-shoot back down, so there's no setting that
self-corrects in both directions -- unlike SAC's auto-tuned `alpha`.
`target_entropy` gives PPO that same self-correcting property: `entropy_
coef` becomes a learned parameter driven by dual-ascent toward whatever
value makes measured (TRUE, correctly-signed) entropy track a target.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

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
            "state_dependent_std": False,
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
        obs = {"actor": rng.standard_normal((num_envs, OBS_DIM)).astype(np.float32)}
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


def test_ppo_config_defaults_to_no_target_entropy() -> None:
    cfg = PPOConfig()
    assert cfg.target_entropy is None


def test_disabled_by_default_uses_fixed_entropy_coef_composer_path() -> None:
    """target_entropy=None must reproduce the old fixed-entropy_coef
    behavior exactly -- no log_entropy_coef parameter, no optimizer."""
    agent = _make_agent(entropy_coef=0.01)
    assert agent.log_entropy_coef is None
    assert agent.entropy_coef_optimizer is None
    assert not agent._target_entropy_active

    rng = np.random.default_rng(0)
    _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
    metrics = agent.update()
    assert metrics["entropy_weight"] == 0.01


# ──────────────────────────────────────────────────────────────────────────
# Direction of the dual-ascent correction.
# ──────────────────────────────────────────────────────────────────────────


# `_adapt_entropy_coef` in isolation -- mirrors `_adapt_lr`'s own isolated
# tests (test_ppo_adaptive_kl_lr.py): calling it directly with a controlled
# `true_entropy` value is far more reliable than driving the same direction
# through several full, noisy update() calls on a tiny toy network, where
# the ordinary policy-gradient noise (and, per this task's own real-run
# findings, a genuine independent effect on log_std from the policy
# gradient itself) can dominate a small number of minibatches.


def test_adapt_entropy_coef_shrinks_when_entropy_above_target() -> None:
    agent = _make_agent(target_entropy=0.0, entropy_coef=0.01, entropy_coef_lr=0.05)
    before = agent.log_entropy_coef.exp().item()
    agent._adapt_entropy_coef(torch.tensor(2.0))  # entropy > target=0.0
    after = agent.log_entropy_coef.exp().item()
    assert after < before


def test_adapt_entropy_coef_grows_when_entropy_below_target() -> None:
    agent = _make_agent(target_entropy=3.0, entropy_coef=0.01, entropy_coef_lr=0.05)
    before = agent.log_entropy_coef.exp().item()
    agent._adapt_entropy_coef(torch.tensor(-2.0))  # entropy < target=3.0
    after = agent.log_entropy_coef.exp().item()
    assert after > before


def test_adapt_entropy_coef_unchanged_at_exact_target() -> None:
    agent = _make_agent(target_entropy=1.5, entropy_coef=0.01, entropy_coef_lr=0.05)
    before = agent.log_entropy_coef.exp().item()
    agent._adapt_entropy_coef(torch.tensor(1.5))  # entropy == target -> zero gradient
    after = agent.log_entropy_coef.exp().item()
    assert after == pytest.approx(before)


def test_adapt_entropy_coef_repeated_growth_stays_within_max_bound() -> None:
    agent = _make_agent(
        target_entropy=10.0, entropy_coef=0.01, entropy_coef_lr=0.5, max_entropy_coef=0.2
    )
    for _ in range(50):
        agent._adapt_entropy_coef(torch.tensor(-10.0))  # always far below target
    assert agent.log_entropy_coef.exp().item() == pytest.approx(0.2, rel=1e-3)


def test_adapt_entropy_coef_repeated_shrink_stays_within_min_bound() -> None:
    agent = _make_agent(
        target_entropy=-10.0, entropy_coef=0.01, entropy_coef_lr=0.5, min_entropy_coef=1e-4
    )
    for _ in range(50):
        agent._adapt_entropy_coef(torch.tensor(10.0))  # always far above target
    assert agent.log_entropy_coef.exp().item() == pytest.approx(1e-4, rel=1e-3)


def test_entropy_coef_actually_moves_across_real_update_calls() -> None:
    """Not isolating direction here (see above) -- just confirms the full
    update() pipeline actually calls _adapt_entropy_coef and the returned
    entropy_weight genuinely changes across real calls, the same "wired
    into update(), not just correct in isolation" check
    test_ppo_adaptive_kl_lr.py's test_adaptive_schedule_changes_lr_across_
    updates does for the LR schedule.
    """
    rng = np.random.default_rng(0)
    agent = _make_agent(target_entropy=1.0, entropy_coef=0.01, entropy_coef_lr=0.01)
    seen = set()
    for _ in range(6):
        _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
        metrics = agent.update()
        seen.add(round(metrics["entropy_weight"], 8))
        assert "true_entropy" in metrics
        assert "entropy_coef_loss" in metrics
    assert len(seen) > 1, "entropy_weight never moved across 6 real updates"


def test_entropy_coef_bounded_by_min_and_max() -> None:
    rng = np.random.default_rng(2)
    agent = _make_agent(
        target_entropy=0.0,
        entropy_coef=0.01,
        entropy_coef_lr=1.0,  # aggressive, to actually hit the bounds fast
        min_entropy_coef=1e-4,
        max_entropy_coef=1e-2,
    )
    for _ in range(10):
        _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
        metrics = agent.update()
        assert 1e-4 - 1e-9 <= metrics["entropy_weight"] <= 1e-2 + 1e-9


# ──────────────────────────────────────────────────────────────────────────
# Uses the correctly-signed entropy value, not the negated loss.
# ──────────────────────────────────────────────────────────────────────────


def test_true_entropy_metric_is_not_sign_flipped() -> None:
    """metrics["true_entropy"] must be the real mean entropy (positive
    near a freshly-initialized moderate-std Gaussian), not `entropy_loss`'s
    negated value -- this is the exact bug this whole feature exists to
    avoid re-introducing.
    """
    agent = _make_agent(target_entropy=0.0, entropy_coef=0.01)
    rng = np.random.default_rng(0)
    _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
    metrics = agent.update()

    theoretical_init_entropy = ACTION_DIM * 0.5 * math.log(2 * math.pi * math.e) - 0.5 * ACTION_DIM
    # log_std_init=-0.5 for this model -- true_entropy should be in the
    # right ballpark of the theoretical value (not exact: a few gradient
    # steps have already run), and unambiguously POSITIVE, not the ~-1.8
    # a sign-flipped read would produce.
    assert metrics["true_entropy"] > 0.5
    assert abs(metrics["true_entropy"] - theoretical_init_entropy) < 2.0


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint round-trip.
# ──────────────────────────────────────────────────────────────────────────


def test_log_entropy_coef_round_trips_through_checkpoint_payload() -> None:
    agent = _make_agent(target_entropy=0.0, entropy_coef=0.01, entropy_coef_lr=0.05)
    rng = np.random.default_rng(0)
    _fill_buffer(agent, n_steps=8, num_envs=2, rng=rng)
    agent.update()
    moved_value = agent.log_entropy_coef.item()
    assert moved_value != math.log(0.01)  # sanity: actually moved from init

    payload = agent.checkpoint_payload()
    assert payload["log_entropy_coef"] is not None
    assert pytest.approx(payload["log_entropy_coef"].item()) == moved_value

    fresh_agent = _make_agent(target_entropy=0.0, entropy_coef=0.01, entropy_coef_lr=0.05)
    assert fresh_agent.log_entropy_coef.item() == pytest.approx(math.log(0.01))
    fresh_agent.load_checkpoint_payload(payload)
    assert fresh_agent.log_entropy_coef.item() == pytest.approx(moved_value)


def test_checkpoint_payload_without_log_entropy_coef_key_is_backward_compatible() -> None:
    """Checkpoints saved before this feature existed (or saved by a
    fixed/annealed-entropy_coef run) have no "log_entropy_coef" key --
    loading one into a target_entropy=None agent must not raise."""
    agent = _make_agent(entropy_coef=0.01)  # target_entropy=None
    old_style_payload = {
        "model_state": agent.model.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "algo_step": 100,
    }
    agent.load_checkpoint_payload(old_style_payload)  # must not raise
    assert agent.log_entropy_coef is None
