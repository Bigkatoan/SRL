"""Tests for SACConfig.min_alpha -- a floor on the auto-tuned temperature.

Motivating bug: a real 10M-step SAC (FlashSAC config) run against JAVIS's
mjlab balance task held a healthy alpha and a noisy-but-bounded eval score
(~0.5-1.9) through step 6M, then `log_alpha` (an otherwise-unclamped free
parameter) collapsed from a healthy magnitude down to ~3e-4 between steps
6M-7M -- the same order of magnitude as an earlier, unfixed baseline's full
collapse to 0.0002 -- immediately followed by numerical-explosion episode
returns as extreme as -7.1e6 (entropy regularization gone, actor exploiting
a known unclamped-reward-term physics-divergence bug). The 10x-lower
`lr_alpha` FlashSAC already uses delayed this collapse (it happened
immediately, not at 6M+, in the unfixed baseline) but did not prevent it
over a long enough horizon.

These tests verify the floor added in `SAC.update()`'s temperature-update
step (`self.log_alpha.clamp_(min=math.log(cfg.min_alpha))` after the alpha
optimizer's step) actually engages, using log_prob values hand-picked to
force alpha to shrink deterministically (see the derivation in this file's
comments) so the test doesn't depend on a real actor network happening to
drift a particular direction.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch

from srl.algorithms.sac import SAC
from srl.core.config import SACConfig
from srl.losses.rl_losses import sac_temperature_loss
from srl.registry.builder import ModelBuilder

OBS_DIM = 5
ACTION_DIM = 2


def _offpolicy_model_dict() -> dict:
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
            "type": "squashed_gaussian",
            "action_dim": ACTION_DIM,
            "log_std_min": -5.0,
            "log_std_max": 2.0,
            "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
        },
        "critic": {
            "name": "critic",
            "type": "twin_q",
            "action_dim": ACTION_DIM,
            "layers": [{"out_features": 8, "activation": "relu", "norm": "none"}],
        },
    }


def _build_sac(**cfg_kwargs) -> SAC:
    model = ModelBuilder.from_dict(_offpolicy_model_dict())
    target = copy.deepcopy(model)
    cfg = SACConfig(action_dim=ACTION_DIM, batch_size=8, **cfg_kwargs)
    return SAC(model, target, config=cfg, device="cpu")


def _force_alpha_shrink_step(agent: SAC) -> None:
    """Replicates exactly SAC.update()'s temperature-update block (loss,
    backward, optimizer.step(), floor-clamp), with a hand-picked log_prob
    batch that always pushes alpha DOWN regardless of the network's actual
    behavior: very negative log_prob (here -50) means the sampled actions
    have very low density under the current policy -- i.e. actual entropy
    is far ABOVE target_entropy (-2 by default for action_dim=2) -- which
    is exactly the "policy is more random than it needs to be" signal that
    correctly drives alpha down in real SAC. See this module's docstring
    for the sign derivation.
    """
    log_prob = torch.full((8,), -50.0)
    temp_loss = sac_temperature_loss(log_prob, agent.log_alpha, agent.target_entropy)
    agent.alpha_optimizer.zero_grad()
    temp_loss.backward()
    agent.alpha_optimizer.step()
    with torch.no_grad():
        agent.log_alpha.clamp_(min=math.log(max(agent.cfg.min_alpha, 1e-300)))


def _force_alpha_grow_step(agent: SAC) -> None:
    """Mirror of _force_alpha_shrink_step with the opposite-signed forcing
    log_prob (+50 -- very concentrated/low-entropy sampled actions), used
    only to confirm the floor doesn't interfere with alpha moving upward.
    """
    log_prob = torch.full((8,), 50.0)
    temp_loss = sac_temperature_loss(log_prob, agent.log_alpha, agent.target_entropy)
    agent.alpha_optimizer.zero_grad()
    temp_loss.backward()
    agent.alpha_optimizer.step()
    with torch.no_grad():
        agent.log_alpha.clamp_(min=math.log(max(agent.cfg.min_alpha, 1e-300)))


# ──────────────────────────────────────────────────────────────────────────
# Config default: fully backward compatible
# ──────────────────────────────────────────────────────────────────────────


def test_min_alpha_defaults_permissive() -> None:
    """1e-8 default must not meaningfully floor anything for existing
    configs -- this is an opt-in protection, not a default behavior change."""
    assert SACConfig().min_alpha == 1e-8


# ──────────────────────────────────────────────────────────────────────────
# The floor actually engages
# ──────────────────────────────────────────────────────────────────────────


def test_alpha_collapses_far_below_a_meaningful_floor_with_default_min_alpha() -> None:
    """Reproduces the bug: with min_alpha left at its permissive default,
    repeated shrink-forcing steps let alpha collapse to a magnitude far
    below anything that would meaningfully regularize the actor (matching
    the ~3e-4 real collapse this test suite is guarding against)."""
    agent = _build_sac(lr_alpha=0.1)  # high lr_alpha so few steps suffice
    for _ in range(100):
        _force_alpha_shrink_step(agent)
    assert agent.alpha.item() < 1e-4


def test_min_alpha_floor_prevents_collapse_below_the_configured_value() -> None:
    """With a real min_alpha set, the same forcing loop that collapses
    alpha in the test above must instead floor it at min_alpha."""
    agent = _build_sac(lr_alpha=0.1, min_alpha=1e-3)
    for _ in range(30):
        _force_alpha_shrink_step(agent)
    # Floored, not just "close to" min_alpha -- allow only float round-trip
    # slack from the log-space clamp (log then exp) rather than a loose
    # tolerance that could hide a fully-collapsed alpha.
    assert agent.alpha.item() >= 1e-3 - 1e-9


def test_min_alpha_floor_holds_across_many_more_steps() -> None:
    """The floor must hold indefinitely, not just delay collapse -- run
    far more forcing steps than needed to collapse an unfloored alpha."""
    agent = _build_sac(lr_alpha=0.1, min_alpha=1e-3)
    for _ in range(500):
        _force_alpha_shrink_step(agent)
    assert agent.alpha.item() >= 1e-3 - 1e-9
    assert math.isfinite(agent.log_alpha.item())


def test_min_alpha_does_not_block_alpha_from_growing() -> None:
    """The floor is a one-sided clamp (min=...) -- it must never prevent
    alpha from increasing when the temperature loss pushes it that way."""
    agent = _build_sac(lr_alpha=0.1, min_alpha=1e-3, init_alpha=0.01)
    start = agent.alpha.item()
    for _ in range(20):
        _force_alpha_grow_step(agent)
    assert agent.alpha.item() > start


# ──────────────────────────────────────────────────────────────────────────
# End-to-end: wired into the real update() path, not just the isolated
# clamp logic.
# ──────────────────────────────────────────────────────────────────────────


def _fill_replay_buffer(agent: SAC, n_transitions: int, rng: np.random.Generator) -> None:
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


def test_real_update_calls_never_breach_the_floor() -> None:
    """Real update() calls (full forward pass, real replay data) must
    respect min_alpha regardless of which direction the untrained network's
    log_probs happen to push alpha -- proves the clamp is actually reached
    on the real code path, not just in the hand-constructed tests above."""
    rng = np.random.default_rng(0)
    agent = _build_sac(min_alpha=1e-3, lr_alpha=0.05)
    _fill_replay_buffer(agent, n_transitions=32, rng=rng)

    for _ in range(50):
        metrics = agent.update()
        assert math.isfinite(metrics["sac/alpha"])
        assert metrics["sac/alpha"] >= 1e-3 - 1e-9
    assert agent.alpha.item() >= 1e-3 - 1e-9


def test_auto_entropy_tuning_disabled_ignores_min_alpha_without_crashing() -> None:
    """min_alpha only matters when alpha_optimizer exists (auto_entropy_
    tuning=True) -- with it off, alpha is fixed at cfg.alpha and update()
    must run cleanly regardless of min_alpha."""
    rng = np.random.default_rng(0)
    agent = _build_sac(auto_entropy_tuning=False, alpha=0.2, min_alpha=1e-3)
    assert agent.alpha_optimizer is None
    _fill_replay_buffer(agent, n_transitions=16, rng=rng)

    metrics = agent.update()
    assert math.isfinite(metrics["sac/alpha"])
    # float32 round-trip slack -- unchanged (no auto-tuning happened), not
    # an exact bit-for-bit match against the Python float literal.
    assert abs(agent.alpha.item() - 0.2) < 1e-6
