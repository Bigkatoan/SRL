"""Regression tests for on-policy (PPO / A2C / A3C) correctness fixes.

Each test here failed against commit aa634cf and pins one specific bug:
see the individual docstrings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from srl.algorithms.a2c import A2C
from srl.algorithms.ppo import PPO
from srl.core.config import A2CConfig, PPOConfig, VisualPPOConfig
from srl.losses.rl_losses import ppo_value_loss
from srl.registry.builder import ModelBuilder

OBS_DIM = 3
ACTION_DIM = 2


def _model_dict(**actor_overrides) -> dict:
    actor = {
        "name": "actor",
        "type": "gaussian",
        "action_dim": ACTION_DIM,
        "log_std_init": -0.5,
        "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
    }
    actor.update(actor_overrides)
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
        "actor": actor,
        "critic": {
            "name": "critic",
            "type": "value",
            "layers": [{"out_features": 8, "activation": "tanh", "norm": "none"}],
        },
    }


def _fill_buffer(agent, n_steps: int, num_envs: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n_steps):
        obs = {"state": rng.standard_normal((num_envs, OBS_DIM)).astype(np.float32)}
        action, log_prob, value, _ = agent.predict(
            {"state": torch.from_numpy(obs["state"])}
        )
        agent.buffer.add(
            obs=obs,
            action=action.numpy(),
            reward=rng.standard_normal(num_envs).astype(np.float32),
            done=np.zeros(num_envs, dtype=bool),
            log_prob=log_prob.numpy(),
            value=value.numpy(),
        )
    agent.buffer.compute_returns_and_advantages(last_value=np.zeros(num_envs))


def _encoder_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        f"{enc_name}.{n}": p.detach().clone()
        for enc_name, enc in model.encoders.items()
        for n, p in enc.named_parameters()
    }


# ---------------------------------------------------------------------------
# Bug 1: PPO detached encoder latents unconditionally -> encoders never trained
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_cls", [PPOConfig, VisualPPOConfig])
def test_ppo_update_trains_encoders(cfg_cls):
    torch.manual_seed(0)
    model = ModelBuilder.from_dict(_model_dict())
    cfg = cfg_cls(n_steps=16, num_envs=2, batch_size=8, n_epochs=2)
    agent = PPO(model, config=cfg, device="cpu")
    before = _encoder_snapshot(model)
    _fill_buffer(agent, 16, 2)
    agent.update()
    after = _encoder_snapshot(model)
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    # Every encoder (actor-side AND critic-side) must receive gradient.
    for enc_name in model.encoders:
        assert any(k.startswith(enc_name + ".") for k in changed), enc_name


def test_a2c_update_trains_encoders():
    torch.manual_seed(0)
    model = ModelBuilder.from_dict(_model_dict())
    agent = A2C(model, config=A2CConfig(n_steps=8, num_envs=2), device="cpu")
    before = _encoder_snapshot(model)
    _fill_buffer(agent, 8, 2)
    agent.update()
    after = _encoder_snapshot(model)
    assert any(not torch.equal(before[k], after[k]) for k in before)


# ---------------------------------------------------------------------------
# Bug 5: value clipping took max over batch MEANS instead of elementwise max
# ---------------------------------------------------------------------------


def test_ppo_value_loss_clipping_is_elementwise():
    values = torch.tensor([1.0, 0.0])
    old_values = torch.tensor([0.0, 0.0])
    returns = torch.tensor([0.0, 1.0])
    clip = 0.2
    # Element 0: unclipped err (1-0)^2 = 1,    clipped v=0.2 -> 0.04  -> max 1
    # Element 1: unclipped err (0-1)^2 = 1,    clipped v=0.0 -> 1     -> max 1
    expected = torch.tensor(1.0)
    got = ppo_value_loss(values, returns, old_values=old_values, clip_eps=clip)
    assert torch.allclose(got, expected)

    values = torch.tensor([1.0, 0.5])
    old_values = torch.tensor([0.0, 0.0])
    returns = torch.tensor([0.0, 0.5])
    # elem0: unclipped 1, clipped (0.2)^2=0.04 -> 1
    # elem1: unclipped 0, clipped (0.2-0.5)^2=0.09 -> 0.09
    expected = torch.tensor((1.0 + 0.09) / 2)
    got = ppo_value_loss(values, returns, old_values=old_values, clip_eps=clip)
    assert torch.allclose(got, expected)
    # (old buggy formula: max(mean(1,0)=0.5, mean(0.04,0.09)=0.065) = 0.5)


def test_ppo_value_loss_none_clip_is_plain_mse():
    values = torch.tensor([1.0, 0.5])
    old_values = torch.tensor([0.0, 0.0])
    returns = torch.tensor([0.0, 0.5])
    got = ppo_value_loss(values, returns, old_values=old_values, clip_eps=None)
    assert torch.allclose(got, torch.tensor(0.5))


def test_ppo_config_clip_range_vf_defaults_to_none():
    assert PPOConfig().clip_range_vf is None


# ---------------------------------------------------------------------------
# Bug 7: partial final rollout -> GAE / minibatches ran over empty slots
# ---------------------------------------------------------------------------


def test_rollout_buffer_partial_rollout_uses_only_filled_slots():
    from srl.core.rollout_buffer import RolloutBuffer

    gamma, lam = 0.9, 0.8
    buf = RolloutBuffer(capacity=5, num_envs=1, gamma=gamma, lam=lam)
    rewards = [1.0, 2.0]
    values = [0.5, 0.25]
    for r, v in zip(rewards, values, strict=True):
        buf.add(
            obs={"state": np.ones(OBS_DIM, dtype=np.float32)},
            action=np.zeros(ACTION_DIM, dtype=np.float32),
            reward=r,
            done=False,
            value=v,
            log_prob=0.0,
        )
    last_value = 3.0
    buf.compute_returns_and_advantages(last_value=last_value)

    d1 = rewards[1] + gamma * last_value - values[1]
    d0 = rewards[0] + gamma * values[1] - values[0]
    a1 = d1
    a0 = d0 + gamma * lam * a1
    np.testing.assert_allclose(buf.advantages[:2, 0], [a0, a1], rtol=1e-6)

    batches = list(buf.get_batches(batch_size=100))
    assert sum(b.actions.shape[0] for b in batches) == 2
    assert buf.get_batch().actions.shape[0] == 2


# ---------------------------------------------------------------------------
# Bug 6: state_dependent_std defaulted True and was never passed to the head
# ---------------------------------------------------------------------------


def test_gaussian_head_defaults_to_state_independent_std():
    model = ModelBuilder.from_dict(_model_dict())
    assert model.actor.state_dependent_std is False
    assert model.actor.log_std_head is None


def test_state_dependent_std_flag_is_passed_through_builder():
    model = ModelBuilder.from_dict(_model_dict(state_dependent_std=True))
    assert model.actor.state_dependent_std is True
    assert model.actor.log_std_head is not None
