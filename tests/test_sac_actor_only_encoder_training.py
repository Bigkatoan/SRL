"""Regression test for a real, pre-existing SAC bug: an encoder that feeds
ONLY the actor (a config with separate per-head encoders, e.g. a `flows:`
graph like `["actor_state_enc -> actor", "critic_state_enc -> critic"]`)
never received a gradient update any optimizer actually consumed.

Why: `encoder_optimizer` trains encoders off critic_loss's backward pass
alone (DrQ-v2 style, so actor_loss can never "contaminate" a shared
encoder). But `compute_actor=False` critic-only forward passes correctly
prune encoders that don't feed the critic head -- so an actor-only encoder
is never even part of critic_loss's graph, and gets exactly zero gradient
from it. The only place a REAL gradient into it exists is actor_loss's own
backward pass (which does run through it) -- but that contribution was
unconditionally zeroed out before `encoder_optimizer.step()` could ever
fire again and consume it, by the same "actor loss never touches encoder
weights" rule that (correctly) protects a *shared* encoder.

Net effect, confirmed directly before the fix: such an encoder's weights
never moved a single bit from random initialization across real
`update()` calls -- discovered while investigating why adding BatchNorm to
an actor-only encoder configuration made training measurably *worse* than
a LayerNorm baseline (BatchNorm's running-stat mechanism is far less
forgiving of an upstream encoder being permanently frozen at random init
than LayerNorm's per-sample normalization is, which surfaced this as a
visible symptom: the encoder's BatchNorm bias stuck at an exact all-zeros
vector for an entire 2M-step run).

Fix: `_partition_encoder_params` splits a model's encoders into
critic-reachable (unchanged: trained via `encoder_optimizer` off
critic_loss) and actor-only (now trained via a new
`actor_only_encoder_optimizer`, stepped off actor_loss's backward pass --
the only signal it has).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from srl.algorithms.sac import SAC
from srl.core.config import SACConfig
from srl.registry.builder import ModelBuilder

OBS_DIM = 5
ACTION_DIM = 2


def _separate_encoders_model_dict(norm: str = "layer_norm") -> dict:
    """Actor and critic each have their OWN dedicated encoder -- no shared
    encoder, no implicit-single-encoder fallback. This is the exact shape
    that used to starve the actor's encoder of any training signal."""
    return {
        "encoders": [
            {
                "name": "actor_state_enc",
                "type": "mlp",
                "input_dim": OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "relu", "norm": norm}],
            },
            {
                "name": "critic_state_enc",
                "type": "mlp",
                "input_dim": OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "relu", "norm": norm}],
            },
        ],
        "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
        "actor": {
            "name": "actor",
            "type": "squashed_gaussian",
            "action_dim": ACTION_DIM,
            "log_std_min": -5.0,
            "log_std_max": 2.0,
            "layers": [{"out_features": 8, "activation": "relu", "norm": norm}],
        },
        "critic": {
            "name": "critic",
            "type": "twin_q",
            "action_dim": ACTION_DIM,
            "layers": [{"out_features": 8, "activation": "relu", "norm": norm}],
        },
    }


def _shared_encoder_model_dict(norm: str = "layer_norm") -> dict:
    """One implicit encoder feeding both heads (no explicit `flows:`) --
    the common case this fix must leave byte-for-byte unaffected."""
    return {
        "encoders": [
            {
                "name": "state_enc",
                "type": "mlp",
                "input_dim": OBS_DIM,
                "latent_dim": 8,
                "layers": [{"out_features": 8, "activation": "relu", "norm": norm}],
            },
        ],
        "actor": {
            "name": "actor",
            "type": "squashed_gaussian",
            "action_dim": ACTION_DIM,
            "log_std_min": -5.0,
            "log_std_max": 2.0,
            "layers": [{"out_features": 8, "activation": "relu", "norm": norm}],
        },
        "critic": {
            "name": "critic",
            "type": "twin_q",
            "action_dim": ACTION_DIM,
            "layers": [{"out_features": 8, "activation": "relu", "norm": norm}],
        },
    }


def _fill_replay_buffer(agent, n_transitions: int, rng: np.random.Generator) -> None:
    for _ in range(n_transitions):
        obs = {"policy": rng.standard_normal(OBS_DIM).astype(np.float32)}
        next_obs = {"policy": rng.standard_normal(OBS_DIM).astype(np.float32)}
        action = rng.uniform(-1.0, 1.0, ACTION_DIM).astype(np.float32)
        reward = np.array([float(rng.standard_normal())], dtype=np.float32)
        agent.buffer.add(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=np.array([False]),
            truncated=np.array([False]),
        )


def _build_sac(model_dict_fn, norm: str = "layer_norm") -> SAC:
    model = ModelBuilder.from_dict(model_dict_fn(norm))
    target = copy.deepcopy(model)
    cfg = SACConfig(action_dim=ACTION_DIM, batch_size=8)
    return SAC(model, target, config=cfg, device="cpu")


def test_actor_only_encoder_now_trains() -> None:
    rng = np.random.default_rng(0)
    agent = _build_sac(_separate_encoders_model_dict)
    _fill_replay_buffer(agent, n_transitions=32, rng=rng)

    before = {k: v.clone() for k, v in agent.model.encoders["actor_state_enc"].state_dict().items()}
    for _ in range(20):
        agent.update()
    after = agent.model.encoders["actor_state_enc"].state_dict()

    moved = {k: not torch.equal(before[k], after[k]) for k in before}
    assert all(moved.values()), f"actor_state_enc params that never moved: {moved}"


def test_critic_reachable_encoder_still_trains_via_critic_only() -> None:
    """The fix must not regress the existing (correct) behaviour for the
    encoder that DOES feed the critic: still trained via `encoder_optimizer`
    off critic_loss, still isolated from actor_loss's contribution."""
    rng = np.random.default_rng(0)
    agent = _build_sac(_separate_encoders_model_dict)
    _fill_replay_buffer(agent, n_transitions=32, rng=rng)

    assert agent.actor_only_encoder_optimizer is not None
    assert agent.encoder_optimizer is not None
    # `critic_state_enc` must be in the critic-reachable (original
    # encoder_optimizer) set, not the new actor-only one.
    actor_only_ids = {id(p) for p in agent._actor_only_encoder_param_list}
    critic_enc_ids = {id(p) for p in agent.model.encoders["critic_state_enc"].parameters()}
    assert critic_enc_ids.isdisjoint(actor_only_ids)
    assert critic_enc_ids <= {id(p) for p in agent._encoder_param_list}

    before = {
        k: v.clone() for k, v in agent.model.encoders["critic_state_enc"].state_dict().items()
    }
    for _ in range(20):
        agent.update()
    after = agent.model.encoders["critic_state_enc"].state_dict()
    assert all(not torch.equal(before[k], after[k]) for k in before)


def test_shared_single_encoder_config_unaffected() -> None:
    """A config with one implicit encoder feeding both heads (the common
    case) must end up with NO actor-only encoder at all -- the whole
    encoder is critic-reachable, exactly like before this fix existed."""
    rng = np.random.default_rng(0)
    agent = _build_sac(_shared_encoder_model_dict)

    assert agent.actor_only_encoder_optimizer is None
    assert agent._actor_only_encoder_param_list == []
    assert agent._actor_only_encoder_modules == []

    _fill_replay_buffer(agent, n_transitions=16, rng=rng)
    metrics = agent.update()
    assert metrics and all(np.isfinite(v) for v in metrics.values())


def test_actor_only_encoder_trains_with_weight_norm_projection_too() -> None:
    """End-to-end with weight_norm_projection=True: the actor-only
    encoder's Linear rows must be trained (moving in a meaningful,
    non-frozen-at-init direction over several updates), not just
    unit-norm-rescaled copies of their random init forever."""
    rng = np.random.default_rng(0)
    model = ModelBuilder.from_dict(_separate_encoders_model_dict("layer_norm"))
    target = copy.deepcopy(model)
    agent = SAC(
        model,
        target,
        config=SACConfig(action_dim=ACTION_DIM, batch_size=8, weight_norm_projection=True),
        device="cpu",
    )
    _fill_replay_buffer(agent, n_transitions=32, rng=rng)

    enc = agent.model.encoders["actor_state_enc"]
    linear = next(m for m in enc.modules() if isinstance(m, torch.nn.Linear))
    row_directions_before = torch.nn.functional.normalize(linear.weight.data.clone(), dim=1)

    for _ in range(20):
        agent.update()

    row_directions_after = torch.nn.functional.normalize(linear.weight.data, dim=1)
    # Rows are unit-norm either way (weight_norm_projection guarantees
    # that), so compare DIRECTION -- if the encoder were still frozen,
    # after == before exactly (projection of an already-unit vector is a
    # no-op). Real training moves the direction.
    assert not torch.allclose(row_directions_before, row_directions_after, atol=1e-4)
