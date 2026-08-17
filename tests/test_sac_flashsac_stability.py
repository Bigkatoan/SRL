"""Tests for SAC's two FlashSAC-inspired stability mechanisms
(arXiv:2604.04539 section 4.2), added to make the large-batch/few-
gradient-step regime (`batch_size` way up, `gradient_steps` way down --
see `SACConfig`'s own docstring on those fields) safe to actually use:

* Weight normalization: `SACConfig(weight_norm_projection=True)` projects
  every Linear weight row to unit L2 norm, and every norm layer's affine
  vector to L2 norm sqrt(d), right after each optimizer.step().
* BatchNorm support (`norm: batch_norm` in a config's encoder/actor/critic
  layers): target_model must stay in eval() mode permanently (running
  statistics, never this-call batch statistics) and `self.model`'s mode
  must be set atomically with its forward call so a concurrent
  `predict()` (as `AsyncOffPolicyRunner`'s collector thread performs) can
  never race `update()`'s mode-setting on the shared model instance.

Both are off-by-default / only activate when a config actually asks for
them (`weight_norm_projection: true`, or a BatchNorm norm layer
respectively) -- `test_algorithms.py::test_sac_one_gradient_step_...`
already covers that the untouched default path is unaffected.
"""

from __future__ import annotations

import copy
import threading

import numpy as np
import torch
import torch.nn as nn

from srl.algorithms.sac import SAC, _project_weights_unit_norm, _sync_target_buffers
from srl.core.config import SACConfig
from srl.registry.builder import ModelBuilder

OBS_DIM = 5
ACTION_DIM = 2


def _offpolicy_model_dict(norm: str = "none") -> dict:
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


def _build_sac(norm: str, **cfg_kwargs) -> SAC:
    model = ModelBuilder.from_dict(_offpolicy_model_dict(norm))
    target = copy.deepcopy(model)
    cfg = SACConfig(action_dim=ACTION_DIM, batch_size=8, **cfg_kwargs)
    return SAC(model, target, config=cfg, device="cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Weight normalization
# ──────────────────────────────────────────────────────────────────────────────


def test_weight_norm_projection_disabled_by_default() -> None:
    """SACConfig() with no override must be `weight_norm_projection=False`
    -- opt-in, not a behaviour change for every existing SAC config."""
    assert SACConfig().weight_norm_projection is False


def test_project_weights_unit_norm_normalizes_linear_rows_and_affine_vectors() -> None:
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(6, 10),
        nn.LayerNorm(10),
        nn.ReLU(),
        nn.Linear(10, 3),
    )
    # Deliberately blow the weights up (and, since LayerNorm's bias starts
    # at exactly zero -- `mul_` alone would leave it there, a degenerate
    # all-zero vector no scalar projection can rescale to a nonzero norm --
    # move every parameter well off its default init) so the projection has
    # visible work to do on every parameter tensor, bias included.
    with torch.no_grad():
        for p in net.parameters():
            p.add_(5.0).mul_(37.0)

    _project_weights_unit_norm(net)

    linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 2
    for lin in linear_layers:
        row_norms = lin.weight.data.norm(p=2, dim=1)
        assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-5)

    ln = next(m for m in net.modules() if isinstance(m, nn.LayerNorm))
    d = ln.weight.numel()
    assert torch.isclose(ln.weight.data.norm(p=2), torch.tensor(float(d) ** 0.5), atol=1e-4)
    assert torch.isclose(ln.bias.data.norm(p=2), torch.tensor(float(d) ** 0.5), atol=1e-4)


def test_sac_weight_norm_projection_end_to_end_after_update() -> None:
    """After a real update() with weight_norm_projection=True, every Linear
    weight row in the actor/critic/encoders is unit-norm and every norm
    layer's affine vector is norm sqrt(d) -- exercising the actual call
    sites inside SAC.update() (after critic/encoder/actor optimizer.step()),
    not just the helper in isolation."""
    rng = np.random.default_rng(0)
    agent = _build_sac("layer_norm", weight_norm_projection=True)
    _fill_replay_buffer(agent, n_transitions=16, rng=rng)

    metrics = agent.update()
    assert metrics and all(np.isfinite(v) for v in metrics.values())

    for submodule in (agent.model.actor, agent.model.critic, agent.model.encoders):
        for m in submodule.modules():
            if isinstance(m, nn.Linear):
                row_norms = m.weight.data.norm(p=2, dim=1)
                assert torch.allclose(row_norms, torch.ones_like(row_norms), atol=1e-4)
            elif isinstance(m, nn.LayerNorm):
                d = m.weight.numel()
                assert torch.isclose(
                    m.weight.data.norm(p=2), torch.tensor(float(d) ** 0.5), atol=1e-3
                )


# ──────────────────────────────────────────────────────────────────────────────
# BatchNorm: target-network eval mode + running-stat propagation
# ──────────────────────────────────────────────────────────────────────────────


def test_sac_target_model_is_always_eval_mode() -> None:
    """target_model must never end up in train() mode -- it's constructed
    in eval() and nothing in SAC should ever flip it, regardless of how
    many predict()/update() calls (which toggle `self.model`'s mode) run
    in between."""
    rng = np.random.default_rng(0)
    agent = _build_sac("batch_norm")
    assert agent.target_model.training is False

    _fill_replay_buffer(agent, n_transitions=16, rng=rng)
    for _ in range(3):
        obs = {"policy": torch.randn(1, OBS_DIM)}
        agent.predict(obs)
        agent.update()
        assert agent.target_model.training is False


def test_sac_batchnorm_detected_only_when_present() -> None:
    bn_agent = _build_sac("batch_norm")
    assert bn_agent._has_batchnorm is True

    plain_agent = _build_sac("layer_norm")
    assert plain_agent._has_batchnorm is False

    none_agent = _build_sac("none")
    assert none_agent._has_batchnorm is False


def test_sac_batchnorm_update_is_finite_and_uses_cross_batch_path() -> None:
    """With a BatchNorm critic, a real update() call must still produce
    finite metrics and move parameters -- exercising `_critic_forward`'s
    "cross-batch value prediction" path (concatenated (s,a)/(s',a') batch,
    split back apart) rather than the plain two-separate-calls path."""
    rng = np.random.default_rng(0)
    agent = _build_sac("batch_norm")
    _fill_replay_buffer(agent, n_transitions=16, rng=rng)

    before = float(sum(p.detach().abs().sum().item() for p in agent.model.parameters()))
    metrics = agent.update()
    after = float(sum(p.detach().abs().sum().item() for p in agent.model.parameters()))

    assert metrics and all(np.isfinite(v) for v in metrics.values())
    assert before != after


def test_sync_target_buffers_hard_copies_batchnorm_running_stats() -> None:
    torch.manual_seed(0)
    src = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    tgt = copy.deepcopy(src)

    # Diverge src's running stats from tgt's via a few train-mode forward
    # passes (BatchNorm updates running_mean/var as a side effect).
    src.train()
    for _ in range(5):
        src(torch.randn(6, 4) * 3.0 + 1.0)

    src_bn = src[1]
    tgt_bn = tgt[1]
    assert not torch.allclose(src_bn.running_mean, tgt_bn.running_mean)

    _sync_target_buffers(src, tgt)

    assert torch.equal(src_bn.running_mean, tgt_bn.running_mean)
    assert torch.equal(src_bn.running_var, tgt_bn.running_var)
    assert torch.equal(src_bn.num_batches_tracked, tgt_bn.num_batches_tracked)


def test_sac_batchnorm_target_running_stats_track_online_after_updates() -> None:
    """End-to-end: after several update() calls, target_model's BatchNorm
    running stats must match self.model's exactly (hard-copied every
    update() via `_sync_target_buffers`), not be stuck at their step-0
    initial values -- which is what would happen if target_model (always
    eval()) never had its buffers propagated at all."""
    rng = np.random.default_rng(0)
    agent = _build_sac("batch_norm")
    _fill_replay_buffer(agent, n_transitions=32, rng=rng)

    for _ in range(4):
        agent.update()

    online_bns = [m for m in agent.model.modules() if isinstance(m, nn.BatchNorm1d)]
    target_bns = [m for m in agent.target_model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert online_bns, "fixture's `norm: batch_norm` layers didn't produce any BatchNorm1d"
    assert len(online_bns) == len(target_bns)
    for ob, tb in zip(online_bns, target_bns, strict=True):
        assert torch.equal(ob.running_mean, tb.running_mean)
        assert torch.equal(ob.running_var, tb.running_var)
        # Online BN running stats must actually have moved off their
        # freshly-initialized values (mean=0, var=1) -- otherwise the
        # equality checks above would trivially pass without the sync
        # logic being exercised at all.
        assert not torch.allclose(ob.running_mean, torch.zeros_like(ob.running_mean))


# ──────────────────────────────────────────────────────────────────────────────
# predict()/update() mode-race guard
# ──────────────────────────────────────────────────────────────────────────────


def test_sac_predict_and_update_interleave_safely_under_concurrency() -> None:
    """Smoke test for the `_model_lock` guard `_forward_model` takes:
    hammer predict() from one thread while update() runs on another (the
    same pattern `AsyncOffPolicyRunner` uses -- collector thread calling
    predict(), background trainer thread calling update()), with a
    BatchNorm model so mode-setting is actually load-bearing. Must not
    crash, deadlock, or produce non-finite parameters/actions."""
    rng = np.random.default_rng(0)
    agent = _build_sac("batch_norm")
    _fill_replay_buffer(agent, n_transitions=64, rng=rng)

    stop = threading.Event()
    errors: list[BaseException] = []

    def _predict_loop() -> None:
        try:
            for _ in range(200):
                if stop.is_set():
                    return
                obs = {"policy": torch.randn(4, OBS_DIM)}
                action, _, _, _ = agent.predict(obs)
                assert torch.isfinite(action).all()
        except BaseException as exc:  # noqa: BLE001 -- surfaced via `errors`
            errors.append(exc)

    def _update_loop() -> None:
        try:
            for _ in range(30):
                if stop.is_set():
                    return
                agent.update()
        except BaseException as exc:  # noqa: BLE001 -- surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=_predict_loop), threading.Thread(target=_update_loop)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    stop.set()
    for t in threads:
        t.join(timeout=5)

    assert not errors, f"concurrent predict()/update() raised: {errors}"
    for p in agent.model.parameters():
        assert torch.isfinite(p).all()
