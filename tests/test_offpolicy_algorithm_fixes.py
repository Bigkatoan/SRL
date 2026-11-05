"""Regression tests for SAC/TD3/DDPG correctness fixes.

* n-step targets bootstrap with the per-transition ``gamma**k`` discount.
* TD3/DDPG actor-only encoders (separate ``actor_state_enc``) actually train.
* SAC ``predict(deterministic=True)`` returns tanh(mean), not the raw mean.
* TD3/DDPG ``update()`` runs in train mode even after ``predict()``.
* DDPG exploration noise is per-env and resettable.
* Optional observation normalizer (saved in checkpoints).
* Prioritized replay is wired (IS weights + priority updates).
"""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from srl.algorithms.ddpg import DDPG, OrnsteinUhlenbeckNoise
from srl.algorithms.sac import SAC
from srl.algorithms.td3 import TD3
from srl.core.config import DDPGConfig, SACConfig, TD3Config
from srl.core.prioritized_replay_buffer import PrioritizedReplayBuffer
from srl.core.replay_buffer import ReplayBatch
from srl.registry.builder import ModelBuilder

OBS_DIM = 4
ACT_DIM = 2


def _layers(dropout: float = 0.0):
    layer = {"out_features": 16, "activation": "relu"}
    if dropout:
        layer["dropout"] = dropout
    return [layer]


def _model_dict(actor_type: str, critic_type: str, *, separate: bool, dropout: float = 0.0):
    actor = {"name": "actor", "type": actor_type, "action_dim": ACT_DIM, "layers": _layers(dropout)}
    if actor_type == "squashed_gaussian":
        actor.update(log_std_min=-5.0, log_std_max=2.0)
    critic = {
        "name": "critic",
        "type": critic_type,
        "action_dim": ACT_DIM,
        "layers": _layers(dropout),
    }
    enc = {"type": "mlp", "input_dim": OBS_DIM, "latent_dim": 16, "layers": _layers()}
    if separate:
        return {
            "encoders": [
                {"name": "actor_state_enc", **enc},
                {"name": "critic_state_enc", **enc},
            ],
            "flows": ["actor_state_enc -> actor", "critic_state_enc -> critic"],
            "actor": actor,
            "critic": critic,
        }
    return {"encoders": [{"name": "state_enc", **enc}], "actor": actor, "critic": critic}


def _build(algo: str, *, separate: bool = True, dropout: float = 0.0, **cfg_kw):
    if algo == "sac":
        md = _model_dict("squashed_gaussian", "twin_q", separate=separate, dropout=dropout)
        cls, cfg = SAC, SACConfig(action_dim=ACT_DIM, batch_size=16, **cfg_kw)
    elif algo == "td3":
        md = _model_dict("deterministic", "twin_q", separate=separate, dropout=dropout)
        cls, cfg = TD3, TD3Config(action_dim=ACT_DIM, batch_size=16, policy_delay=1, **cfg_kw)
    else:
        md = _model_dict("deterministic", "q_function", separate=separate, dropout=dropout)
        cls, cfg = DDPG, DDPGConfig(action_dim=ACT_DIM, batch_size=16, **cfg_kw)
    model = ModelBuilder.from_dict(md)
    return cls(model, copy.deepcopy(model), config=cfg, device="cpu")


def _fill(agent, n: int = 64, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        agent.buffer.add(
            obs={"state": rng.standard_normal(OBS_DIM).astype(np.float32)},
            action=rng.uniform(-1, 1, ACT_DIM).astype(np.float32),
            reward=np.array([rng.standard_normal()], dtype=np.float32),
            next_obs={"state": rng.standard_normal(OBS_DIM).astype(np.float32)},
            done=np.array([False]),
            truncated=np.array([False]),
        )


def _obs_t(batch: int = 3, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"state": torch.randn(batch, OBS_DIM, generator=g)}


# ---------------------------------------------------------------------------
# n-step discount in targets
# ---------------------------------------------------------------------------

_TARGET_KEY = {"sac": "sac/target_q_mean", "td3": "td3/target_q_mean", "ddpg": "ddpg/target_q_mean"}


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg"])
def test_target_uses_per_transition_discount(algo: str) -> None:
    """With discounts=0 the target must be exactly the stored reward; with
    the old `gamma` bootstrap it would also include gamma * Q(s', a')."""
    agent = _build(algo)
    B = agent.cfg.batch_size
    rewards = torch.linspace(-1.0, 1.0, B)
    batch = ReplayBatch(
        observations=_obs_t(B, 2),
        actions=torch.zeros(B, ACT_DIM),
        rewards=rewards,
        next_observations=_obs_t(B, 3),
        dones=torch.zeros(B),
        discounts=torch.zeros(B),
    )
    _fill(agent, B)
    agent.buffer.sample = lambda _bs: batch
    metrics = agent.update()
    assert metrics[_TARGET_KEY[algo]] == pytest.approx(rewards.mean().item(), abs=1e-6)


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg"])
def test_nstep_config_builds_buffer_with_gamma_k_discounts(algo: str) -> None:
    agent = _build(algo, replay_n_step=3, gamma=0.9)
    _fill(agent, 64)
    batch = agent.buffer.sample(32)
    assert torch.allclose(batch.discounts, torch.full((32,), 0.9**3))


# ---------------------------------------------------------------------------
# Actor-only encoders train (TD3 / DDPG)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["td3", "ddpg", "sac"])
def test_actor_only_encoder_receives_updates(algo: str) -> None:
    torch.manual_seed(0)
    agent = _build(algo, separate=True)
    _fill(agent, 64)
    actor_enc = agent.model.encoders["actor_state_enc"]
    critic_enc = agent.model.encoders["critic_state_enc"]
    a0 = [p.detach().clone() for p in actor_enc.parameters()]
    c0 = [p.detach().clone() for p in critic_enc.parameters()]
    for _ in range(5):
        agent.update()
    assert any(not torch.equal(p0, p) for p0, p in zip(a0, actor_enc.parameters(), strict=True))
    assert any(not torch.equal(p0, p) for p0, p in zip(c0, critic_enc.parameters(), strict=True))


@pytest.mark.parametrize("algo", ["td3", "ddpg"])
def test_shared_encoder_is_not_trained_by_actor_loss(algo: str) -> None:
    """Shared-encoder behaviour stays DrQ-style: only the critic trains it."""
    agent = _build(algo, separate=False)
    assert agent.actor_only_encoder_optimizer is None
    assert agent.encoder_optimizer is not None


# ---------------------------------------------------------------------------
# SAC deterministic action
# ---------------------------------------------------------------------------


def test_sac_deterministic_action_is_tanh_of_mean() -> None:
    torch.manual_seed(0)
    agent = _build("sac", separate=False)
    # Scale up the mean head so the pre-tanh mean is far outside [-1, 1].
    with torch.no_grad():
        agent.model.actor.mean_head.weight.mul_(50.0)
        agent.model.actor.mean_head.bias.fill_(3.0)
    obs = _obs_t(8)
    action, *_ = agent.predict(obs, deterministic=True)
    assert action.abs().max() <= 1.0
    with torch.no_grad():
        out = agent.model(obs, compute_critic=False)["actor_out"]
    assert torch.allclose(action, torch.tanh(out["mean"]), atol=1e-6)


# ---------------------------------------------------------------------------
# Train/eval mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["td3", "ddpg", "sac"])
def test_update_runs_in_train_mode_after_predict(algo: str) -> None:
    agent = _build(algo, dropout=0.1)
    _fill(agent, 64)
    modes: list[bool] = []
    agent.model.critic.register_forward_pre_hook(lambda m, _inp: modes.append(m.training))
    agent.predict(_obs_t(2))  # used to leave the model in eval() for good
    modes.clear()
    agent.update()
    assert modes and all(modes)
    # And predict() itself still runs in eval mode.
    actor_modes: list[bool] = []
    agent.model.actor.register_forward_pre_hook(lambda m, _inp: actor_modes.append(m.training))
    agent.predict(_obs_t(2))
    assert actor_modes == [False]


def test_target_network_stays_in_eval_mode() -> None:
    for algo in ("td3", "ddpg"):
        agent = _build(algo, dropout=0.1)
        assert not agent.target_model.training


# ---------------------------------------------------------------------------
# DDPG per-env noise
# ---------------------------------------------------------------------------


def test_ddpg_noise_is_independent_per_env() -> None:
    torch.manual_seed(0)
    agent = _build("ddpg", action_noise="ou", noise_sigma=0.3)
    obs = {"state": torch.zeros(4, OBS_DIM)}  # identical obs for every env
    noisy, *_ = agent.predict(obs)
    assert noisy.shape == (4, ACT_DIM)
    # Same observation, independent noise per row.
    assert not torch.allclose(noisy[0], noisy[1])


def test_ou_noise_reset_per_env() -> None:
    noise = OrnsteinUhlenbeckNoise(ACT_DIM, sigma=0.5)
    for _ in range(10):
        noise.sample(3)
    before = noise.state.clone()
    noise.reset(np.array([False, True, False]))
    assert torch.equal(noise.state[1], torch.zeros(ACT_DIM))
    assert torch.equal(noise.state[0], before[0]) and torch.equal(noise.state[2], before[2])
    noise.reset()
    assert torch.equal(noise.state, torch.zeros(3, ACT_DIM))
    # Single-env API keeps returning a 1-D sample.
    assert OrnsteinUhlenbeckNoise(ACT_DIM).sample().shape == (ACT_DIM,)


def test_ddpg_reset_noise_hook() -> None:
    agent = _build("ddpg", action_noise="ou", noise_sigma=0.5)
    agent.predict({"state": torch.zeros(2, OBS_DIM)})
    agent.reset_noise([0])
    assert torch.equal(agent.noise.state[0], torch.zeros(ACT_DIM))


# ---------------------------------------------------------------------------
# Observation normalizer
# ---------------------------------------------------------------------------


def test_obs_normalizer_stats_and_clip() -> None:
    from srl.algorithms._common import ObsNormalizer

    norm = ObsNormalizer(clip=5.0)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(10.0, 2.0, size=(5000, 3)), dtype=torch.float32)
    for chunk in x.split(500):
        norm.update({"k": chunk})
    out = norm.normalize({"k": x, "img": torch.zeros(2, 3, 4, 4)})["k"]
    assert out.mean(0).abs().max() < 0.05
    assert (out.std(0) - 1).abs().max() < 0.05
    big = norm.normalize({"k": torch.full((1, 3), 1e6)})["k"]
    assert torch.all(big == 5.0)


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg"])
def test_normalizer_is_used_and_checkpointed(algo: str, tmp_path) -> None:
    agent = _build(algo, normalize_obs=True)
    assert agent.obs_normalizer is not None
    rng = np.random.default_rng(0)
    for _ in range(64):
        agent.buffer.add(
            obs={"state": (100 + rng.standard_normal(OBS_DIM)).astype(np.float32)},
            action=rng.uniform(-1, 1, ACT_DIM).astype(np.float32),
            reward=np.array([0.0], dtype=np.float32),
            next_obs={"state": (100 + rng.standard_normal(OBS_DIM)).astype(np.float32)},
            done=np.array([False]),
            truncated=np.array([False]),
        )
    agent.update()
    state = agent.obs_normalizer.state_dict()
    assert state["mean"] and all(v.mean() > 90 for v in state["mean"].values())
    path = str(tmp_path / "ckpt.pt")
    agent.save(path)
    fresh = _build(algo, normalize_obs=True)
    fresh.load(path)
    fresh_state = fresh.obs_normalizer.state_dict()
    for k, v in state["mean"].items():
        assert torch.allclose(fresh_state["mean"][k], v)
    # Deterministic actions agree after a reload (same weights + same stats).
    obs = {"state": torch.full((2, OBS_DIM), 100.0)}
    a1, *_ = agent.predict(obs, deterministic=True)
    a2, *_ = fresh.predict(obs, deterministic=True)
    assert torch.allclose(a1, a2, atol=1e-6)


def test_normalizer_is_off_by_default() -> None:
    for algo in ("sac", "td3", "ddpg"):
        assert _build(algo).obs_normalizer is None


# ---------------------------------------------------------------------------
# Prioritized replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["sac", "td3", "ddpg"])
def test_use_per_builds_prioritized_buffer_and_updates_priorities(algo: str) -> None:
    agent = _build(algo, use_per=True)
    assert isinstance(agent.buffer, PrioritizedReplayBuffer)
    _fill(agent, 64)
    leaves = agent.buffer._sum_tree._tree[agent.buffer._sum_tree._cap :][:64].copy()
    assert np.allclose(leaves, leaves[0])  # all new transitions at max priority
    agent.update()
    after = agent.buffer._sum_tree._tree[agent.buffer._sum_tree._cap :][:64]
    assert not np.allclose(after, leaves)


def test_per_sampling_is_proportional_for_non_power_of_two_capacity() -> None:
    buf = PrioritizedReplayBuffer(capacity=5, alpha=1.0, gamma=0.9)
    for i in range(5):
        buf.add(
            obs={"state": np.array([float(i)], dtype=np.float32)},
            action=np.zeros(1, dtype=np.float32),
            reward=np.array([0.0], dtype=np.float32),
            next_obs={"state": np.array([0.0], dtype=np.float32)},
            done=np.array([False]),
        )
    buf.update_priorities(np.arange(5), np.array([1.0, 1.0, 1.0, 1.0, 6.0]) - 1e-6)
    counts = np.zeros(5)
    np.random.seed(0)
    for _ in range(400):
        b = buf.sample(10)
        np.add.at(counts, b.indices, 1)
    freq = counts / counts.sum()
    assert freq[4] == pytest.approx(0.6, abs=0.05)
    assert freq[:4] == pytest.approx([0.1] * 4, abs=0.03)
    b = buf.sample(10)
    # IS weights: max-normalised, lowest for the most-sampled slot.
    assert b.weights.max() == pytest.approx(1.0)
    assert b.weights[b.indices == 4].max() < 1.0 if (b.indices == 4).any() else True


def test_per_with_her_is_rejected() -> None:
    cfg = SACConfig(action_dim=ACT_DIM, use_her=True, use_per=True)
    model = ModelBuilder.from_dict(_model_dict("squashed_gaussian", "twin_q", separate=False))
    with pytest.raises(ValueError, match="use_per"):
        SAC(model, copy.deepcopy(model), config=replace(cfg), device="cpu")
