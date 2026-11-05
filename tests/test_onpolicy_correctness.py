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


# ---------------------------------------------------------------------------
# Shared fakes for the training-loop tests
# ---------------------------------------------------------------------------


class _Box:
    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape


class _FakeSingleEnv:
    """Plain (non-auto-resetting) single env, GymnasiumWrapper-style API.

    Terminates after `episode_len` steps; stepping a finished episode
    without reset() raises, like a real dead env should.
    """

    num_envs = 1

    def __init__(self, episode_len: int = 3, low=-0.5, high=0.5):
        self.episode_len = episode_len
        self.act_space = _Box([low] * ACTION_DIM, [high] * ACTION_DIM)
        self.obs_space = None
        self.t = 0
        self.resets = 0
        self.dead = True
        self.actions: list[np.ndarray] = []

    def _obs(self):
        return {"state": np.full(OBS_DIM, float(self.t), dtype=np.float32)}

    def reset(self, **kwargs):
        self.t = 0
        self.resets += 1
        self.dead = False
        return self._obs(), {}

    def step(self, action):
        assert not self.dead, "stepped a finished episode without reset()"
        self.actions.append(np.asarray(action).copy())
        self.t += 1
        done = self.t >= self.episode_len
        self.dead = done
        return self._obs(), 1.0, done, False, {}

    def close(self):
        pass


class _FakeLogger:
    def __init__(self):
        self.episode_ends = 0

    def update_episodes(self, reward, done, trunc, step=None, info=None):
        self.episode_ends += int(np.any(np.asarray(done) | np.asarray(trunc)))

    def set_step(self, step):
        pass

    def record_metrics(self, *a, **k):
        pass


def _loop_args(steps: int):
    import argparse

    return argparse.Namespace(steps=steps, seed=0, eval_freq=0)


def _run_loop(agent, env, steps):
    from srl.cli import train as train_module

    logger = _FakeLogger()
    train_module._run_on_policy(agent, env, _loop_args(steps), [], logger, device="cpu")
    return logger


# ---------------------------------------------------------------------------
# Bug 3: deterministic actions were the raw (unbounded) Gaussian mean, and
# stochastic samples reached env.step unclipped
# ---------------------------------------------------------------------------


def _agents_for_bounds():
    from srl.algorithms.a3c import A3C
    from srl.core.config import A3CConfig

    torch.manual_seed(0)
    yield PPO(ModelBuilder.from_dict(_model_dict()), config=PPOConfig(), device="cpu")
    yield A2C(ModelBuilder.from_dict(_model_dict()), config=A2CConfig(), device="cpu")
    yield A3C(ModelBuilder.from_dict(_model_dict()), config=A3CConfig(), device="cpu")


@pytest.mark.parametrize("agent", list(_agents_for_bounds()), ids=["ppo", "a2c", "a3c"])
def test_deterministic_action_is_clipped_to_bounds(agent):
    # Push the Gaussian mean far outside the bounds.
    with torch.no_grad():
        agent.model.actor.mean_head.bias.fill_(50.0)
    agent.set_action_bounds(np.full(ACTION_DIM, -0.5), np.full(ACTION_DIM, 0.25))
    obs = {"state": torch.randn(4, OBS_DIM)}
    action, *_ = agent.predict(obs, deterministic=True)
    assert torch.all(action <= 0.25 + 1e-6) and torch.all(action >= -0.5 - 1e-6)
    # Stochastic predict returns the raw sample (needed for the log-prob).
    raw, *_ = agent.predict(obs, deterministic=False)
    assert torch.any(raw > 0.25)


def test_action_bounds_roundtrip_through_checkpoint():
    agent = PPO(ModelBuilder.from_dict(_model_dict()), config=PPOConfig(), device="cpu")
    agent.set_action_bounds(np.full(ACTION_DIM, -2.0), np.full(ACTION_DIM, 2.0))
    payload = agent.checkpoint_payload()
    fresh = PPO(ModelBuilder.from_dict(_model_dict()), config=PPOConfig(), device="cpu")
    fresh.load_checkpoint_payload(payload)
    low, high = fresh.action_bounds
    assert torch.allclose(high, torch.full((ACTION_DIM,), 2.0))
    assert torch.allclose(low, torch.full((ACTION_DIM,), -2.0))


def test_on_policy_loop_clips_env_actions_but_stores_raw_actions():
    torch.manual_seed(0)
    model = ModelBuilder.from_dict(_model_dict())
    with torch.no_grad():
        model.actor.mean_head.bias.fill_(50.0)
    agent = PPO(model, config=PPOConfig(n_steps=4, batch_size=4, n_epochs=1), device="cpu")
    stored = []
    orig_add = agent.buffer.add

    def _spy_add(**kw):
        stored.append(np.asarray(kw["action"]).copy())
        return orig_add(**kw)

    agent.buffer.add = _spy_add
    env = _FakeSingleEnv(episode_len=100, low=-0.5, high=0.5)
    _run_loop(agent, env, steps=4)
    sent = np.stack(env.actions)
    assert np.all(sent <= 0.5) and np.all(sent >= -0.5)
    assert np.any(np.stack(stored) > 0.5)  # the unclipped sample was stored


# ---------------------------------------------------------------------------
# Bug 2: n_envs == 1 on-policy loop never reset the (non-auto-resetting) env
# ---------------------------------------------------------------------------


def test_on_policy_loop_resets_single_env_after_episode_end():
    torch.manual_seed(0)
    agent = PPO(
        ModelBuilder.from_dict(_model_dict()),
        config=PPOConfig(n_steps=8, batch_size=8, n_epochs=1),
        device="cpu",
    )
    env = _FakeSingleEnv(episode_len=3)
    logger = _run_loop(agent, env, steps=16)
    assert len(env.actions) == 16
    # 1 initial reset + one per finished episode (16 // 3 = 5 episodes).
    assert env.resets == 1 + 16 // 3
    assert logger.episode_ends == 16 // 3


# ---------------------------------------------------------------------------
# Bug 4: PPO obs/reward normalisation was dead code
# ---------------------------------------------------------------------------


def test_ppo_normalization_defaults_on():
    cfg = PPOConfig()
    assert cfg.normalize_obs is True and cfg.normalize_reward is True


def test_ppo_reward_normalization_matches_vecnormalize():
    gamma = 0.9
    agent = PPO(
        ModelBuilder.from_dict(_model_dict()), config=PPOConfig(gamma=gamma), device="cpu"
    )
    rng = np.random.default_rng(0)
    rewards = rng.standard_normal((50, 2)) * 7.0 + 3.0
    ends = rng.random((50, 2)) < 0.1

    # Reference: SB3 VecNormalize reward path (RunningMeanStd count=1e-4).
    ret = np.zeros(2)
    mean, var, count = np.zeros(1), np.ones(1), 1e-4
    for r, e in zip(rewards, ends, strict=True):
        ret = ret * gamma + r
        b_mean, b_var, b_count = ret.mean(), ret.var(), 2
        delta = b_mean - mean
        tot = count + b_count
        mean = mean + delta * b_count / tot
        var = (var * count + b_var * b_count + delta**2 * count * b_count / tot) / tot
        count = tot
        expected = np.clip(r / np.sqrt(var + 1e-8), -10, 10)
        got = agent.normalize_reward(r, e)
        np.testing.assert_allclose(got, expected, rtol=1e-5)
        ret[e] = 0.0


def test_ppo_obs_normalization_in_loop_predict_and_checkpoint():
    torch.manual_seed(0)
    agent = PPO(
        ModelBuilder.from_dict(_model_dict()),
        config=PPOConfig(n_steps=8, batch_size=8, n_epochs=1),
        device="cpu",
    )
    stored = []
    orig_add = agent.buffer.add

    def _spy_add(**kw):
        stored.append({k: np.asarray(v).copy() for k, v in kw["obs"].items()})
        return orig_add(**kw)

    agent.buffer.add = _spy_add

    class _OffsetEnv(_FakeSingleEnv):
        def _obs(self):
            return {"state": np.full(OBS_DIM, 100.0 + 5.0 * self.t, dtype=np.float32)}

    _run_loop(agent, _OffsetEnv(episode_len=4), steps=16)
    assert agent._obs_normalizers, "no obs statistics were collected"
    norm = next(iter(agent._obs_normalizers.values()))
    assert np.all(norm.mean > 50.0)
    # The buffer holds normalised (not raw ~100-scale) observations.
    later = np.concatenate([np.concatenate(list(d.values())) for d in stored[4:]])
    assert np.all(np.abs(later) <= 10.0) and np.abs(later).max() < 50.0

    # predict() applies the same transform to raw obs ...
    raw = {"state": torch.full((1, OBS_DIM), 110.0)}
    a1, *_ = agent.predict(raw, deterministic=True)
    agent_no_norm_view = agent.normalize_obs(raw)
    assert not torch.allclose(next(iter(agent_no_norm_view.values())), raw["state"])

    # ... and the statistics survive a checkpoint round trip.
    fresh = PPO(
        ModelBuilder.from_dict(_model_dict()),
        config=PPOConfig(n_steps=8, batch_size=8, n_epochs=1),
        device="cpu",
    )
    fresh.load_checkpoint_payload(agent.checkpoint_payload())
    a2, *_ = fresh.predict(raw, deterministic=True)
    assert torch.allclose(a1, a2)
    assert fresh._ret_normalizer.count == pytest.approx(agent._ret_normalizer.count)


def test_ppo_normalization_can_be_disabled():
    agent = PPO(
        ModelBuilder.from_dict(_model_dict()),
        config=PPOConfig(normalize_obs=False, normalize_reward=False),
        device="cpu",
    )
    obs = {"state": np.full(OBS_DIM, 100.0, dtype=np.float32)}
    agent.update_obs_normalizer(obs)
    assert agent.normalize_obs(obs) is obs
    np.testing.assert_allclose(agent.normalize_reward([5.0], [False]), [5.0])


# ---------------------------------------------------------------------------
# Bug 8: A3C dropped `truncated` (time-limit envs never reset) and seeded
# workers from rank only
# ---------------------------------------------------------------------------


class _TimeLimitEnv(_FakeSingleEnv):
    """Never terminates; truncates after `episode_len` steps."""

    def step(self, action):
        obs, reward, _, _, info = super().step(action)
        truncated = self.t >= self.episode_len
        self.dead = truncated
        return obs, reward, False, truncated, info


class _StopAfter:
    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > self.n

    def set(self):
        self.n = 0


class _Counter:
    value = 0


def _run_a3c_worker(env, n_steps=7, iterations=2, seed=0, rank=0):
    import threading

    from srl.algorithms.a3c import _worker_fn
    from srl.core.config import A3CConfig

    torch.manual_seed(0)
    model = ModelBuilder.from_dict(_model_dict())
    cfg = A3CConfig(n_steps=n_steps, batch_size=n_steps, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _worker_fn(rank, model, opt, lambda: env, cfg, _Counter(), threading.Lock(), _StopAfter(iterations))
    return model


def test_a3c_worker_resets_on_truncation():
    env = _TimeLimitEnv(episode_len=3, low=-0.5, high=0.5)
    _run_a3c_worker(env, n_steps=7, iterations=2)
    assert len(env.actions) == 14
    # Initial reset + one per truncated episode.
    assert env.resets == 1 + 14 // 3
    sent = np.stack(env.actions)
    assert np.all(sent <= 0.5) and np.all(sent >= -0.5)


def test_a3c_worker_seed_derives_from_global_seed():
    def _first_actions(seed):
        env = _TimeLimitEnv(episode_len=100)
        _run_a3c_worker(env, n_steps=3, iterations=1, seed=seed, rank=1)
        return np.stack(env.actions)

    np.testing.assert_allclose(_first_actions(5), _first_actions(5))
    assert not np.allclose(_first_actions(5), _first_actions(6))
