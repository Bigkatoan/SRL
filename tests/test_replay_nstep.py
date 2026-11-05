"""Regression tests for n-step replay (CPU ReplayBuffer and GPUReplayBuffer).

Bugs fixed:

* ``ReplayBuffer`` collected n steps, wrote ONE transition and cleared the
  window -- (n-1)/n of all data was lost (300 steps at n=3 stored ~100).
* No buffer stored the per-transition discount, so algorithms bootstrapped an
  n-step return with ``gamma`` instead of ``gamma**k``.
* ``GPUReplayBuffer``'s n-step path funnelled every env into one shared
  window (env 0), mixing rewards across envs, and never cut windows at
  episode boundaries.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from srl.core.gpu_replay_buffer import GPUReplayBuffer
from srl.core.replay_buffer import ReplayBuffer

GAMMA = 0.9


def _obs(v: float) -> dict:
    return {"state": np.array([v], dtype=np.float32)}


def _rollout_cpu(buf: ReplayBuffer, rewards, dones, truncs, env_idx: int = 0, t0: int = 0):
    """Feed one env's trajectory; obs value = global step index."""
    for i, (r, d, tr) in enumerate(zip(rewards, dones, truncs, strict=True)):
        t = t0 + i
        buf.add(
            obs=_obs(t),
            action=np.zeros(1, dtype=np.float32),
            reward=np.array([r], dtype=np.float32),
            next_obs=_obs(t + 1),
            done=np.array([d]),
            truncated=np.array([tr]),
            env_idx=env_idx,
        )


def _stored(buf: ReplayBuffer) -> list[tuple]:
    n = len(buf)
    return [
        (
            float(buf._obs["state"][i, 0]),
            float(buf._rewards[i]),
            float(buf._next_obs["state"][i, 0]),
            float(buf._dones[i]),
            float(buf._truncated[i]),
            float(buf._discounts[i]),
        )
        for i in range(n)
    ]


def test_cpu_nstep_is_a_sliding_window_and_loses_no_data() -> None:
    buf = ReplayBuffer(capacity=1000, n_step=3, gamma=GAMMA)
    n_steps = 300
    rng = np.random.default_rng(0)
    rewards = rng.standard_normal(n_steps)
    dones = np.zeros(n_steps, dtype=bool)
    truncs = np.zeros(n_steps, dtype=bool)
    truncs[99::100] = True  # three 100-step episodes ending by time limit
    _rollout_cpu(buf, rewards, dones, truncs)
    # One stored transition per env step (old chunked version: ~102).
    assert len(buf) == n_steps


def test_cpu_nstep_returns_discounts_and_episode_flush() -> None:
    buf = ReplayBuffer(capacity=100, n_step=3, gamma=GAMMA)
    # Episode of 5 steps, terminated at the end.
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    _rollout_cpu(buf, rewards, [0, 0, 0, 0, 1], [0, 0, 0, 0, 0])
    rows = {r[0]: r for r in _stored(buf)}
    assert sorted(rows) == [0.0, 1.0, 2.0, 3.0, 4.0]
    g = GAMMA
    # Full windows: k=3, bootstrap from obs t+3 with gamma**3.
    assert rows[0.0][1] == pytest.approx(1 + g * 2 + g * g * 3)
    assert rows[0.0][2] == 3.0 and rows[0.0][3] == 0.0
    assert rows[0.0][5] == pytest.approx(g**3)
    assert rows[2.0][1] == pytest.approx(3 + g * 4 + g * g * 5)
    assert rows[2.0][3] == 1.0  # window ends on the terminal step
    # Flushed partial windows at the terminal step: k = 2, 1.
    assert rows[3.0][1] == pytest.approx(4 + g * 5)
    assert rows[3.0][5] == pytest.approx(g**2)
    assert rows[4.0][1] == pytest.approx(5.0)
    assert rows[4.0][5] == pytest.approx(g)
    assert rows[4.0][3] == 1.0 and rows[4.0][2] == 5.0


def test_cpu_nstep_truncation_bootstraps_not_terminal() -> None:
    buf = ReplayBuffer(capacity=100, n_step=3, gamma=GAMMA)
    _rollout_cpu(buf, [1.0, 1.0], [0, 0], [0, 1])
    rows = {r[0]: r for r in _stored(buf)}
    assert len(rows) == 2
    # Truncated: done stays 0 (bootstrap), truncated flag recorded, k=2 / k=1.
    assert rows[0.0][3] == 0.0 and rows[0.0][4] == 1.0
    assert rows[0.0][5] == pytest.approx(GAMMA**2)
    assert rows[0.0][2] == 2.0
    assert rows[1.0][5] == pytest.approx(GAMMA)


def test_cpu_nstep_windows_are_per_env() -> None:
    buf = ReplayBuffer(capacity=100, n_step=2, gamma=GAMMA)
    # Interleave two envs: env0 rewards 1, env1 rewards 100.
    for t in range(4):
        for e, r in ((0, 1.0), (1, 100.0)):
            buf.add(
                obs=_obs(1000 * e + t),
                action=np.zeros(1, dtype=np.float32),
                reward=np.array([r], dtype=np.float32),
                next_obs=_obs(1000 * e + t + 1),
                done=np.array([False]),
                truncated=np.array([False]),
                env_idx=e,
            )
    for obs0, rew, *_ in _stored(buf):
        expected = 1.0 + GAMMA if obs0 < 1000 else 100.0 * (1 + GAMMA)
        assert rew == pytest.approx(expected)


def test_one_step_buffer_reports_plain_gamma_discount() -> None:
    buf = ReplayBuffer(capacity=10, gamma=GAMMA)
    _rollout_cpu(buf, [1.0, 2.0], [0, 0], [0, 0])
    batch = buf.sample(4)
    assert batch.discounts is not None
    assert torch.allclose(batch.discounts, torch.full((4,), GAMMA))


def test_cpu_nstep_state_dict_round_trip_keeps_windows() -> None:
    buf = ReplayBuffer(capacity=100, n_step=3, gamma=GAMMA)
    _rollout_cpu(buf, [1.0, 2.0], [0, 0], [0, 0])
    assert len(buf) == 0
    buf2 = ReplayBuffer(capacity=100, n_step=3, gamma=GAMMA)
    buf2.load_state_dict(buf.state_dict())
    _rollout_cpu(buf2, [3.0], [0], [0], t0=2)
    assert len(buf2) == 1
    assert float(buf2._rewards[0]) == pytest.approx(1 + GAMMA * 2 + GAMMA**2 * 3)


# ---------------------------------------------------------------------------
# GPUReplayBuffer (exercised on CPU -- the logic is device-agnostic)
# ---------------------------------------------------------------------------


def _gpu_step(buf, t, rewards, dones, truncs, n_envs=2):
    obs = {"state": torch.tensor([[1000.0 * e + t] for e in range(n_envs)])}
    next_obs = {"state": torch.tensor([[1000.0 * e + t + 1] for e in range(n_envs)])}
    buf.add(
        obs=obs,
        action=torch.zeros(n_envs, 1),
        reward=torch.tensor(rewards, dtype=torch.float32),
        done=torch.tensor(dones, dtype=torch.float32),
        next_obs=next_obs,
        truncated=torch.tensor(truncs, dtype=torch.float32),
    )


def _gpu_rows(buf):
    n = len(buf)
    return [
        (
            float(buf._obs_buf["state"][i, 0]),
            float(buf._reward_buf[i]),
            float(buf._next_obs_buf["state"][i, 0]),
            float(buf._done_buf[i]),
            float(buf._discount_buf[i]),
        )
        for i in range(n)
    ]


def test_gpu_nstep_is_per_env_and_episode_aware() -> None:
    buf = GPUReplayBuffer(capacity=100, device="cpu", n_step=3, gamma=GAMMA, num_envs=2)
    # env0: reward 1 every step, terminates at t=3. env1: reward 100, never ends.
    for t in range(6):
        d0 = 1.0 if t == 3 else 0.0
        _gpu_step(buf, t, [1.0, 100.0], [d0, 0.0], [0.0, 0.0])
    rows = _gpu_rows(buf)
    g = GAMMA
    env0 = {r[0]: r for r in rows if r[0] < 1000}
    env1 = {r[0] - 1000: r for r in rows if r[0] >= 1000}
    # No cross-env mixing.
    for _, rew, *_ in env1.values():
        assert rew == pytest.approx(100.0 * (1 + g + g * g))
    assert sorted(env1) == [0.0, 1.0, 2.0, 3.0]  # full windows only, env still running
    # env0 episode 1 (t=0..3): full windows at t=0,1, flush at terminal t=3.
    assert env0[0.0][1] == pytest.approx(1 + g + g * g) and env0[0.0][3] == 0.0
    assert env0[1.0][1] == pytest.approx(1 + g + g * g) and env0[1.0][3] == 1.0
    assert env0[2.0][1] == pytest.approx(1 + g) and env0[2.0][4] == pytest.approx(g**2)
    assert env0[3.0][1] == pytest.approx(1.0) and env0[3.0][4] == pytest.approx(g)
    assert env0[3.0][2] == 4.0
    # Episode 2 (t=4,5): window not full yet, nothing leaked from episode 1.
    assert 4.0 not in env0 and 5.0 not in env0


def test_gpu_nstep_loses_no_data() -> None:
    buf = GPUReplayBuffer(capacity=10_000, device="cpu", n_step=3, gamma=GAMMA, num_envs=4)
    T = 100
    for t in range(T):
        trunc = [1.0 if (t + 1) % 25 == 0 else 0.0] * 4
        _gpu_step(buf, t, [1.0] * 4, [0.0] * 4, trunc, n_envs=4)
    assert len(buf) == 4 * T


def test_gpu_one_step_batched_add_and_discount() -> None:
    buf = GPUReplayBuffer(capacity=100, device="cpu", gamma=GAMMA, num_envs=2)
    _gpu_step(buf, 0, [1.0, 2.0], [0.0, 1.0], [0.0, 0.0])
    assert len(buf) == 2
    batch = buf.sample(8)
    assert torch.allclose(batch.discounts, torch.full((8,), GAMMA))
    assert set(batch.rewards.tolist()) <= {1.0, 2.0}


def test_gpu_single_transitions_with_env_idx_use_separate_windows() -> None:
    buf = GPUReplayBuffer(capacity=100, device="cpu", n_step=2, gamma=GAMMA, num_envs=2)
    for t in range(3):
        for e, r in ((0, 1.0), (1, 100.0)):
            buf.add(
                obs={"state": torch.tensor([1000.0 * e + t])},
                action=torch.zeros(1),
                reward=torch.tensor(r),
                done=torch.tensor(0.0),
                next_obs={"state": torch.tensor([1000.0 * e + t + 1])},
                env_idx=e,
            )
    for obs0, rew, *_ in _gpu_rows(buf):
        expected = 1.0 + GAMMA if obs0 < 1000 else 100.0 * (1 + GAMMA)
        assert rew == pytest.approx(expected)
