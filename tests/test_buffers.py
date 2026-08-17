"""Add/sample round-trip tests for every buffer class.

Beyond plain shape/dtype checks, these lock in the P0 fixes from the
release-stabilization pass: `truncated` tracked as a field genuinely separate
from `done` (ReplayBuffer, GPUReplayBuffer), HERReplayBuffer storing real
per-timestep `done` instead of fabricating it from the reward, and
PrioritizedReplayBuffer's `_write()` override accepting the `truncated`
positional arg the base class's `add()` now always passes.
"""

from __future__ import annotations

import numpy as np
import torch

OBS_DIM = 4
ACTION_DIM = 2


def test_replay_buffer_add_sample_round_trip_shapes_and_dtypes() -> None:
    from srl.core.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=16, obs_shape={"state": (OBS_DIM,)}, action_dim=ACTION_DIM)
    for i in range(10):
        buf.add(
            obs={"state": np.full(OBS_DIM, i, dtype=np.float32)},
            action=np.ones(ACTION_DIM, dtype=np.float32),
            reward=np.array([1.0], dtype=np.float32),
            next_obs={"state": np.full(OBS_DIM, i + 1, dtype=np.float32)},
            done=np.array([False]),
            truncated=np.array([False]),
        )
    assert len(buf) == 10

    batch = buf.sample(4)
    assert batch.observations["state"].shape == (4, OBS_DIM)
    assert batch.next_observations["state"].shape == (4, OBS_DIM)
    assert batch.actions.shape == (4, ACTION_DIM)
    assert batch.rewards.shape == (4,)
    assert batch.dones.shape == (4,)
    assert batch.truncated.shape == (4,)
    assert batch.dones.dtype == torch.float32
    assert batch.truncated.dtype == torch.float32


def test_replay_buffer_lazy_init_preserves_image_obs_channels() -> None:
    """Lazy init must not eat the channel dim of image observations.

    `add()`/`_write()` store exactly one transition per call, so the array
    handed in IS the per-transition shape. Stripping a leading axis allocated
    (96, 96) slots for a (3, 96, 96) frame, and every pixel-based off-policy
    run (SAC/DDPG/TD3 on CarRacing, Atari-likes, tiled cameras) died on the
    first add() with a broadcast error.
    """
    from srl.core.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=8, action_dim=ACTION_DIM)  # no obs_shape -> lazy init
    frame = np.zeros((3, 96, 96), dtype=np.float32)
    buf.add(
        obs={"state": frame},
        action=np.ones(ACTION_DIM, dtype=np.float32),
        reward=np.array([1.0], dtype=np.float32),
        next_obs={"state": frame},
        done=np.array([False]),
        truncated=np.array([False]),
    )

    assert buf.sample(2).observations["state"].shape == (2, 3, 96, 96)


def test_replay_buffer_keeps_truncated_and_done_independent() -> None:
    """The Phase-1 bootstrap-correctness fix: a timeout must not look like a
    real termination, and vice versa -- they are tracked in separate arrays."""
    from srl.core.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=4, obs_shape={"state": (OBS_DIM,)}, action_dim=ACTION_DIM)
    obs = {"state": np.zeros(OBS_DIM, dtype=np.float32)}
    # idx 0: truncated only (time-limit cutoff, NOT a real termination)
    buf.add(
        obs=obs,
        action=np.zeros(ACTION_DIM),
        reward=np.array([1.0]),
        next_obs=obs,
        done=np.array([False]),
        truncated=np.array([True]),
    )
    # idx 1: terminated only (real termination, no time limit involved)
    buf.add(
        obs=obs,
        action=np.zeros(ACTION_DIM),
        reward=np.array([1.0]),
        next_obs=obs,
        done=np.array([True]),
        truncated=np.array([False]),
    )

    assert buf._dones.tolist()[:2] == [0.0, 1.0]
    assert buf._truncated.tolist()[:2] == [1.0, 0.0]


def test_replay_buffer_checkpoint_round_trip_preserves_truncated() -> None:
    from srl.core.replay_buffer import ReplayBuffer

    buf = ReplayBuffer(capacity=8, obs_shape={"state": (OBS_DIM,)}, action_dim=ACTION_DIM)
    obs = {"state": np.zeros(OBS_DIM, dtype=np.float32)}
    buf.add(
        obs=obs,
        action=np.zeros(ACTION_DIM),
        reward=np.array([1.0]),
        next_obs=obs,
        done=np.array([False]),
        truncated=np.array([True]),
    )

    restored = ReplayBuffer(capacity=8, obs_shape={"state": (OBS_DIM,)}, action_dim=ACTION_DIM)
    restored.load_state_dict(buf.state_dict())
    assert restored._truncated.tolist() == buf._truncated.tolist()


def test_prioritized_replay_buffer_add_sample_round_trip() -> None:
    """Regression test for the `_write()` signature mismatch introduced by
    the truncated-tracking fix: PrioritizedReplayBuffer overrides `_write()`
    and must accept the `truncated` positional arg the base class now always
    passes, or every add() raises TypeError."""
    from srl.core.prioritized_replay_buffer import PrioritizedReplayBuffer

    buf = PrioritizedReplayBuffer(
        capacity=16, obs_shape={"state": (OBS_DIM,)}, action_dim=ACTION_DIM
    )
    obs = {"state": np.zeros(OBS_DIM, dtype=np.float32)}
    for i in range(10):
        buf.add(
            obs=obs,
            action=np.ones(ACTION_DIM, dtype=np.float32),
            reward=np.array([1.0]),
            next_obs=obs,
            done=np.array([i == 9]),
            truncated=np.array([False]),
        )

    batch = buf.sample(4)
    assert batch.weights is not None
    assert batch.weights.shape == (4,)
    assert batch.indices is not None and batch.indices.shape == (4,)

    buf.update_priorities(batch.indices, td_errors=np.ones(4, dtype=np.float32))


def test_gpu_replay_buffer_add_sample_round_trip_on_cpu() -> None:
    from srl.core.gpu_replay_buffer import GPUReplayBuffer

    buf = GPUReplayBuffer(capacity=16, device="cpu")
    for i in range(10):
        buf.add(
            obs={"state": torch.zeros(OBS_DIM)},
            action=torch.ones(ACTION_DIM),
            reward=1.0,
            done=False,
            next_obs={"state": torch.ones(OBS_DIM)},
            truncated=(i == 9),
        )
    assert len(buf) == 10

    batch = buf.sample(4)
    assert batch.observations["state"].shape == (4, OBS_DIM)
    assert batch.dones.shape == (4, 1)
    assert batch.truncated.shape == (4, 1)


def test_gpu_replay_buffer_add_batched_vectorized_input() -> None:
    from srl.core.gpu_replay_buffer import GPUReplayBuffer

    buf = GPUReplayBuffer(capacity=32, device="cpu")
    num_envs = 4
    buf.add(
        obs={"state": torch.zeros(num_envs, OBS_DIM)},
        action=torch.ones(num_envs, ACTION_DIM),
        reward=torch.ones(num_envs),
        done=torch.zeros(num_envs, dtype=torch.bool),
        next_obs={"state": torch.ones(num_envs, OBS_DIM)},
        truncated=torch.tensor([False, False, False, True]),
    )
    assert len(buf) == num_envs
    assert buf._truncated_buf[:num_envs, 0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_gpu_replay_buffer_batched_add_matches_per_row_add_and_handles_wraparound() -> None:
    """The batched `add()` path writes with a single vectorised scatter per
    tensor instead of looping `add()` once per env row (see gpu_replay_buffer.py).
    Lock in that it produces byte-identical buffer contents to the row-by-row
    semantics it replaces -- including when a batch straddles the circular
    buffer's wraparound point, where a naive contiguous-slice write would be
    wrong but per-index fancy indexing is not.
    """
    from srl.core.gpu_replay_buffer import GPUReplayBuffer

    def add_row_by_row(buf, obs, action, reward, done, next_obs, truncated) -> None:
        for i in range(action.shape[0]):
            buf.add(
                {k: v[i] for k, v in obs.items()},
                action[i],
                reward[i],
                done[i],
                {k: v[i] for k, v in next_obs.items()},
                truncated[i],
            )

    capacity = 16
    num_envs = 6
    # Distinct per-env values so a row mixup (wrong index math) would be caught.
    obs = {
        "state": torch.arange(num_envs * OBS_DIM, dtype=torch.float32).reshape(num_envs, OBS_DIM)
    }
    next_obs = {"state": obs["state"] + 100.0}
    action = (
        torch.arange(num_envs * ACTION_DIM, dtype=torch.float32).reshape(num_envs, ACTION_DIM)
        + 1000.0
    )
    reward = torch.arange(num_envs, dtype=torch.float32) + 2000.0
    done = torch.tensor([False, True, False, False, True, False])
    truncated = torch.tensor([False, False, True, False, False, True])

    # Pre-fill both buffers to the same near-full pointer position so the
    # batch add below straddles the wraparound (ptr=13, capacity=16, batch=6).
    prefill_n = 13
    prefill = dict(
        obs={"state": torch.zeros(prefill_n, OBS_DIM)},
        action=torch.zeros(prefill_n, ACTION_DIM),
        reward=torch.zeros(prefill_n),
        done=torch.zeros(prefill_n, dtype=torch.bool),
        next_obs={"state": torch.ones(prefill_n, OBS_DIM)},
        truncated=torch.zeros(prefill_n, dtype=torch.bool),
    )

    buf_vectorized = GPUReplayBuffer(capacity=capacity, device="cpu")
    buf_vectorized.add(**prefill)
    buf_vectorized.add(
        obs=obs, action=action, reward=reward, done=done, next_obs=next_obs, truncated=truncated
    )

    buf_reference = GPUReplayBuffer(capacity=capacity, device="cpu")
    add_row_by_row(buf_reference, **prefill)
    add_row_by_row(buf_reference, obs, action, reward, done, next_obs, truncated)

    assert buf_vectorized._ptr == buf_reference._ptr == (prefill_n + num_envs) % capacity
    assert buf_vectorized._size == buf_reference._size == capacity
    assert torch.equal(buf_vectorized._obs_buf["state"], buf_reference._obs_buf["state"])
    assert torch.equal(buf_vectorized._next_obs_buf["state"], buf_reference._next_obs_buf["state"])
    assert torch.equal(buf_vectorized._action_buf, buf_reference._action_buf)
    assert torch.equal(buf_vectorized._reward_buf, buf_reference._reward_buf)
    assert torch.equal(buf_vectorized._done_buf, buf_reference._done_buf)
    assert torch.equal(buf_vectorized._truncated_buf, buf_reference._truncated_buf)


def test_gpu_replay_buffer_checkpoint_round_trip() -> None:
    from srl.core.gpu_replay_buffer import GPUReplayBuffer

    buf = GPUReplayBuffer(capacity=8, device="cpu")
    buf.add(
        obs={"state": torch.zeros(OBS_DIM)},
        action=torch.ones(ACTION_DIM),
        reward=1.0,
        done=False,
        next_obs={"state": torch.ones(OBS_DIM)},
        truncated=True,
    )
    restored = GPUReplayBuffer(capacity=8, device="cpu")
    restored.load_state_dict(buf.state_dict())
    assert restored._truncated_buf[0, 0].item() == 1.0


def test_rollout_buffer_add_get_batches_round_trip() -> None:
    from srl.core.rollout_buffer import RolloutBuffer

    n_steps, n_envs = 6, 2
    buf = RolloutBuffer(n_steps=n_steps, n_envs=n_envs)
    for _ in range(n_steps):
        buf.add(
            obs={"state": np.zeros((n_envs, OBS_DIM), dtype=np.float32)},
            action=np.zeros((n_envs, ACTION_DIM), dtype=np.float32),
            reward=np.ones(n_envs, dtype=np.float32),
            done=np.zeros(n_envs, dtype=bool),
            value=np.zeros((n_envs, 1), dtype=np.float32),
            log_prob=np.zeros(n_envs, dtype=np.float32),
        )
    assert len(buf) == n_steps * n_envs
    assert buf.is_full()

    buf.compute_returns_and_advantages(last_value=np.zeros(n_envs, dtype=np.float32))
    batches = list(buf.get_batches(batch_size=4))
    total = sum(b.actions.shape[0] for b in batches)
    assert total == n_steps * n_envs
    for b in batches:
        assert b.obs["state"].shape[1] == OBS_DIM
        assert b.actions.shape[1] == ACTION_DIM


def test_her_replay_buffer_sample_uses_real_done_for_non_relabelled_fraction() -> None:
    """Regression test for the HER done-tracking fix: with her_ratio=0 (no
    relabelling at all), every sampled `done` must come from the real stored
    per-timestep termination, not the `reward == 0.0` heuristic -- an episode
    that truly ended but never achieved reward 0.0 must still show done=1."""
    from srl.core.her_replay_buffer import HERReplayBuffer

    def reward_fn(achieved, desired, info):
        return -1.0  # never "achieves" the goal -- reward is never 0.0

    buf = HERReplayBuffer(
        capacity=100,
        obs_dim=OBS_DIM,
        goal_dim=2,
        action_dim=ACTION_DIM,
        reward_fn=reward_fn,
        her_ratio=0.0,
        max_episode_len=10,
    )

    episode_len = 5
    for t in range(episode_len):
        buf.add_transition(
            obs=np.zeros(OBS_DIM, dtype=np.float32),
            achieved_goal=np.zeros(2, dtype=np.float32),
            desired_goal=np.ones(2, dtype=np.float32),
            action=np.zeros(ACTION_DIM, dtype=np.float32),
            next_obs=np.zeros(OBS_DIM, dtype=np.float32),
            next_achieved_goal=np.zeros(2, dtype=np.float32),
            done=(t == episode_len - 1),
        )
    assert len(buf) == 1

    # Only one episode is stored, so sample() draws t_idx uniformly from its 5
    # timesteps per sample; `done` is real only at the last one. Seeded (for
    # reproducibility) and oversized (for margin): with a 1-in-5 hit rate per
    # sample, a small batch_size here previously had a real ~1% chance of
    # drawing zero hits and failing on a fix that was actually correct --
    # batch_size=200 makes that probability effectively zero (~1e-19) even
    # before the seed pins it exactly.
    np.random.seed(0)
    batch = buf.sample(batch_size=200)
    assert batch.dones.sum().item() > 0, (
        "expected some sampled transitions to carry the real terminal done=1; "
        "got all zeros -- done tracking regressed to the reward==0.0 heuristic"
    )
