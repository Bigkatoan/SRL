"""Regression tests for HERReplayBuffer correctness.

* Relabelled transitions used to get ``done = (reward == 0)``. Fetch-style goal
  tasks never terminate (episodes end by time limit), so this cut the
  bootstrap exactly on the successful relabelled transitions. ``done`` must
  be the real termination flag for every sampled transition.
* "future" relabelling sampled achieved goals from t..T; ag[t] belongs to the
  state the transition *started* in, so the range must be t+1..T.
"""

from __future__ import annotations

import numpy as np
import pytest

from srl.core.her_replay_buffer import HERReplayBuffer

OBS_DIM, GOAL_DIM, ACT_DIM = 4, 2, 2
EP_LEN = 10


def _vec_reward(achieved, desired, info):
    d = np.linalg.norm(np.asarray(achieved) - np.asarray(desired), axis=-1)
    return -(d > 0.05).astype(np.float32)


def _scalar_reward(achieved, desired, info):
    return 0.0 if float(np.linalg.norm(np.asarray(achieved) - np.asarray(desired))) < 0.05 else -1.0


def _filled(strategy="future", reward_fn=_vec_reward, n_eps=20, terminal_last=False):
    buf = HERReplayBuffer(
        capacity=n_eps * EP_LEN,
        obs_dim=OBS_DIM,
        goal_dim=GOAL_DIM,
        action_dim=ACT_DIM,
        reward_fn=reward_fn,
        strategy=strategy,
        her_ratio=1.0,
        max_episode_len=EP_LEN,
    )
    for ep in range(n_eps):
        for t in range(EP_LEN):
            # obs encodes t; achieved goal at obs t is (t, t).
            last = t == EP_LEN - 1
            buf.add_transition(
                obs=np.full(OBS_DIM, t, dtype=np.float32),
                achieved_goal=np.full(GOAL_DIM, t, dtype=np.float32),
                desired_goal=np.full(GOAL_DIM, 1000.0 + ep, dtype=np.float32),
                action=np.zeros(ACT_DIM, dtype=np.float32),
                next_obs=np.full(OBS_DIM, t + 1, dtype=np.float32),
                next_achieved_goal=np.full(GOAL_DIM, t + 1, dtype=np.float32),
                done=bool(terminal_last and last),
                truncated=bool(last and not terminal_last),
            )
    return buf


@pytest.mark.parametrize("reward_fn", [_vec_reward, _scalar_reward])
def test_relabelled_success_is_not_terminal_for_infinite_horizon_tasks(reward_fn) -> None:
    np.random.seed(0)
    buf = _filled(strategy="final", reward_fn=reward_fn)
    batch = buf.sample(256)
    # Plenty of relabelled successes (reward 0) ...
    assert (batch.rewards == 0).sum() > 10
    # ... but none of them is a real termination.
    assert float(batch.dones.sum()) == 0.0


def test_real_terminations_are_kept() -> None:
    np.random.seed(0)
    buf = _filled(strategy="future", terminal_last=True)
    batch = buf.sample(512)
    t = batch.obs["state"][:, 0].numpy()
    dones = batch.dones.numpy()
    assert np.all(dones[t == EP_LEN - 1] == 1.0)
    assert np.all(dones[t != EP_LEN - 1] == 0.0)


def test_future_strategy_samples_strictly_later_achieved_goals() -> None:
    np.random.seed(0)
    buf = _filled(strategy="future")
    batch = buf.sample(2048)
    full = batch.obs["state"].numpy()
    t = full[:, 0]
    goal = full[:, OBS_DIM]  # relabelled desired goal = ag[future_t] = future_t
    assert np.all(goal >= t + 1)
    assert np.all(goal <= EP_LEN)
    # The goal achieved by the transition itself (ag[t+1]) is reachable ...
    assert np.any(goal == t + 1)
    # ... so the very last transition always sees reward 0 under "future".
    last = t == EP_LEN - 1
    assert np.all(batch.rewards.numpy()[last] == 0.0)


def test_vectorised_and_scalar_reward_fns_agree() -> None:
    for fn in (_vec_reward, _scalar_reward):
        np.random.seed(1)
        buf = _filled(strategy="future", reward_fn=fn)
        r = buf.sample(128).rewards.numpy()
        assert set(np.unique(r)) <= {0.0, -1.0}
