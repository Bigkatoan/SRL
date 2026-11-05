"""Regression tests for AsyncOffPolicyRunner:

* warmup on an n-env CPU SyncVectorEnv must sample one action PER sub-env
  (it used to draw a single action from the per-env space -> wrong shape);
* `start_step` (resume) makes the runner continue from that absolute step
  instead of restarting at 0;
* a single time-limited env is reset on truncation, and truncated
  transitions store the true terminal obs from CPU vector envs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from test_envlayer_helpers import COUNTER_EP_LEN, CounterEnv

from srl.core.config import AsyncRunnerConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.sync_vector_env import SyncVectorEnv
from srl.runners.async_off_policy_runner import AsyncOffPolicyRunner


class _Buffer:
    capacity = 1000
    n_step = 1
    gamma = 0.99
    use_fp16 = False
    n_envs = 1

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def __len__(self) -> int:
        return len(self.calls)


class _Agent:
    def __init__(self) -> None:
        self.buffer = _Buffer()
        self.predict_calls = 0

    def predict(self, obs_t, deterministic: bool = False):
        self.predict_calls += 1
        t = next(iter(obs_t.values()))
        n = t.shape[0] if t.ndim > 1 else 1
        return torch.zeros((n, 1)), None, None, None

    def update(self):
        return {"loss": 0.0}


def _vec(n: int) -> SyncVectorEnv:
    return SyncVectorEnv([lambda: GymnasiumWrapper(CounterEnv()) for _ in range(n)])


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_warmup_samples_one_action_per_sub_env(use_async) -> None:
    agent = _Agent()
    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=_vec(3),
        total_steps=12,
        runner_cfg=AsyncRunnerConfig(use_async=use_async),
        random_steps=1000,
        update_after=1000,
    )
    runner.run()
    assert agent.predict_calls == 0
    assert len(agent.buffer) == 12
    assert all(np.asarray(c["action"]).shape == (1,) for c in agent.buffer.calls)
    # True terminal obs (t == ep_len) stored on truncation, not the reset frame.
    trunc = [c for c in agent.buffer.calls if bool(c["truncated"].reshape(-1)[0])]
    assert trunc and all(float(c["next_obs"]["state"][0]) == COUNTER_EP_LEN for c in trunc)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_resume_continues_from_start_step(use_async) -> None:
    agent = _Agent()
    seen: list[int] = []
    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=_vec(2),
        total_steps=10,
        runner_cfg=AsyncRunnerConfig(use_async=use_async),
        random_steps=50,
        update_after=1000,
        eval_fn=seen.append,
        start_step=100,
    )
    runner.run()
    assert seen[0] == 102 and seen[-1] == 110
    assert len(agent.buffer) == 10
    # Already past random_steps at step 100 -> policy actions, no warmup.
    assert agent.predict_calls == 5


def test_single_env_resets_on_truncation() -> None:
    agent = _Agent()
    env = GymnasiumWrapper(CounterEnv())
    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=3 * COUNTER_EP_LEN,
        random_steps=1000,
        update_after=1000,
    )
    runner.run()
    ts = [float(np.asarray(c["next_obs"]["state"]).reshape(-1)[0]) for c in agent.buffer.calls]
    assert max(ts) == COUNTER_EP_LEN
    assert ts == [1.0, 2.0, 3.0, 4.0] * 3
