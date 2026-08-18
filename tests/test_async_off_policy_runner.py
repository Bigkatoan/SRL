"""Direct unit tests for `AsyncOffPolicyRunner` against a fake vectorised env.

Regression tests for a cluster of bugs found when actually exercising this
runner against a vectorised (isaaclab/mjlab-shaped) environment for the
first time -- previously the only coverage on this class was indirect (via
`GPUReplayBuffer`'s own tests in test_buffers.py); nothing exercised the
runner's own collection loop under `num_envs > 1`:

* `step` was incremented by 1 per `env.step()` call regardless of
  `num_envs`, instead of by `num_envs` (matching `srl/cli/train.py`'s own
  sync `_run_off_policy` convention, where one vectorised `env.step()` call
  advances `env_step` by `active_envs`). This makes `total_steps` mean
  "number of `env.step()` calls" under the async runner but "total
  env-transitions across all parallel envs" everywhere else -- a
  `num_envs`x discrepancy in both wall-clock run length and every
  step-gated threshold (`random_steps`, `update_after`, `update_every`,
  and derived metrics like `train/fps`).
* `env.reset()` (a full vectorised-env reset) was called whenever *any*
  parallel env terminated, discarding every other env's in-progress
  episode. mjlab/isaaclab-style vectorised envs auto-reset individual
  sub-envs internally -- calling `env.reset()` mid-rollout is only correct
  for a genuinely single (non-vectorised) env.
* A full `(num_envs, ...)` batch was hand off to `agent.buffer.add()` in one
  call. `GPUReplayBuffer.add()` tolerates this (it splits internally), but
  the default numpy `ReplayBuffer` does not -- it silently collapses the
  whole batch into a single buffer slot (mean-reduced reward, any()-reduced
  done) and would raise `TypeError` outright the moment the observation
  tensor lives on CUDA (numpy cannot ingest a CUDA tensor).
* `_run_async`'s shutdown joined the trainer thread with a fixed 10s
  timeout and moved on regardless of whether it had actually finished, and
  `_trainer_loop` exited as soon as `_stop_event` was set even with a
  nonzero backlog still queued (dropped outright if it happened to be idle-
  waiting right as stop was set; otherwise left for the timeout to cut off
  mid-drain). Confirmed via a real 15000-step/16-env SAC run against
  `Javis-Payload-Rough` with `gradient_steps=512`: the trainer fell behind
  the collector's rate of update triggers, `run()` returned -- and the
  caller checkpointed -- with only ~6250 of the 10240 gradient updates the
  config's `update_every`/`gradient_steps` actually call for (roughly 39%
  silently missing), and the abandoned daemon thread's still-in-flight CUDA
  work reliably crashed the process at exit (SIGABRT, "terminate called
  without an active exception") immediately after "Done." had already
  printed and the (silently incomplete) checkpoint had already saved.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from srl.core.config import AsyncRunnerConfig
from srl.runners.async_off_policy_runner import AsyncOffPolicyRunner

OBS_DIM = 4
ACTION_DIM = 2


class _FakeVecEnv:
    """Mirrors IsaacLabWrapper's contract: numpy dict obs, batched act_space,
    `num_envs`, and internal per-sub-env auto-reset (env index 0 "terminates"
    every third step, no other env ever does -- if the runner ever calls a
    full `env.reset()` because of that, `reset_calls` catches it)."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.reset_calls = 0
        self.step_calls = 0
        self.act_space = SimpleNamespace(
            shape=(num_envs, ACTION_DIM),
            low=np.full((num_envs, ACTION_DIM), -1.0, dtype=np.float32),
            high=np.full((num_envs, ACTION_DIM), 1.0, dtype=np.float32),
        )

    def reset(self, **kwargs):
        self.reset_calls += 1
        obs = {"state": np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)}
        return obs, {}

    def step(self, action):
        self.step_calls += 1
        obs = {"state": np.full((self.num_envs, OBS_DIM), float(self.step_calls), dtype=np.float32)}
        # Distinct per-env reward (0..num_envs-1) so a mean-reduction bug
        # (all envs collapsed into one buffer slot) is directly observable.
        reward = np.arange(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        if self.step_calls % 3 == 0:
            terminated[0] = True  # only env 0 -- others must keep running
        truncated = np.zeros(self.num_envs, dtype=bool)
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        pass


class _FakeBuffer:
    """Records every `add()` call; enough attributes for the runner's
    optional GPUReplayBuffer swap to introspect (`capacity`, etc.)."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self.n_step = 1
        self.gamma = 0.99
        self.use_fp16 = False
        self.n_envs = 1
        self.calls: list[dict] = []

    def add(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def __len__(self) -> int:
        return len(self.calls)


class _FakeAgent:
    def __init__(self) -> None:
        self.buffer = _FakeBuffer()
        self.update_calls = 0

    def predict(self, obs_t, deterministic: bool = False):
        n = next(iter(obs_t.values())).shape[0]
        action = torch.zeros((n, ACTION_DIM), dtype=torch.float32)
        return action, None, None, None

    def update(self):
        self.update_calls += 1
        return {"loss": 0.0}


def _obs_to_tensor(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_step_count_matches_total_steps_not_env_step_call_count(use_async) -> None:
    """`total_steps` must be reached in real env-transitions (num_envs each
    `env.step()` call), not in raw `env.step()` call count."""
    num_envs = 4
    n_iters = 5
    total_steps = num_envs * n_iters
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=use_async),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
    )
    runner.run()

    assert env.step_calls == n_iters, (
        f"expected exactly {n_iters} env.step() calls for total_steps="
        f"{total_steps} at num_envs={num_envs}, got {env.step_calls} -- "
        "step counting is not accounting for num_envs per call"
    )
    assert len(agent.buffer) == total_steps, (
        f"expected {total_steps} individual buffer transitions (one per "
        f"env per step), got {len(agent.buffer)}"
    )


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_vectorized_env_never_gets_a_full_reset_mid_rollout(use_async) -> None:
    """env 0 "terminates" every 3rd step in `_FakeVecEnv`; the other 3 envs
    never do. A full `env.reset()` must never fire mid-rollout for a
    vectorised env -- mjlab/isaaclab auto-reset the terminated sub-env
    internally."""
    num_envs = 4
    total_steps = num_envs * 9  # step_calls hits the %3==0 branch 3x
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=use_async),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
    )
    runner.run()

    assert env.reset_calls == 1, (
        f"env.reset() was called {env.reset_calls} times; expected exactly "
        "1 (the initial reset) -- a vectorised env must never be fully "
        "reset just because one of its parallel sub-envs terminated"
    )


def test_batched_transition_is_split_per_env_not_averaged() -> None:
    """The default (non-GPU) buffer must receive `num_envs` separate
    `add()` calls per `env.step()`, each with that one env's own reward --
    not one call per step with the whole batch (or its mean) baked in."""
    num_envs = 4
    total_steps = num_envs * 2
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=False),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
    )
    runner.run()

    assert len(agent.buffer.calls) == total_steps
    # First env.step() call's rewards are np.arange(num_envs) = [0, 1, 2, 3];
    # the first `num_envs` buffer.add() calls must carry those distinct
    # per-env values, not a single mean-reduced scalar repeated 4x.
    first_batch_rewards = sorted(float(c["reward"][0]) for c in agent.buffer.calls[:num_envs])
    assert first_batch_rewards == [0.0, 1.0, 2.0, 3.0], (
        f"expected per-env rewards [0, 1, 2, 3] from the first env.step() "
        f"call, got {first_batch_rewards} -- looks like the batch was "
        "collapsed (e.g. mean-reduced) instead of split per env"
    )
    # Every recorded obs/action/reward/done/truncated must be numpy, not a
    # lingering CUDA/CPU torch tensor -- the default ReplayBuffer's storage
    # is plain numpy.
    for call in agent.buffer.calls:
        assert isinstance(call["reward"], np.ndarray)
        assert isinstance(call["obs"]["state"], np.ndarray)


def test_async_runner_with_gpu_buffer_on_vectorized_env_runs_to_completion() -> None:
    """The full combination the runner's own docstring advertises --
    `use_async=True` *and* `use_gpu_buffer=True` -- against a vectorised,
    dict-obs env. Runs on CPU (device="cpu" is a supported GPUReplayBuffer
    target, see test_buffers.py) since this suite has no CUDA device, but
    exercises the exact same code path CUDA would take."""
    num_envs = 4
    total_steps = num_envs * 6
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=True, use_gpu_buffer=True),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=num_envs,
        update_every=num_envs,
        gradient_steps=1,
    )
    runner.run()

    from srl.core.gpu_replay_buffer import GPUReplayBuffer

    assert isinstance(agent.buffer, GPUReplayBuffer)
    assert len(agent.buffer) == total_steps
    assert env.step_calls == total_steps // num_envs
    assert env.reset_calls == 1
    assert agent.update_calls > 0, "no gradient updates ran at all"


class _SlowFakeAgent(_FakeAgent):
    """`update()` deliberately takes long enough that the trainer thread
    cannot keep up with a fast (near-instant, fake-env) collector -- forcing
    a real backlog to build up in `_pending_updates`, the exact condition
    that exposed the drain bug on a real mjlab run (there, caused by
    `gradient_steps=512` worth of real GPU work per trigger outpacing a slow
    collector instead, but the runner-level effect -- backlog accumulates
    faster than the trainer can drain it -- is identical)."""

    def update(self):
        time.sleep(0.004)
        return super().update()


def test_run_fully_drains_backlog_before_returning_even_when_trainer_falls_behind() -> None:
    """Regression test for the async shutdown bug: `run()` must not return
    until every triggered `update()` call has actually executed, even if
    the trainer was still working through a backlog when collection
    finished. The old code joined the trainer thread with a fixed timeout
    and moved on regardless -- on a real run this meant checkpointing with
    ~39% of the intended gradient updates silently missing, and left a
    daemon thread with live CUDA work running past `run()` returning
    (confirmed to reliably SIGABRT the process at exit)."""
    num_envs = 4
    n_iters = 20
    total_steps = num_envs * n_iters
    update_every = num_envs  # every iteration is an update trigger
    gradient_steps = 25
    expected_total_updates = n_iters * gradient_steps  # 500

    env = _FakeVecEnv(num_envs)
    agent = _SlowFakeAgent()

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=True),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=update_every,
        gradient_steps=gradient_steps,
    )

    t0 = time.perf_counter()
    runner.run()
    elapsed = time.perf_counter() - t0

    assert agent.update_calls == expected_total_updates, (
        f"expected all {expected_total_updates} triggered update() calls to "
        f"have run by the time run() returned, got {agent.update_calls} -- "
        "the trainer's backlog was abandoned instead of fully drained"
    )
    # The fake collector loop itself finishes in well under 100ms (pure
    # numpy on tiny arrays); 500 updates x 4ms = ~2s of unavoidable trainer
    # work. If run() returned quickly anyway, the drain was skipped rather
    # than waited for.
    assert elapsed >= 1.5, (
        f"run() returned after only {elapsed:.2f}s, too fast to have actually "
        "waited for ~2s of trainer backlog -- looks like it returned without "
        "draining"
    )

    # No thread should still be alive/touching the agent after run()
    # returns -- the whole point of draining fully before returning is so a
    # caller can safely checkpoint/close the env/exit immediately after.
    trainer_threads = [t for t in threading.enumerate() if t.name == "srl-trainer"]
    assert not any(t.is_alive() for t in trainer_threads)


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_eval_fn_called_every_iteration_with_current_step(use_async) -> None:
    """`eval_fn(step)` must be invoked once per collector-loop iteration for
    both sync and async modes.

    Regression test: before `eval_fn` existed, `AsyncOffPolicyRunner` had no
    hook at all for periodic evaluation. `srl.cli.train._run_off_policy`
    hands off to this runner and `return`s as soon as `run()` completes --
    entirely skipping the `--eval-freq`/`--eval-episodes` machinery further
    down in that same function. Net effect on a real run: `--eval-freq` is
    silently inert (zero `eval/score_mean` entries ever logged) any time
    `use_async` or `use_gpu_buffer` is set, i.e. exactly the configuration
    this runner exists for.
    """
    num_envs = 4
    n_iters = 5
    total_steps = num_envs * n_iters
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()
    calls: list[int] = []

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=use_async),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
        eval_fn=calls.append,
    )
    runner.run()

    assert calls == [num_envs * i for i in range(1, n_iters + 1)], (
        "expected eval_fn to be called once per collector-loop iteration "
        f"with the running step count, got {calls}"
    )


def test_eval_fn_not_called_when_absent() -> None:
    """`eval_fn=None` (the default) must not change collection behavior at
    all -- a plain smoke test that the new parameter is truly optional."""
    num_envs = 4
    total_steps = num_envs * 5
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=True),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
    )
    runner.run()  # must not raise

    assert env.step_calls == total_steps // num_envs


def test_eval_fn_called_from_collector_thread_not_trainer_thread() -> None:
    """`eval_fn` must run on the collector (main) thread, never the
    background trainer thread.

    Calling it from the trainer thread would let eval-env construction/
    stepping race the trainer's concurrent CUDA work on `_train_stream` --
    the same category of bug already documented and fixed once in this
    codebase for `--visualize` (`srl.cli.train._make_visualizer`: building a
    new mjlab env concurrently with another thread's ongoing GPU work
    corrupts mjlab's one-time CUDA graph capture, confirmed repro was a hard
    SIGABRT). The caller (`srl.cli.train._run_off_policy`) is expected to
    build its eval env before starting this runner's trainer thread and
    reuse it thereafter; this test only guards the runner's half of that
    contract -- that it never invokes the callback from `_trainer_loop`.
    """
    num_envs = 4
    total_steps = num_envs * 5
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()
    thread_names: list[str] = []

    def _eval_fn(step: int) -> None:
        thread_names.append(threading.current_thread().name)

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=True),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
        eval_fn=_eval_fn,
    )
    runner.run()

    assert thread_names, "eval_fn was never called"
    assert all(name != "srl-trainer" for name in thread_names), (
        f"eval_fn was called from thread(s) {set(thread_names)} -- must "
        "only ever run on the collector thread"
    )


@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
def test_episode_fn_called_after_every_env_step_with_raw_transition(use_async) -> None:
    """`episode_fn(reward, done, truncated, step, info)` must fire once per
    collector-loop iteration, right after `env.step()`, with that step's raw
    (possibly batched) transition data.

    Regression test: before `episode_fn` existed, nothing on this runner's
    collection path ever called `srl.utils.logger.Logger.update_episodes` --
    the one call `srl.cli.train._run_off_policy`'s own sync loop makes every
    step to accumulate per-env episode returns and detect completions. Net
    effect on a real run: `sac/*` loss metrics and `runner/collect_fps`
    printed on schedule as expected, but `train/score_mean`/`train/len` (the
    actual "is this thing learning" signal) never appeared at all, silently,
    for the entire duration `AsyncOffPolicyRunner` is active.
    """
    num_envs = 4
    n_iters = 5
    total_steps = num_envs * n_iters
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()
    calls: list[tuple] = []

    def _episode_fn(reward, done, truncated, step, info) -> None:
        calls.append((np.asarray(reward).copy(), np.asarray(done).copy(), step))

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=use_async),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
        episode_fn=_episode_fn,
    )
    runner.run()

    assert len(calls) == n_iters, (
        f"expected episode_fn called once per collector-loop iteration "
        f"({n_iters} iterations), got {len(calls)} calls"
    )
    # _FakeVecEnv's first step() returns reward=[0,1,2,3] -- confirms the
    # raw, unreduced per-env array reaches the callback, not e.g. a mean.
    first_reward, _, first_step = calls[0]
    assert first_reward.tolist() == [0.0, 1.0, 2.0, 3.0]
    # Post-increment step, matching `_run_off_policy`'s own off-policy sync
    # loop (`logger.update_episodes(..., step=env_step, ...)` called after
    # that loop's `env_step += active_envs`).
    assert first_step == num_envs
    # env 0 "terminates" every 3rd env.step() call in _FakeVecEnv -- the
    # 3rd callback invocation must see that in `done`.
    _, third_done, _ = calls[2]
    assert bool(third_done[0]) is True
    assert not bool(third_done[1])


def test_episode_fn_not_called_when_absent() -> None:
    """`episode_fn=None` (the default) must not change collection behavior
    at all -- a plain smoke test that the new parameter is truly optional."""
    num_envs = 4
    total_steps = num_envs * 5
    env = _FakeVecEnv(num_envs)
    agent = _FakeAgent()

    runner = AsyncOffPolicyRunner(
        agent=agent,
        env=env,
        total_steps=total_steps,
        runner_cfg=AsyncRunnerConfig(use_async=True),
        obs_to_tensor_fn=_obs_to_tensor,
        device="cpu",
        random_steps=0,
        update_after=0,
        update_every=num_envs,
        gradient_steps=1,
    )
    runner.run()  # must not raise

    assert env.step_calls == total_steps // num_envs
