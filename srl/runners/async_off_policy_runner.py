"""AsyncOffPolicyRunner — overlapped data collection and model training.

Architecture
------------
Two execution contexts:

Collector (caller / main thread)
    Owns the environment.  Continuously collects transitions and writes them
    into the replay buffer.  When the update conditions are met it signals the
    trainer via a ``threading.Condition``.  Isaac Lab environments live on the
    main CUDA context — keeping collection here avoids any inter-thread CUDA
    context migration.

Trainer (background daemon thread)
    Owns gradient computation.  Sleeps until signalled by the collector, then
    calls ``agent.update()`` × ``gradient_steps``.  Uses a **separate CUDA
    stream** so GPU compute overlaps with environment stepping on the main
    thread.

Staleness
    The trainer always sees the latest model weights (shared ``agent.model``).
    Actions are sampled from whatever weights the trainer has most recently
    finished writing (Ape-X / IMPALA style), which is negligible staleness
    for off-policy algorithms *as long as the trainer can broadly keep up*
    with the collector's rate of ``update()`` triggers. If it can't --
    e.g. a large ``gradient_steps`` burst per trigger against a small model
    where per-call Python/kernel-launch overhead dominates actual compute --
    ``_pending_updates`` backs up over the course of a run; this is bounded
    (the collector never blocks waiting for it) but not unlimited, and
    ``run()`` always fully drains the backlog before returning (see
    ``_run_async``'s ``finally`` block) rather than silently completing with
    fewer gradient updates than the config's ``update_every``/
    ``gradient_steps`` promise for however many steps were collected.

GPU buffer
    When ``AsyncRunnerConfig.use_gpu_buffer=True`` the runner replaces the
    agent's default CPU numpy buffer with a
    :class:`~srl.core.gpu_replay_buffer.GPUReplayBuffer`.  Isaac Lab step()
    returns CUDA tensors — these are written directly to the GPU buffer with
    no host round-trip.

Usage::

    from srl.runners import AsyncOffPolicyRunner
    from srl.core.config import AsyncRunnerConfig

    runner = AsyncOffPolicyRunner(
        agent=sac_agent,
        env=isaac_env,
        total_steps=1_000_000,
        runner_cfg=AsyncRunnerConfig(use_async=True, use_gpu_buffer=True),
        obs_to_tensor_fn=lambda obs, dev: {k: v.to(dev) for k, v in obs.items()},
        device="cuda:0",
    )
    runner.run()
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from srl.core.base_agent import BaseAgent
from srl.core.config import AsyncRunnerConfig
from srl.utils.spaces import sample_action_space


class AsyncOffPolicyRunner:
    """Off-policy runner with optional async collection / training split.

    Parameters
    ----------
    agent:
        An off-policy agent (SAC / DDPG / TD3) with a ``buffer`` attribute,
        ``predict()``, and ``update()`` methods.
    env:
        Vectorised or single-env Gymnasium-compatible environment.
    total_steps:
        Total environment steps to collect.
    runner_cfg:
        :class:`~srl.core.config.AsyncRunnerConfig` controlling async mode /
        GPU buffer selection.
    log_fn:
        Optional callback ``(step, metrics_dict) -> None`` for metric logging.
    obs_to_tensor_fn:
        Converts raw env observation to a ``dict[str, Tensor]`` on *device*.
        Default: wraps flat numpy array in ``{\"obs\": tensor}``.
    device:
        Device for training tensors.
    random_steps:
        Number of steps to take with random actions before updates start.
    update_after:
        Minimum buffer size before any updates.
    update_every:
        Updates are triggered every ``update_every`` env steps.
    gradient_steps:
        Number of ``agent.update()`` calls per trigger.
    eval_fn:
        Optional callback ``(step) -> None`` invoked once per collector-loop
        iteration (from the collector/main thread, never the background
        trainer thread) with the runner's current step count. Own scheduling
        (e.g. "only actually evaluate every N steps") and metric logging are
        entirely the callback's responsibility -- the runner does not know
        or care about ``--eval-freq``, it just calls this every iteration so
        the callback's own step check can fire when due. This mirrors
        ``srl.cli.train``'s ``_maybe_run_evaluation`` contract: build the
        eval env once (before this runner's background trainer thread
        starts, if the caller wants to dodge the mjlab CUDA-graph-capture
        race documented in ``_make_visualizer``'s use of the same env type),
        reuse it across calls, close it once the caller is done with
        ``run()``. Without this, ``--eval-freq``/``--eval-episodes`` are
        silently inert whenever ``AsyncOffPolicyRunner`` is active (i.e.
        whenever ``use_async`` or ``use_gpu_buffer`` is set) -- confirmed on
        a real run: zero ``eval/score_mean`` entries ever written despite a
        configured ``--eval-freq``, because ``_run_off_policy`` in
        ``srl.cli.train`` hands off to this class and ``return``s before
        reaching its own (otherwise fully working) eval loop further down.
    episode_fn:
        Optional callback ``(reward, done, truncated, step, info) -> None``
        invoked once per collector-loop iteration, right after
        ``env.step()``, with that step's raw (possibly batched) reward/
        done/truncated/info -- the exact positional shape
        ``srl.utils.logger.Logger.update_episodes`` expects. Without this,
        the console/TensorBoard ``train/score_mean``/``train/score_max``/
        ``train/len`` rollout-quality metrics silently never appear whenever
        this runner is active: `_run_off_policy` in ``srl.cli.train`` calls
        ``logger.update_episodes(...)`` itself in its own (sync) collector
        loop, but hands that responsibility entirely to the caller here --
        confirmed on a real run: `sac/*` loss metrics and `runner/collect_fps`
        printed every log interval as expected, but no `score`/`len` ever
        showed up, because nothing was accumulating per-env episode returns
        or detecting when an episode actually ended. Same class of gap as
        ``eval_fn`` above (see its docstring) -- the runner steps the env and
        has the raw per-step reward/done in hand, but was never wired to
        forward it anywhere.
    """

    def __init__(
        self,
        agent: BaseAgent,
        env,
        total_steps: int,
        runner_cfg: AsyncRunnerConfig | None = None,
        log_fn: Callable[[int, dict], None] | None = None,
        obs_to_tensor_fn: Callable | None = None,
        device: str | torch.device = "cpu",
        random_steps: int = 0,
        update_after: int = 1000,
        update_every: int = 1,
        gradient_steps: int = 1,
        eval_fn: Callable[[int], None] | None = None,
        episode_fn: Callable[[Any, Any, Any, int, Any], None] | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.total_steps = total_steps
        self.cfg = runner_cfg or AsyncRunnerConfig()
        self.log_fn = log_fn
        self.device = torch.device(device)
        self.random_steps = random_steps
        self.update_after = update_after
        self.update_every = update_every
        self.gradient_steps = gradient_steps
        self.eval_fn = eval_fn
        self.episode_fn = episode_fn

        if obs_to_tensor_fn is not None:
            self._obs_to_tensor = obs_to_tensor_fn
        else:
            self._obs_to_tensor = self._default_obs_to_tensor

        # ------------------------------------------------------------------
        # GPU buffer swap (optional)
        # ------------------------------------------------------------------
        if self.cfg.use_gpu_buffer:
            from srl.core.gpu_replay_buffer import GPUReplayBuffer

            old_buf = agent.buffer
            agent.buffer = GPUReplayBuffer(
                capacity=old_buf.capacity,
                device=self.device,
                n_step=getattr(old_buf, "n_step", 1),
                gamma=getattr(old_buf, "gamma", 0.99),
                use_fp16=getattr(old_buf, "use_fp16", False),
                num_envs=getattr(old_buf, "n_envs", getattr(old_buf, "num_envs", 1)),
            )

        # `GPUReplayBuffer.add()` accepts a full `(num_envs, ...)` batch of
        # CUDA tensors directly and splits it into per-env rows internally
        # (see its own `add()`). The default `ReplayBuffer` has no such
        # logic -- its storage is plain numpy, so it expects exactly one
        # already-split, numpy-backed transition per call. `_add_transitions`
        # below uses this flag to decide whether to hand the buffer a raw
        # batch or split+convert it first.
        self._buffer_accepts_batched_tensors = bool(self.cfg.use_gpu_buffer)

        # ------------------------------------------------------------------
        # Async inter-thread signalling
        # ------------------------------------------------------------------
        self._train_cond = threading.Condition(threading.Lock())
        self._stop_event = threading.Event()
        self._pending_updates: int = 0  # number of update() calls queued
        self._metrics_lock = threading.Lock()
        self._latest_metrics: dict[str, float] = {}

        # Trainer CUDA stream (separate from default stream used by collector)
        self._train_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the full training loop (blocks until ``total_steps`` reached)."""
        if self.cfg.use_async:
            self._run_async()
        else:
            self._run_sync()

    # ------------------------------------------------------------------
    # Sync mode (same as original _run_off_policy but GPU-buffer-aware)
    # ------------------------------------------------------------------

    def _run_sync(self) -> None:
        obs, _ = self.env.reset()
        num_envs = self._num_envs()
        vectorized = num_envs > 1
        step = 0
        since_last_update = 0
        t_start = time.perf_counter()
        collect_steps = 0

        while step < self.total_steps:
            obs_t = self._obs_to_tensor(obs, self.device)
            remaining = max(self.total_steps - step, 0)
            active_envs = min(num_envs, remaining) if vectorized else 1

            if step < self.random_steps:
                # `.act_space`, not `.action_space`: IsaacLabWrapper is a
                # plain class (not a gym.Wrapper subclass), so it only
                # exposes `.act_space`; GymnasiumWrapper answers to both via
                # gymnasium's own Wrapper property, so this works either
                # way. sample_action_space() falls back to a plain uniform
                # draw when the space has no `.sample()` (isaaclab/mjlab's
                # lightweight space dataclass doesn't). On isaaclab/mjlab,
                # `act_space` is already the BATCHED `(num_envs, action_dim)`
                # space, so a single draw is already the right shape.
                action = sample_action_space(self.env.act_space)
                action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            else:
                action_t, _, _, _ = self.agent.predict(obs_t, deterministic=False)
                action = action_t.cpu().numpy() if hasattr(action_t, "cpu") else action_t

            result = self.env.step(action)
            next_obs, reward, terminated, truncated, info = (
                result if len(result) == 5 else (*result[:4], False)
            )
            done = terminated

            # `done` is true termination only, never OR'd with `truncated`
            # -- see replay_buffer.py's ReplayBatch docstring for why
            # SAC/DDPG/TD3's bootstrap target needs that. Separately: store
            # `info["true_final_obs"]` (IsaacLabWrapper), not `next_obs`,
            # as the transition's next-state -- for isaaclab/mjlab,
            # `next_obs` here is already the FOLLOWING episode's first
            # frame for any sub-env that just ended (right for `obs =
            # next_obs` below, wrong for a Q-target bootstrap). Falls back
            # to `next_obs` itself when the key is absent (any env type
            # without the fix -- same behavior as before this existed).
            true_final_obs = (
                info.get("true_final_obs", next_obs) if isinstance(info, dict) else next_obs
            )
            self._add_transitions(
                obs_t,
                action_t,
                reward,
                done,
                self._obs_to_tensor(true_final_obs, self.device),
                truncated,
            )

            obs = next_obs
            if not vectorized:
                is_done = (
                    bool(terminated)
                    if isinstance(terminated, bool)
                    else bool(np.asarray(terminated).any())
                )
                if is_done:
                    obs, _ = self.env.reset()
            # Vectorised envs (isaaclab/mjlab) auto-reset individual
            # sub-envs inside `env.step()` itself -- calling `env.reset()`
            # here on `terminated.any()` would wipe every parallel env's
            # in-progress episode just because one of them terminated. See
            # `srl/cli/train.py`'s `_run_off_policy`, which has the same
            # `not vectorized_env` guard around its own reset call for the
            # same reason.

            step += active_envs
            since_last_update += active_envs
            collect_steps += active_envs

            # `step` here (post-increment) matches `_run_off_policy`'s own
            # off-policy sync loop, which calls `logger.update_episodes(...)`
            # with `step=env_step` after that same loop's `env_step +=
            # active_envs` -- keep this call site's step value consistent
            # with that convention, not the on-policy loop's pre-increment
            # one (the two sync loops disagree with each other; this runner
            # only replaces the off-policy one).
            if self.episode_fn is not None:
                self.episode_fn(reward, done, truncated, step, info)

            # Update
            if step >= self.update_after and since_last_update >= self.update_every:
                for _ in range(self.gradient_steps):
                    metrics = self.agent.update()
                since_last_update = 0
                if self.log_fn is not None and metrics:
                    elapsed = time.perf_counter() - t_start
                    metrics["runner/collect_fps"] = collect_steps / elapsed
                    self.log_fn(step, metrics)
                    collect_steps = 0
                    t_start = time.perf_counter()

            # Periodic eval -- see this class's docstring (`eval_fn`) for why
            # this call exists at all: without it, `--eval-freq` is silently
            # inert for this runner. Called every iteration (cheap: the
            # callback owns its own due-or-not check) from this thread only
            # -- `_run_sync` has no background trainer thread, so there's no
            # concurrent-CUDA-work hazard to dodge here the way `_run_async`
            # has to.
            if self.eval_fn is not None:
                self.eval_fn(step)

    # ------------------------------------------------------------------
    # Async mode
    # ------------------------------------------------------------------

    def _run_async(self) -> None:
        trainer_thread = threading.Thread(
            target=self._trainer_loop,
            name="srl-trainer",
            daemon=True,
        )
        trainer_thread.start()

        obs, _ = self.env.reset()
        num_envs = self._num_envs()
        vectorized = num_envs > 1
        step = 0
        since_last_update = 0
        t_start = time.perf_counter()
        collect_steps = 0
        log_interval = max(self.update_every, 1000)
        next_log_step = log_interval

        try:
            while step < self.total_steps:
                obs_t = self._obs_to_tensor(obs, self.device)
                remaining = max(self.total_steps - step, 0)
                active_envs = min(num_envs, remaining) if vectorized else 1

                if step < self.random_steps:
                    # See _run_sync's comment on the same line.
                    action = sample_action_space(self.env.act_space)
                    action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                else:
                    action_t, _, _, _ = self.agent.predict(obs_t, deterministic=False)
                    action = (
                        action_t.detach().cpu().numpy() if hasattr(action_t, "cpu") else action_t
                    )

                result = self.env.step(action)
                next_obs, reward, terminated, truncated, info = (
                    result if len(result) == 5 else (*result[:4], False)
                )
                done = terminated

                # `done` stays true-termination-only, and the stored next-
                # state uses `info["true_final_obs"]` when available -- see
                # _run_sync's comments on both, immediately above.
                true_final_obs = (
                    info.get("true_final_obs", next_obs) if isinstance(info, dict) else next_obs
                )
                self._add_transitions(
                    obs_t,
                    action_t,
                    reward,
                    done,
                    self._obs_to_tensor(true_final_obs, self.device),
                    truncated,
                )

                obs = next_obs
                if not vectorized:
                    is_done = (
                        bool(terminated)
                        if isinstance(terminated, bool)
                        else bool(np.asarray(terminated).any())
                    )
                    if is_done:
                        obs, _ = self.env.reset()
                # See _run_sync's comment on the same guard: vectorised envs
                # auto-reset individual sub-envs internally, so a full
                # `env.reset()` here on any single termination would wipe
                # every parallel env's progress.

                step += active_envs

                # Post-increment step, matching _run_sync's comment on the
                # equivalent call.
                if self.episode_fn is not None:
                    self.episode_fn(reward, done, truncated, step, info)
                since_last_update += active_envs
                collect_steps += active_envs

                # Signal trainer when update conditions met
                if step >= self.update_after and since_last_update >= self.update_every:
                    with self._train_cond:
                        self._pending_updates += self.gradient_steps
                        self._train_cond.notify()
                    since_last_update = 0

                # Logging. `step` no longer advances by exactly 1 each
                # iteration once `active_envs` can be > 1, so an exact `%`
                # trigger could stride past every multiple of
                # `log_interval` and never fire -- use a monotonic
                # threshold instead.
                if self.log_fn is not None and step >= next_log_step:
                    elapsed = time.perf_counter() - t_start
                    with self._metrics_lock:
                        m = dict(self._latest_metrics)
                    m["runner/collect_fps"] = collect_steps / max(elapsed, 1e-6)
                    self.log_fn(step, m)
                    collect_steps = 0
                    t_start = time.perf_counter()
                    next_log_step = step + log_interval

                # Periodic eval -- see this class's docstring (`eval_fn`).
                # Called from this thread (the collector) only, never from
                # `_trainer_loop`'s background thread: the callback's own
                # eval env was built by the caller *before* `run()` started
                # the trainer thread, specifically so its one-time mjlab
                # CUDA-graph capture never overlaps with the trainer's
                # concurrent `agent.update()` kernels on `_train_stream` --
                # see `srl.cli.train._make_visualizer`'s near-identical race
                # (confirmed repro there: env construction concurrent with
                # another thread's GPU work corrupts the capture, hard
                # SIGABRT). Once built, stepping the reused eval env here is
                # no different from the collector already sharing the GPU
                # with the trainer thread every other iteration of this
                # loop.
                if self.eval_fn is not None:
                    self.eval_fn(step)

        finally:
            # `_stop_event` means "no more *new* work is coming" (the
            # collector loop above has already stopped signalling by this
            # point) -- it must NOT mean "abandon whatever's still queued".
            # `_trainer_loop`'s own loop condition (below) keeps draining
            # `_pending_updates` to 0 even after this is set, so a plain,
            # untimed `join()` here is safe (bounded by however long that
            # backlog takes to process, never infinite) and is what makes
            # the wait actually complete rather than abandoning it.
            #
            # This matters for two real, confirmed-via-a-real-mjlab-run
            # failure modes of the old "join with a short timeout, then
            # move on regardless" version:
            #
            # 1. Silent under-training: if the trainer can't fully keep up
            #    with the collector in real time (a real risk with a large
            #    `gradient_steps` burst per trigger -- e.g. this repo's own
            #    SAC config uses `gradient_steps: 512` -- vs. a small model
            #    where `agent.update()`'s Python/kernel-launch overhead
            #    dominates its actual compute), `_pending_updates` backs up
            #    over the course of a run. Cutting the wait off part way
            #    through the backlog means `run()` returns -- and the
            #    caller checkpoints -- with a materially smaller
            #    `encoder_update_counter` than the config's own
            #    `update_every`/`gradient_steps` promise for that many
            #    collected steps (confirmed: 6212-6350 actual vs 10240
            #    expected for a real 15000-step/16-env SAC run against
            #    `Javis-Payload-Rough` -- roughly 38% of the intended
            #    gradient updates silently missing from the saved
            #    checkpoint).
            # 2. A process-exit crash: a daemon thread left running past
            #    `run()` returning is still touching CUDA (`self.
            #    _train_stream`, `self.agent`'s parameters/optimizer state)
            #    exactly when a caller's next steps -- checkpoint save,
            #    `env.close()`, process exit -- start tearing down CUDA
            #    state out from under it. Confirmed via the same real run:
            #    reliable SIGABRT ("terminate called without an active
            #    exception", exit code 134) immediately after "Done." even
            #    though training and checkpointing had both already
            #    completed successfully.
            self._stop_event.set()
            with self._train_cond:
                self._train_cond.notify_all()
            trainer_thread.join()
            if self._train_stream is not None:
                # Flush the trainer's stream deterministically here, in the
                # collector thread, right after we know it has fully
                # stopped touching it -- rather than leaving that
                # synchronisation to whatever finalises `self._train_stream`
                # (a `torch.cuda.Stream` object) at some later,
                # unpredictable point (a `del`, GC, or interpreter
                # shutdown). `_trainer_loop` also self-synchronises this
                # same stream right before it returns; this is a second,
                # belt-and-suspenders sync from the collector thread's own
                # context, since it's this thread (not the now-finished
                # trainer thread) whose CUDA context state a caller's next
                # steps will actually observe.
                torch.cuda.synchronize(self.device)

    def _trainer_loop(self) -> None:
        """Background daemon: wait for signal, run gradient_steps updates."""
        stream_ctx = (
            torch.cuda.stream(self._train_stream) if self._train_stream is not None else _nullctx()
        )
        with stream_ctx:
            while True:
                with self._train_cond:
                    while self._pending_updates == 0 and not self._stop_event.is_set():
                        self._train_cond.wait(timeout=0.1)
                    # Stopping only ends the loop once every already-queued
                    # update has actually been claimed here -- `stop_event`
                    # being set does not by itself mean there's nothing left
                    # to do; see `_run_async`'s `finally` block for why
                    # abandoning a nonzero backlog here is exactly the bug
                    # this loop used to have (either dropped it silently, if
                    # this was reached while idle-waiting right as stop was
                    # set, or left it for a fixed, sometimes-insufficient
                    # join timeout to cut off).
                    if self._pending_updates == 0 and self._stop_event.is_set():
                        break
                    n = self._pending_updates
                    self._pending_updates = 0

                metrics: dict[str, float] = {}
                for _ in range(n):
                    m = self.agent.update()
                    if m:
                        metrics = m  # keep latest

                if metrics:
                    with self._metrics_lock:
                        self._latest_metrics.update(metrics)

            # About to return (and exit the `stream_ctx` block above) with
            # the stop signal observed -- make sure every kernel already
            # queued on this stream has actually finished before the
            # collector thread (in `_run_async`'s `finally`, right after
            # `trainer_thread.join()`) treats us as fully stopped and moves
            # on to whatever comes next (checkpoint save, `env.close()`,
            # process exit). Without this, `join()` returning only means the
            # Python thread function returned, not that the GPU has actually
            # caught up -- the queued work is still in flight on
            # `self._train_stream` when it does.
            if self._train_stream is not None:
                self._train_stream.synchronize()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _num_envs(self) -> int:
        return int(getattr(self.env, "num_envs", 1))

    def _add_transitions(
        self,
        obs_t,
        action_t: torch.Tensor,
        reward,
        done,
        next_obs_t,
        truncated,
    ) -> None:
        """Write one ``env.step()``'s worth of transitions into the buffer.

        Vectorised envs (``num_envs > 1``) must be split into ``num_envs``
        separate per-env transitions before reaching the default numpy
        :class:`~srl.core.replay_buffer.ReplayBuffer` -- unlike
        :class:`~srl.core.gpu_replay_buffer.GPUReplayBuffer` (which splits a
        batched ``(num_envs, ...)`` tensor into per-env rows internally, see
        its own ``add()``), the plain buffer has no such logic: a bare
        batched call would raise ``TypeError`` the moment ``obs_t`` lives on
        CUDA (numpy can't ingest a CUDA tensor), and even on CPU would
        silently collapse all ``num_envs`` transitions into a single buffer
        slot -- mean-reduced reward, any()-reduced done, one buffer slot
        spent per ``env.step()`` call instead of ``num_envs``.
        """
        n = self._num_envs()
        if n <= 1:
            self._write_one(obs_t, action_t, reward, done, next_obs_t, truncated, env_idx=0)
            return

        reward_t = reward if isinstance(reward, torch.Tensor) else torch.as_tensor(reward)
        done_t = done if isinstance(done, torch.Tensor) else torch.as_tensor(done)
        trunc_t = truncated if isinstance(truncated, torch.Tensor) else torch.as_tensor(truncated)
        for i in range(n):
            obs_i = {k: v[i] for k, v in obs_t.items()} if isinstance(obs_t, dict) else obs_t[i]
            next_obs_i = (
                {k: v[i] for k, v in next_obs_t.items()}
                if isinstance(next_obs_t, dict)
                else next_obs_t[i]
            )
            self._write_one(
                obs_i,
                action_t[i],
                reward_t[i],
                done_t[i],
                next_obs_i,
                trunc_t[i],
                env_idx=i,
            )

    def _write_one(
        self,
        obs_i,
        action_i,
        reward_i,
        done_i,
        next_obs_i,
        truncated_i,
        *,
        env_idx: int,
    ) -> None:
        if self._buffer_accepts_batched_tensors:
            # GPUReplayBuffer: zero-copy, stays on `self.device`.
            self.agent.buffer.add(
                obs=obs_i,
                action=action_i,
                reward=reward_i,
                next_obs=next_obs_i,
                done=done_i,
                truncated=truncated_i,
            )
            return

        # Default ReplayBuffer: numpy-backed, needs plain CPU data -- and a
        # single already-split transition, not a batch. `env_idx` selects
        # the right per-env n-step accumulator when `n_step > 1`.
        def _to_np(t):
            return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

        obs_np = (
            {k: _to_np(v) for k, v in obs_i.items()} if isinstance(obs_i, dict) else _to_np(obs_i)
        )
        next_obs_np = (
            {k: _to_np(v) for k, v in next_obs_i.items()}
            if isinstance(next_obs_i, dict)
            else _to_np(next_obs_i)
        )
        self.agent.buffer.add(
            obs=obs_np,
            action=_to_np(action_i),
            reward=np.array([float(reward_i)], dtype=np.float32),
            done=np.array([bool(done_i)], dtype=bool),
            truncated=np.array([bool(truncated_i)], dtype=bool),
            next_obs=next_obs_np,
            env_idx=env_idx,
        )

    def _default_obs_to_tensor(
        self,
        obs,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float32)
                out[k] = t.to(device=device, non_blocking=True)
            return out
        t = obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        return {"obs": t.to(device=device, non_blocking=True)}


class _nullctx:
    """No-op context manager for non-CUDA devices."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
