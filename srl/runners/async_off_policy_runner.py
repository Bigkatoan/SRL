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
    A maximum staleness of one gradient step is accepted (Ape-X / IMPALA
    style), which is negligible for off-policy algorithms.

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

import time
import threading
from typing import Callable

import torch

from srl.core.base_agent import BaseAgent
from srl.core.config import AsyncRunnerConfig


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

        # ------------------------------------------------------------------
        # Async inter-thread signalling
        # ------------------------------------------------------------------
        self._train_cond = threading.Condition(threading.Lock())
        self._stop_event = threading.Event()
        self._pending_updates: int = 0       # number of update() calls queued
        self._metrics_lock = threading.Lock()
        self._latest_metrics: dict[str, float] = {}

        # Trainer CUDA stream (separate from default stream used by collector)
        self._train_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
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
        step = 0
        since_last_update = 0
        t_start = time.perf_counter()
        collect_steps = 0

        while step < self.total_steps:
            obs_t = self._obs_to_tensor(obs, self.device)

            if step < self.random_steps:
                action = self.env.action_space.sample()
                action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            else:
                action_t, _, _, _ = self.agent.predict(obs_t, deterministic=False)
                action = action_t.cpu().numpy() if hasattr(action_t, "cpu") else action_t

            result = self.env.step(action)
            next_obs, reward, terminated, truncated, info = result if len(result) == 5 else (*result[:4], {})
            done = terminated

            # Add to buffer — GPU buffer accepts CUDA tensors directly
            self.agent.buffer.add(obs_t, action_t, reward, done, self._obs_to_tensor(next_obs, self.device))

            obs = next_obs
            if isinstance(terminated, (bool,)) and terminated or (hasattr(terminated, "any") and terminated.any()):
                obs, _ = self.env.reset()

            step += 1
            since_last_update += 1
            collect_steps += 1

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
        step = 0
        since_last_update = 0
        t_start = time.perf_counter()
        collect_steps = 0
        train_steps_logged = 0

        try:
            while step < self.total_steps:
                obs_t = self._obs_to_tensor(obs, self.device)

                if step < self.random_steps:
                    action = self.env.action_space.sample()
                    action_t = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                else:
                    action_t, _, _, _ = self.agent.predict(obs_t, deterministic=False)
                    action = action_t.detach().cpu().numpy() if hasattr(action_t, "cpu") else action_t

                result = self.env.step(action)
                next_obs, reward, terminated, truncated, info = result if len(result) == 5 else (*result[:4], {})
                done = terminated

                self.agent.buffer.add(obs_t, action_t, reward, done, self._obs_to_tensor(next_obs, self.device))

                obs = next_obs
                is_done = bool(terminated) if isinstance(terminated, bool) else bool(terminated.any())
                if is_done:
                    obs, _ = self.env.reset()

                step += 1
                since_last_update += 1
                collect_steps += 1

                # Signal trainer when update conditions met
                if step >= self.update_after and since_last_update >= self.update_every:
                    with self._train_cond:
                        self._pending_updates += self.gradient_steps
                        self._train_cond.notify()
                    since_last_update = 0

                # Logging
                if self.log_fn is not None and step % max(self.update_every, 1000) == 0:
                    elapsed = time.perf_counter() - t_start
                    with self._metrics_lock:
                        m = dict(self._latest_metrics)
                    m["runner/collect_fps"] = collect_steps / max(elapsed, 1e-6)
                    self.log_fn(step, m)
                    collect_steps = 0
                    t_start = time.perf_counter()

        finally:
            self._stop_event.set()
            with self._train_cond:
                self._train_cond.notify_all()
            trainer_thread.join(timeout=10.0)

    def _trainer_loop(self) -> None:
        """Background daemon: wait for signal, run gradient_steps updates."""
        stream_ctx = (
            torch.cuda.stream(self._train_stream)
            if self._train_stream is not None
            else _nullctx()
        )
        with stream_ctx:
            while not self._stop_event.is_set():
                with self._train_cond:
                    while self._pending_updates == 0 and not self._stop_event.is_set():
                        self._train_cond.wait(timeout=0.1)
                    if self._stop_event.is_set():
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    def __enter__(self): return self
    def __exit__(self, *_): pass
