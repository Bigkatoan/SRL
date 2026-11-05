"""Loggers for ``agent.learn()`` when no log directory is requested."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

__all__ = ["NullLogger", "make_logger"]


class NullLogger:
    """Implements the trainer-facing subset of :class:`~srl.utils.logger.Logger`.

    Writes nothing to disk. Tracks episode returns/lengths (exposed as
    :attr:`episode_returns`) and, with ``verbose=True``, prints a one-line
    progress summary every *console_interval* steps.
    """

    def __init__(self, verbose: bool = False, console_interval: int = 5_000, window: int = 20):
        self.verbose = verbose
        self.console_interval = max(int(console_interval), 1)
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self._recent: deque[float] = deque(maxlen=window)
        self._running_ret: np.ndarray | None = None
        self._running_len: np.ndarray | None = None
        self._last_metrics: dict[str, float] = {}
        self._next_print = self.console_interval
        self._step = 0

    def set_metadata(self, **metadata: Any) -> None:
        pass

    def configure_env(self, num_envs: int) -> None:
        n = max(int(num_envs), 1)
        self._running_ret = np.zeros(n)
        self._running_len = np.zeros(n, dtype=int)

    def set_step(self, step: int) -> None:
        self._step = int(step)

    def update_episodes(self, reward, done, truncated, step: int | None = None, info=None) -> None:
        r = np.asarray(reward, dtype=np.float64).reshape(-1)
        ended = np.asarray(done, dtype=bool).reshape(-1) | np.asarray(
            truncated, dtype=bool
        ).reshape(-1)
        if self._running_ret is None or self._running_ret.shape[0] != r.shape[0]:
            self.configure_env(r.shape[0])
        self._running_ret[: r.shape[0]] += r
        self._running_len[: r.shape[0]] += 1
        for i in np.flatnonzero(ended):
            self.record_episode(
                step=step or self._step,
                score=float(self._running_ret[i]),
                length=int(self._running_len[i]),
            )
            self._running_ret[i] = 0.0
            self._running_len[i] = 0
        if step is not None:
            self._maybe_print(int(step))

    def record_episode(self, step: int, score: float, length: int, **_: Any) -> None:
        self.episode_returns.append(float(score))
        self.episode_lengths.append(int(length))
        self._recent.append(float(score))

    def record_metrics(
        self,
        metrics: dict[str, float],
        *,
        step: int,
        total_steps: int | None = None,
        prefix: str | None = None,
        console: bool = True,
    ) -> None:
        for k, v in (metrics or {}).items():
            key = f"{prefix}/{k}" if prefix else k
            try:
                self._last_metrics[key] = float(v)
            except (TypeError, ValueError):
                continue
        self._maybe_print(int(step), total_steps)

    def _maybe_print(self, step: int, total_steps: int | None = None) -> None:
        if not self.verbose or step < self._next_print:
            return
        self._next_print = (step // self.console_interval + 1) * self.console_interval
        ret = f"{np.mean(self._recent):.2f}" if self._recent else "n/a"
        total = f"/{total_steps}" if total_steps else ""
        print(f"[srl] step {step}{total} | episode_return_mean={ret}", flush=True)

    @property
    def last_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def close(self) -> None:
        pass

    def finalize(self, status: str = "completed") -> None:
        pass


def make_logger(log_dir: str | None, *, progress: bool = False, log_interval: int = 5_000):
    """A real :class:`~srl.utils.logger.Logger` when *log_dir* is given, else a
    :class:`NullLogger` (printing progress iff *progress*)."""
    if log_dir is None:
        return NullLogger(verbose=progress, console_interval=log_interval)
    from srl.utils.logger import Logger, LoggerConfig

    return Logger(
        log_dir=log_dir,
        verbose=progress,
        config=LoggerConfig(console_interval=log_interval, enable_plots=False),
    )
