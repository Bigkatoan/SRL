"""Training callbacks for logging, early stopping, and evaluation."""

from __future__ import annotations

from typing import Any


class BaseCallback:
    """Abstract callback interface."""

    def on_step_end(self, step: int, info: dict[str, Any]) -> None:
        pass

    def on_episode_end(self, episode: int, info: dict[str, Any]) -> None:
        pass

    def on_training_end(self) -> None:
        pass


class LogCallback(BaseCallback):
    """Log metrics via an SRL Logger every *log_interval* steps."""

    def __init__(self, logger, log_interval: int = 1000) -> None:
        self.logger = logger
        self.log_interval = log_interval

    def on_step_end(self, step: int, info: dict[str, Any]) -> None:
        if step % self.log_interval == 0:
            if hasattr(self.logger, "record_metrics"):
                self.logger.record_metrics(info, step=step)
            else:
                self.logger.log_dict(info, step=step)


class CheckpointCallback(BaseCallback):
    """Save a checkpoint every *save_interval* steps.

    Triggers on CROSSING a multiple of ``save_interval`` (``step >=
    self._next_save_step``), not on landing exactly on one (the previous
    ``step % self.save_interval == 0`` -- found via a real run where zero
    periodic checkpoints were ever written across a 120M-step PPO run: with
    `n_steps=24`/`n_envs=4096`, `step` only ever advances in increments of
    98304, and `gcd(98304, 100_000) == 32`, so the first step number that is
    ALSO an exact multiple of the default `save_interval=100_000` is
    `lcm(98304, 100_000) == 307,200,000` -- past the end of every run this
    project has actually done. Any rollout granularity that doesn't happen
    to share a large common factor with `save_interval` hits this same
    silent no-op. Crossing-based triggering saves exactly once per interval
    regardless of the step increment's size, including a single call that
    jumps past more than one interval at once (rare, but possible with a
    very large rollout granularity relative to `save_interval`) -- it saves
    once immediately and reschedules for the next real boundary above the
    current step, rather than either double-saving or silently skipping
    the skipped-over interval(s) entirely.
    """

    def __init__(
        self, checkpoint_manager, save_interval: int = 10_000, model=None, optimizer=None
    ) -> None:
        self.cm = checkpoint_manager
        self.save_interval = save_interval
        self.model = model
        self.optimizer = optimizer
        self._next_save_step = save_interval

    def bind(self, model, optimizer=None) -> None:
        self.model = model
        self.optimizer = optimizer

    def on_step_end(self, step: int, info: dict[str, Any]) -> None:
        if step >= self._next_save_step and self.model is not None:
            self.cm.save(model=self.model, optimizer=self.optimizer, step=step, metrics=info)
            self._next_save_step = ((step // self.save_interval) + 1) * self.save_interval


class EarlyStopping(BaseCallback):
    """Stop training when a monitor metric has not improved for *patience* evaluations."""

    def __init__(
        self,
        monitor: str = "eval/mean_reward",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float = float("-inf") if mode == "max" else float("inf")
        self._waits = 0
        self.should_stop = False

    def on_episode_end(self, episode: int, info: dict[str, Any]) -> None:
        value = info.get(self.monitor)
        if value is None:
            return
        improved = (
            value > self._best + self.min_delta
            if self.mode == "max"
            else value < self._best - self.min_delta
        )
        if improved:
            self._best = value
            self._waits = 0
        else:
            self._waits += 1
            if self._waits >= self.patience:
                self.should_stop = True
