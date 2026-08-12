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
    """Save a checkpoint every *save_interval* steps."""

    def __init__(
        self, checkpoint_manager, save_interval: int = 10_000, model=None, optimizer=None
    ) -> None:
        self.cm = checkpoint_manager
        self.save_interval = save_interval
        self.model = model
        self.optimizer = optimizer

    def bind(self, model, optimizer=None) -> None:
        self.model = model
        self.optimizer = optimizer

    def on_step_end(self, step: int, info: dict[str, Any]) -> None:
        if step > 0 and step % self.save_interval == 0 and self.model is not None:
            self.cm.save(model=self.model, optimizer=self.optimizer, step=step, metrics=info)


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
