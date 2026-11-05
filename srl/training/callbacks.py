"""Training callbacks.

A callback is any object implementing some of the hooks below; every hook is
optional. Trainers call them in this order::

    on_training_start(trainer)
      ┌─ on_step(step)                       # after every (vector) env step
      │  on_rollout_end(step)                 # on-policy: rollout collected, before update()
      │  on_update(step, metrics)             # after an update() produced metrics
      └─ on_eval(step, metrics)               # after an evaluation (Evaluator or EvalCallback)
    on_training_end(trainer)

Any hook may return ``False`` to request that training stops (checked right
after the hook); setting ``trainer.stop_training = True`` or an attribute
``should_stop = True`` on the callback has the same effect.

``on_step`` is not called on the asynchronous off-policy path
(``AsyncOffPolicyRunner``), which owns its own collector thread.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

__all__ = [
    "BaseCallback",
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "EarlyStopping",
    "EvalCallback",
    "LogCallback",
]


class Callback:
    """Base class with no-op hooks (subclassing it is optional)."""

    should_stop: bool = False

    def on_training_start(self, trainer: Any) -> bool | None:  # noqa: B027
        pass

    def on_step(self, step: int) -> bool | None:  # noqa: B027
        pass

    def on_rollout_end(self, step: int) -> bool | None:  # noqa: B027
        pass

    def on_update(self, step: int, metrics: dict[str, Any]) -> bool | None:
        # Backward compatibility: pre-``srl.training`` callbacks implemented
        # ``on_step_end(step, info)``, which the CLI loops called after
        # every update -- exactly this hook's slot.
        return self.on_step_end(step, metrics)

    def on_eval(self, step: int, metrics: dict[str, Any]) -> bool | None:  # noqa: B027
        pass

    def on_training_end(self, trainer: Any) -> bool | None:  # noqa: B027
        pass

    # -- legacy hooks ------------------------------------------------------
    def on_step_end(self, step: int, info: dict[str, Any]) -> bool | None:  # noqa: B027
        """Legacy alias of :meth:`on_update` (kept for old subclasses)."""

    def on_episode_end(self, episode: int, info: dict[str, Any]) -> None:  # noqa: B027
        """Legacy hook; never called by the trainers."""


#: Backward-compatible name used by ``srl.utils.callbacks``.
BaseCallback = Callback


class CallbackList:
    """Dispatch each hook to a list of callbacks (duck-typed, hooks optional)."""

    def __init__(self, callbacks: Iterable[Any] | None = None) -> None:
        if callbacks is None:
            callbacks = []
        elif not isinstance(callbacks, (list, tuple, CallbackList)):
            callbacks = [callbacks]
        self.callbacks: list[Any] = list(callbacks)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self) -> int:
        return len(self.callbacks)

    def append(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def _call(self, hook: str, *args, exclude: Any = None) -> bool:
        keep_going = True
        for cb in self.callbacks:
            if cb is exclude:
                continue
            fn = getattr(cb, hook, None)
            if fn is None and hook == "on_update":
                # Duck-typed legacy callback (not a Callback subclass).
                fn = getattr(cb, "on_step_end", None)
            if fn is None:
                continue
            if fn(*args) is False:
                keep_going = False
            if getattr(cb, "should_stop", False):
                keep_going = False
        return keep_going

    def on_training_start(self, trainer) -> bool:
        return self._call("on_training_start", trainer)

    def on_step(self, step: int) -> bool:
        return self._call("on_step", step)

    def on_rollout_end(self, step: int) -> bool:
        return self._call("on_rollout_end", step)

    def on_update(self, step: int, metrics: dict) -> bool:
        return self._call("on_update", step, metrics)

    def on_eval(self, step: int, metrics: dict, *, exclude: Any = None) -> bool:
        return self._call("on_eval", step, metrics, exclude=exclude)

    def on_training_end(self, trainer) -> bool:
        return self._call("on_training_end", trainer)


class LogCallback(Callback):
    """Log update metrics via an SRL Logger every *log_interval* steps."""

    def __init__(self, logger, log_interval: int = 1000) -> None:
        self.logger = logger
        self.log_interval = log_interval

    def on_update(self, step: int, metrics: dict[str, Any]) -> None:
        if step % self.log_interval == 0:
            if hasattr(self.logger, "record_metrics"):
                self.logger.record_metrics(metrics, step=step)
            else:
                self.logger.log_dict(metrics, step=step)

    on_step_end = on_update


class CheckpointCallback(Callback):
    """Save a checkpoint every *save_interval* steps.

    Triggers on CROSSING a multiple of ``save_interval`` (``step >=
    self._next_save_step``), not on landing exactly on one (the previous
    ``step % self.save_interval == 0`` -- found via a real run where zero
    periodic checkpoints were ever written across a 120M-step PPO run: with
    `n_steps=24`/`n_envs=4096`, `step` only ever advances in increments of
    98304, and `gcd(98304, 100_000) == 32`, so the first step number that is
    ALSO an exact multiple of the default `save_interval=100_000` is
    `lcm(98304, 100_000) == 307,200,000` -- past the end of every run this
    project has actually done. Crossing-based triggering saves exactly once
    per interval regardless of the step increment's size, including a single
    call that jumps past more than one interval at once -- it saves once
    immediately and reschedules for the next real boundary above the current
    step.

    Parameters
    ----------
    checkpoint_manager:
        A :class:`~srl.utils.checkpoint.CheckpointManager`, or a directory
        path (one is created with ``max_keep=5``).
    save_interval:
        Env steps between checkpoints.
    model:
        What to save. Defaults to the trainer's agent (bound in
        :meth:`on_training_start`).
    """

    def __init__(
        self, checkpoint_manager, save_interval: int = 10_000, model=None, optimizer=None
    ) -> None:
        if not hasattr(checkpoint_manager, "save"):
            from srl.utils.checkpoint import CheckpointManager

            checkpoint_manager = CheckpointManager(checkpoint_manager, max_keep=5)
        self.cm = checkpoint_manager
        self.save_interval = int(save_interval)
        self.model = model
        self.optimizer = optimizer
        self._next_save_step = self.save_interval
        self.last_saved_path = None

    def bind(self, model, optimizer=None) -> None:
        self.model = model
        self.optimizer = optimizer

    def on_training_start(self, trainer) -> None:
        if self.model is None:
            self.model = trainer.agent
        start = int(getattr(trainer, "start_step", 0) or 0)
        if start >= self._next_save_step and self.save_interval > 0:
            self._next_save_step = ((start // self.save_interval) + 1) * self.save_interval

    def on_update(self, step: int, metrics: dict[str, Any]) -> None:
        if self.save_interval <= 0:
            return
        if step >= self._next_save_step and self.model is not None:
            self.last_saved_path = self.cm.save(
                model=self.model, optimizer=self.optimizer, step=step, metrics=metrics
            )
            self._next_save_step = ((step // self.save_interval) + 1) * self.save_interval

    on_step_end = on_update


class EarlyStopping(Callback):
    """Stop training when *monitor* has not improved for *patience* evaluations.

    Evaluated on every ``on_eval`` (so it needs evaluation to be enabled:
    ``learn(eval_env=..., eval_freq=...)``, ``--eval-freq`` or an
    :class:`EvalCallback`). Returns ``False`` -- which stops the trainer --
    once ``patience`` consecutive evaluations failed to beat the best value
    by more than ``min_delta``.
    """

    def __init__(
        self,
        monitor: str = "eval/score_mean",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        if monitor == "eval/mean_reward":
            warnings.warn(
                "EarlyStopping(monitor='eval/mean_reward') never matched any metric SRL "
                "emits; using 'eval/score_mean'.",
                DeprecationWarning,
                stacklevel=2,
            )
            monitor = "eval/score_mean"
        self.monitor = monitor
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self._best: float = float("-inf") if mode == "max" else float("inf")
        self._waits = 0
        self.should_stop = False
        self.stopped_step: int | None = None

    def _check(self, step: int | None, metrics: dict[str, Any]) -> bool:
        value = metrics.get(self.monitor)
        if value is None:
            return True
        value = float(value)
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
                self.stopped_step = step
                return False
        return True

    def on_eval(self, step: int, metrics: dict[str, Any]) -> bool:
        return self._check(step, metrics)

    def on_episode_end(self, episode: int, info: dict[str, Any]) -> None:
        """Legacy entry point (pre-``srl.training``); prefer ``on_eval``."""
        self._check(None, info)


class EvalCallback(Callback):
    """Periodically evaluate the agent on *eval_env* during training.

    Results are logged as ``eval/*`` metrics and forwarded to every other
    callback's ``on_eval`` (so e.g. :class:`EarlyStopping` sees them). With
    ``best_model_save_path`` the best-scoring agent is saved there via
    :meth:`BaseAgent.save`.

    Parameters
    ----------
    eval_env:
        Env instance (raw Gymnasium envs are wrapped automatically), or a
        zero-arg factory.
    eval_freq:
        Env steps between evaluations.
    n_eval_episodes:
        Deterministic episodes per evaluation.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        *,
        seed: int = 10_000,
        best_model_save_path: str | None = None,
        final_eval: bool = True,
        verbose: bool = False,
    ) -> None:
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.seed = int(seed)
        self.best_model_save_path = best_model_save_path
        self.final_eval = final_eval
        self.verbose = verbose
        self.best_score: float | None = None
        self.evaluations: list[tuple[int, dict[str, float]]] = []
        self._trainer = None
        self._evaluator = None

    def on_training_start(self, trainer) -> None:
        from srl.training.envs import wrap_env
        from srl.training.evaluation import Evaluator

        env = self.eval_env
        env_fn = None
        if callable(env) and not hasattr(env, "reset"):
            env_fn, env = env, None
        else:
            env = wrap_env(env)
        self._trainer = trainer
        self._evaluator = Evaluator(
            env,
            env_fn=env_fn,
            eval_freq=self.eval_freq,
            n_episodes=self.n_eval_episodes,
            seed=self.seed,
            env_type=getattr(trainer, "env_type", None),
            verbose=self.verbose,
        )
        self._evaluator.reset_schedule(int(getattr(trainer, "start_step", 0) or 0))

    def _after_eval(self, step: int, metrics: dict | None) -> bool:
        if not metrics:
            return True
        self.evaluations.append((step, dict(metrics)))
        score = metrics.get("eval/score_mean")
        if (
            self.best_model_save_path is not None
            and score is not None
            and (self.best_score is None or score > self.best_score)
        ):
            self.best_score = float(score)
            self._trainer.agent.save(self.best_model_save_path)
        return self._trainer.callbacks.on_eval(step, metrics, exclude=self)

    def _maybe_eval(self, step: int) -> bool:
        if self._evaluator is None:
            return True
        metrics = self._evaluator.maybe_evaluate(
            self._trainer.agent,
            step,
            logger=self._trainer.logger,
            total_steps=self._trainer.total_steps,
        )
        return self._after_eval(step, metrics)

    def on_step(self, step: int) -> bool:
        return self._maybe_eval(step)

    def on_update(self, step: int, metrics: dict[str, Any]) -> bool:
        # Also checked here: the async off-policy runner never calls on_step.
        return self._maybe_eval(step)

    def on_training_end(self, trainer) -> None:
        if self._evaluator is None:
            return
        try:
            if self.final_eval:
                metrics = self._evaluator.finalize(
                    trainer.agent,
                    trainer.step,
                    logger=trainer.logger,
                    total_steps=trainer.total_steps,
                )
                self._after_eval(trainer.step, metrics)
        finally:
            self._evaluator.close()
