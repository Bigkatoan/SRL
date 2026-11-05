"""Shared trainer scaffolding (callbacks, evaluation, stop handling)."""

from __future__ import annotations

from typing import Any

from srl.training.callbacks import CallbackList
from srl.training.envs import is_batched_env
from srl.training.loggers import NullLogger


class Trainer:
    """Base class of :class:`OnPolicyTrainer` / :class:`OffPolicyTrainer`.

    Parameters
    ----------
    agent:
        The :class:`~srl.core.base_agent.BaseAgent` to train (anything with
        ``act()``/``update()``/``buffer``/``cfg``/``model`` works).
    env:
        SRL-interface training env (see :func:`srl.training.envs.wrap_env`).
    total_steps:
        Absolute env-step budget: training stops once ``step >= total_steps``.
    start_step:
        Step counter at the start (``> 0`` when resuming).
    seed:
        Seed for the first ``env.reset(seed=...)``.
    logger:
        :class:`~srl.utils.logger.Logger`-like object (``update_episodes``,
        ``set_step``, ``record_metrics``); defaults to a no-op logger.
    callbacks:
        Callbacks (see :mod:`srl.training.callbacks`).
    evaluator:
        Optional :class:`~srl.training.evaluation.Evaluator` run on its own
        schedule plus once at the end of training.
    env_type / env_name:
        Only used to decide whether *env* is batched (isaaclab/mjlab envs are
        batched even at ``num_envs=1``); inferred from *env* when omitted.
    """

    def __init__(
        self,
        agent: Any,
        env: Any,
        *,
        total_steps: int,
        start_step: int = 0,
        seed: int | None = 0,
        logger: Any = None,
        callbacks: Any = None,
        evaluator: Any = None,
        env_type: str | None = None,
        env_name: str | None = None,
    ) -> None:
        self.agent = agent
        self.env = env
        self.total_steps = int(total_steps)
        self.start_step = int(start_step)
        self.step = int(start_step)
        self.seed = seed
        self.logger = logger if logger is not None else NullLogger()
        self.callbacks = (
            callbacks if isinstance(callbacks, CallbackList) else CallbackList(callbacks)
        )
        self.evaluator = evaluator
        self.env_type = env_type
        self.env_name = env_name
        self.stop_training = False
        if evaluator is not None:
            evaluator.reset_schedule(self.start_step)

    # ------------------------------------------------------------------
    @property
    def vectorized_env(self) -> bool:
        return is_batched_env(self.env, self.env_type, self.env_name)

    def _continue(self, keep_going: bool | None) -> bool:
        if keep_going is False:
            self.stop_training = True
        return not self.stop_training

    def _sync_agent_step(self, step: int) -> None:
        try:
            self.agent.num_timesteps = int(step)
        except Exception:  # pragma: no cover - duck-typed agents
            pass

    def _record_update(self, step: int, metrics: dict) -> bool:
        # Keep the agent's counter current so checkpoints saved by callbacks
        # record the step they were taken at.
        self._sync_agent_step(step)
        self.logger.set_step(step)
        self.logger.record_metrics(metrics, step=step, total_steps=self.total_steps)
        return self._continue(self.callbacks.on_update(step, metrics))

    def _maybe_evaluate(self, step: int) -> bool:
        if self.evaluator is None:
            return True
        metrics = self.evaluator.maybe_evaluate(
            self.agent, step, logger=self.logger, total_steps=self.total_steps
        )
        if metrics is None:
            return True
        return self._continue(self.callbacks.on_eval(step, metrics))

    def _final_evaluate(self, step: int) -> None:
        if self.evaluator is None:
            return
        metrics = self.evaluator.finalize(
            self.agent, step, logger=self.logger, total_steps=self.total_steps
        )
        if metrics is not None:
            self.callbacks.on_eval(step, metrics)

    def _close_evaluator(self) -> None:
        if self.evaluator is not None:
            self.evaluator.close()

    # ------------------------------------------------------------------
    def run(self) -> int:
        """Train until ``total_steps`` (or a callback stops it); return the final step."""
        self.stop_training = False
        self._continue(self.callbacks.on_training_start(self))
        try:
            self._train()
        finally:
            self._sync_agent_step(self.step)
            self.callbacks.on_training_end(self)
        return self.step

    def _train(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError
