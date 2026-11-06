"""srl.training — trainers, evaluation, callbacks and env construction.

The training loops behind :meth:`srl.core.base_agent.BaseAgent.learn` and
``srl-train``:

* :class:`OnPolicyTrainer` (PPO, A2C) and :class:`OffPolicyTrainer` (SAC,
  TD3, DDPG; sync or :class:`~srl.runners.AsyncOffPolicyRunner`).
* :class:`Evaluator` / :func:`evaluate_agent` -- deterministic evaluation,
  eval scheduling and best-checkpoint tracking.
* Callbacks: :class:`Callback`, :class:`CheckpointCallback`,
  :class:`EvalCallback`, :class:`EarlyStopping`, :class:`LogCallback`.
* Env helpers: :func:`make_env`, :func:`wrap_env`, :func:`build_training_env`.
"""

from srl.training.algo_config import ConfigurationError, env_derived_overrides
from srl.training.base import Trainer
from srl.training.callbacks import (
    BaseCallback,
    Callback,
    CallbackList,
    CheckpointCallback,
    EarlyStopping,
    EvalCallback,
    LogCallback,
)
from srl.training.envs import build_training_env, make_env, wrap_env
from srl.training.evaluation import Evaluator, evaluate_agent
from srl.training.loggers import NullLogger
from srl.training.off_policy import OffPolicyTrainer
from srl.training.on_policy import OnPolicyTrainer

__all__ = [
    "BaseCallback",
    "Callback",
    "CallbackList",
    "CheckpointCallback",
    "ConfigurationError",
    "EarlyStopping",
    "EvalCallback",
    "Evaluator",
    "LogCallback",
    "NullLogger",
    "OffPolicyTrainer",
    "OnPolicyTrainer",
    "Trainer",
    "build_training_env",
    "env_derived_overrides",
    "evaluate_agent",
    "make_env",
    "wrap_env",
]
