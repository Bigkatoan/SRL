"""Agent contract: :class:`BaseAgent`, :class:`OnPolicyAgent`, :class:`OffPolicyAgent`.

Every SRL algorithm is a :class:`BaseAgent`. The public, numpy-level API is::

    agent = SAC.from_config("configs/envs/pendulum_sac.yaml", env=gym.make("Pendulum-v1"))
    agent.learn(20_000)
    action, state = agent.predict(obs, deterministic=True)   # numpy in, env units out
    agent.save("runs/pendulum")                              # a directory
    agent = SAC.load("runs/pendulum", env=...)

Internals (trainers, the async runner, the live viewer, ROS2) use the
low-level tensor method :meth:`BaseAgent.act`, which takes an
encoder-keyed ``dict[str, Tensor]`` and returns
``(action, log_prob, value, hidden)`` in the model's normalised action
domain.

Subclasses implement :meth:`act`, :meth:`update`, :meth:`_algo_payload` and
:meth:`_load_algo_payload`; :class:`OnPolicyAgent` / :class:`OffPolicyAgent`
hold the code shared by each family (rollout/replay buffer construction,
target networks + Polyak/BatchNorm sync, train/eval mode locking, optimizer
construction, env binding) and wire :meth:`learn` to the matching trainer in
:mod:`srl.training`.
"""

from __future__ import annotations

import abc
import copy
import dataclasses
import threading
import warnings
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn

__all__ = ["BaseAgent", "OffPolicyAgent", "OnPolicyAgent"]


class _hybridmethod:
    """``Cls.load(path, ...)`` builds a new agent; ``agent.load(path)`` restores in place."""

    def __init__(self, fclass, finstance) -> None:
        self.fclass = fclass
        self.finstance = finstance
        self.__doc__ = fclass.__doc__

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.fclass.__get__(None, objtype)
        return self.finstance.__get__(obj, objtype)


def _is_tensor_obs(observation: Any) -> bool:
    if isinstance(observation, torch.Tensor):
        return True
    if isinstance(observation, dict) and observation:
        return any(isinstance(v, torch.Tensor) for v in observation.values())
    return False


def _jsonable(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _jsonable(v) for k, v in value.items())
    return False


class BaseAgent(abc.ABC):
    """Abstract base for all SRL agents."""

    #: Registry key (``"sac"``, ``"ppo"``, ...).
    algo_name: ClassVar[str] = ""
    #: Default config dataclass.
    config_cls: ClassVar[type | None] = None

    model: nn.Module
    cfg: Any
    _device: torch.device
    #: Total env steps this agent has been trained for (updated by ``learn``).
    num_timesteps: int = 0
    #: The bound (SRL-interface) training env, if any.
    env: Any = None
    #: The YAML/dict the model was built from (enables ``load`` without a model).
    _model_spec: dict | None = None
    #: Affine map from the model's [-1, 1] action domain to env units.
    _env_action_rescale: dict | None = None
    _obs_key: str = "state"
    _obs_shape: tuple | None = None
    #: ``env_type`` of the YAML the agent was built from (isaaclab/mjlab/goal/...).
    _env_type: str | None = None
    #: The resolved :class:`~srl.config.ExperimentConfig` (``from_config`` /
    #: ``srl-train`` agents); saved as ``config.yaml`` by :meth:`save`.
    _experiment: Any = None

    # ------------------------------------------------------------------
    # Abstract low-level interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def act(
        self,
        obs: dict[str, torch.Tensor],
        hidden: Any = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, Any]:
        """Tensor action selection: ``(action, log_prob, value, new_hidden)``.

        *obs* is a dict of batched tensors keyed by encoder name (already
        remapped); the action is in the model's normalised domain.
        """

    @abc.abstractmethod
    def update(self) -> dict[str, float]:
        """One gradient update from the agent's buffer; returns metrics."""

    @abc.abstractmethod
    def learn(
        self,
        total_timesteps: int,
        env: Any = None,
        eval_env: Any = None,
        callbacks: Any = None,
        log_dir: str | None = None,
        progress: bool = False,
        **kwargs: Any,
    ) -> BaseAgent:
        """Train for *total_timesteps* env steps; returns ``self``."""

    @abc.abstractmethod
    def _algo_payload(self) -> dict[str, Any]:
        """Algorithm state (weights, optimizers, buffers, counters)."""

    @abc.abstractmethod
    def _load_algo_payload(self, payload: dict[str, Any]) -> None:
        """Inverse of :meth:`_algo_payload`."""

    # ------------------------------------------------------------------
    # Properties / small utilities
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    def _as_tensor(self, arr: np.ndarray | Any):
        """Convert numpy array to a tensor on this agent's device."""
        if isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).float().to(self._device)
        return arr.to(self._device)

    # ------------------------------------------------------------------
    # Env binding
    # ------------------------------------------------------------------

    def set_env(self, env: Any) -> None:
        """Bind a training env (raw Gymnasium envs are wrapped automatically)."""
        from srl.training.envs import wrap_env

        env = wrap_env(env, device=str(self.device), env_type=self._env_type)
        self._attach_env(env)
        self._on_env_changed()

    def get_env(self) -> Any:
        return self.env

    def _attach_env(self, env: Any) -> None:
        """Record *env* and everything ``predict()`` needs to know about it."""
        self.env = env
        if env is None:
            return
        self._obs_key = getattr(env, "obs_key", self._obs_key)
        obs_space = getattr(env, "single_obs_space", None) or getattr(env, "obs_space", None)
        shape = getattr(obs_space, "shape", None)
        if shape is not None and not getattr(obs_space, "spaces", None):
            self._obs_shape = tuple(shape)
        rescale = _find_rescale_wrapper(env)
        if rescale is not None:
            self._env_action_rescale = {
                "low": np.asarray(rescale._env_low, dtype=np.float64),
                "high": np.asarray(rescale._env_high, dtype=np.float64),
                "finite": np.asarray(rescale._finite, dtype=bool),
            }
        else:
            self._env_action_rescale = None

    def _on_env_changed(self) -> None:  # noqa: B027 -- optional hook
        """Called by :meth:`set_env` after construction (rebuild env-sized state)."""

    def _resolve_learn_env(self, env: Any) -> Any:
        if env is not None:
            self.set_env(env)
        if self.env is None:
            raise ValueError(
                f"{type(self).__name__}.learn() needs an environment: pass env=... here, "
                "to the constructor, or to from_config()."
            )
        return self.env

    # ------------------------------------------------------------------
    # predict(): numpy in, env units out
    # ------------------------------------------------------------------

    def predict(
        self,
        observation: np.ndarray | dict,
        state: Any = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        """Return ``(action, next_state)`` for a raw observation.

        *observation* is what the env returns: a numpy array (one observation
        or a batch), or a dict of arrays (SRL wrapper / multi-modal envs). The
        action is a numpy array in the env's own units (un-normalised from
        the model's ``[-1, 1]`` domain when the training env rescaled
        actions), batched iff the observation was. *state* carries recurrent
        hidden states; ``None`` for feed-forward policies.

        .. deprecated::
            Calling ``predict`` with a dict of torch tensors (the old
            low-level signature returning a 4-tuple) still works but warns;
            use :meth:`act` instead.
        """
        if _is_tensor_obs(observation):
            warnings.warn(
                f"{type(self).__name__}.predict(tensor_obs) returning "
                "(action, log_prob, value, hidden) is deprecated; use .act() for the "
                "tensor API. predict() now takes numpy observations and returns "
                "(action, state).",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.act(observation, state, deterministic=deterministic)

        obs_t, batched = self._observation_to_tensors(observation)
        action, _, _, new_state = self.act(obs_t, state, deterministic=deterministic)
        action_np = action.detach().cpu().numpy()
        clip = getattr(self, "clip_action", None)
        if clip is not None:
            action_np = clip(action_np)
        action_np = self._to_env_action(action_np)
        if not batched:
            action_np = action_np[0]
        return action_np, new_state

    def _observation_to_tensors(self, observation: Any) -> tuple[dict, bool]:
        from srl.envs._image_obs import maybe_hwc_to_chw
        from srl.training._obs import encoder_layout, remap_obs_to_encoders

        if isinstance(observation, dict):
            obs = {k: np.asarray(v) for k, v in observation.items()}
            batched = None
        else:
            arr = np.asarray(observation)
            batched = (
                arr.ndim == len(self._obs_shape) + 1
                if self._obs_shape is not None
                else (arr.ndim % 2 == 0 and arr.ndim > 0)
            )
            # Same transform GymnasiumWrapper applies to pixel observations.
            obs = {self._obs_key: maybe_hwc_to_chw(arr)}
        names, input_names = encoder_layout(self)
        obs = remap_obs_to_encoders(obs, names, encoder_input_names=input_names)
        if batched is None:
            first = next(iter(obs.values()))
            batched = first.ndim > 0 and first.ndim % 2 == 0
        tensors = {}
        for key, value in obs.items():
            arr = np.asarray(value, dtype=np.float32)
            if not batched:
                arr = arr[None]
            tensors[key] = torch.from_numpy(arr).to(self.device)
        return tensors, bool(batched)

    def _to_env_action(self, action: np.ndarray) -> np.ndarray:
        info = self._env_action_rescale
        if info is None:
            return action
        a = np.asarray(action, dtype=np.float64)
        low, high, finite = info["low"], info["high"], info["finite"]
        scaled = low + (np.clip(a, -1.0, 1.0) + 1.0) * 0.5 * (high - low)
        return np.where(finite, scaled, a).astype(np.float32)

    # ------------------------------------------------------------------
    # State / persistence
    # ------------------------------------------------------------------

    def _agent_meta(self) -> dict[str, Any]:
        cfg_dict: dict[str, Any] = {}
        if dataclasses.is_dataclass(self.cfg):
            for f in dataclasses.fields(self.cfg):
                value = getattr(self.cfg, f.name)
                if _jsonable(value):
                    cfg_dict[f.name] = value
        return {
            "algo": self.algo_name,
            "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config_class": type(self.cfg).__name__,
            "algo_config": cfg_dict,
            "model_spec": copy.deepcopy(self._model_spec),
            "env_action_rescale": copy.deepcopy(self._env_action_rescale),
            "obs_key": self._obs_key,
            "obs_shape": self._obs_shape,
            "num_timesteps": int(self.num_timesteps),
        }

    def _load_agent_meta(self, meta: dict[str, Any] | None) -> None:
        if not meta:
            return
        if meta.get("model_spec") is not None and self._model_spec is None:
            self._model_spec = meta["model_spec"]
        if meta.get("env_action_rescale") is not None:
            self._env_action_rescale = meta["env_action_rescale"]
        self._obs_key = meta.get("obs_key", self._obs_key)
        shape = meta.get("obs_shape")
        self._obs_shape = tuple(shape) if shape is not None else self._obs_shape
        self.num_timesteps = int(meta.get("num_timesteps", self.num_timesteps))

    def checkpoint_payload(self) -> dict[str, Any]:
        """Everything needed to restore this agent (what checkpoints store)."""
        payload = dict(self._algo_payload())
        payload["srl_agent"] = self._agent_meta()
        return payload

    def load_checkpoint_payload(self, payload: dict[str, Any]) -> None:
        self._load_algo_payload(payload)
        self._load_agent_meta(payload.get("srl_agent"))

    def state_dict(self) -> dict[str, Any]:
        """Alias of :meth:`checkpoint_payload`."""
        return self.checkpoint_payload()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Alias of :meth:`load_checkpoint_payload`."""
        self.load_checkpoint_payload(state)

    def save(self, path: str | Path, *, include_replay_buffer: bool = False) -> Path:
        """Save the agent to directory *path* (see :mod:`srl.core.agent_io`).

        The replay buffer is left out unless *include_replay_buffer* (it can
        be large; ``srl-train`` checkpoints always include it for resume).
        """
        from srl.core.agent_io import save_agent

        return save_agent(self, path, include_replay_buffer=include_replay_buffer)

    def _load_classmethod(
        cls, path: str | Path, env: Any = None, device: str | torch.device | None = None, **kwargs
    ) -> BaseAgent:
        """Load an agent saved with :meth:`save` (or an ``srl-train`` checkpoint).

        ``Cls.load(path, env=None, device=None)`` builds a new agent;
        ``agent.load(path)`` restores *agent* in place and returns it.
        """
        from srl.core.agent_io import load_agent

        return load_agent(path, cls=cls, env=env, device=device, **kwargs)

    def _load_inplace(self, path: str | Path) -> BaseAgent:
        from srl.core.agent_io import read_payload

        self.load_checkpoint_payload(read_payload(path, map_location=self.device))
        return self

    load = _hybridmethod(classmethod(_load_classmethod), _load_inplace)

    @classmethod
    def from_config(
        cls,
        config: str | Path | dict,
        env: Any = None,
        device: str | torch.device = "cpu",
        **overrides: Any,
    ) -> BaseAgent:
        """Build an agent from an experiment YAML (path or dict).

        The YAML is strictly validated (:func:`srl.config.load_experiment`;
        unknown keys raise :class:`~srl.config.ConfigError`). The env is
        built first, so ``input_dim: auto`` / ``action_dim: auto`` are
        resolved from its spaces; the model comes from
        :class:`~srl.registry.builder.ModelBuilder`, the algorithm config
        from the YAML ``train:`` block (*overrides* win and are validated).
        *env* may be a raw Gymnasium env (wrapped with SRL's wrapper and
        :class:`~srl.envs.RescaleActionWrapper` automatically), an SRL env,
        an env id string, or ``None`` to build the YAML's ``env_id``.
        Called on :class:`BaseAgent` itself, the class is chosen from the
        YAML's ``algo:`` key.
        """
        from srl.core.agent_io import agent_from_config

        return agent_from_config(cls, config, env=env, device=device, **overrides)

    # ------------------------------------------------------------------
    # Shared learn() plumbing
    # ------------------------------------------------------------------

    def _build_learn_setup(
        self,
        total_timesteps: int,
        *,
        eval_env: Any,
        callbacks: Any,
        log_dir: str | None,
        progress: bool,
        eval_freq: int | None,
        n_eval_episodes: int,
        seed: int | None,
        logger: Any,
        evaluator: Any,
        reset_num_timesteps: bool,
    ) -> dict[str, Any]:
        from srl.training.envs import wrap_env
        from srl.training.evaluation import Evaluator
        from srl.training.loggers import make_logger

        start = 0 if reset_num_timesteps else int(self.num_timesteps)
        if logger is None:
            logger = make_logger(log_dir, progress=progress)
            if hasattr(logger, "configure_env"):
                logger.configure_env(getattr(self.env, "num_envs", 1))
        if evaluator is None and eval_env is not None:
            evaluator = Evaluator(
                wrap_env(eval_env, device=str(self.device), env_type=self._env_type),
                eval_freq=int(eval_freq if eval_freq is not None else 10_000),
                n_episodes=n_eval_episodes,
                seed=(seed or 0) + 10_000,
                env_type=self._env_type,
                verbose=progress,
            )
        return {
            "total_steps": start + int(total_timesteps),
            "start_step": start,
            "seed": seed,
            "logger": logger,
            "callbacks": callbacks,
            "evaluator": evaluator,
        }


def _find_rescale_wrapper(env: Any):
    from srl.envs.action_rescale import RescaleActionWrapper

    probe, seen = env, 0
    if getattr(env, "envs", None):  # SyncVectorEnv: sub-envs share one space
        probe = env.envs[0]
    while probe is not None and seen < 16:
        if isinstance(probe, RescaleActionWrapper):
            return probe
        probe = probe.__dict__.get("env") if hasattr(probe, "__dict__") else None
        seen += 1
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Family base classes
# ──────────────────────────────────────────────────────────────────────────────

_LEARN_DOC = """Train for *total_timesteps* env steps and return ``self``.

        Parameters
        ----------
        total_timesteps:
            Env steps to collect in this call (on top of :attr:`num_timesteps`
            when ``reset_num_timesteps=False``).
        env:
            Training env (raw Gymnasium envs are wrapped); defaults to the one
            given to the constructor / ``from_config``.
        eval_env:
            Optional eval env, evaluated every *eval_freq* steps
            (default 10_000) for *n_eval_episodes* deterministic episodes and
            once more at the end; results reach callbacks' ``on_eval``.
        callbacks:
            A callback or list of callbacks (:mod:`srl.training.callbacks`).
        log_dir:
            Write TensorBoard/JSONL logs here (nothing is written when None).
        progress:
            Print progress and eval lines to stdout.
        seed:
            Seed for the first ``env.reset``.
        logger / evaluator:
            Advanced: inject a pre-built logger / :class:`~srl.training.Evaluator`
            (what ``srl-train`` does).
        """


class OnPolicyAgent(BaseAgent):
    """Shared code of the on-policy Gaussian agents (PPO, A2C, A3C).

    Action bounds: a ``gaussian`` actor samples from an unbounded Normal; the
    sample is clipped to the action bounds only at the env boundary (see
    :mod:`srl.algorithms._onpolicy_common`).
    """

    _action_low: torch.Tensor | None = None
    _action_high: torch.Tensor | None = None

    def __init__(
        self,
        model: nn.Module,
        config: Any = None,
        device: str | torch.device = "cpu",
        *,
        env: Any = None,
    ) -> None:
        from srl.training.envs import wrap_env

        self.model = model
        self.cfg = config if config is not None else self.config_cls()
        self._device = torch.device(device)
        self.model.to(self._device)
        self._global_step = 0
        self.num_timesteps = 0
        env = wrap_env(env, device=str(self._device))
        if env is not None:
            self._fit_config_to_env(env)
            self._attach_env(env)

    # -- env ----------------------------------------------------------
    def _fit_config_to_env(self, env: Any) -> None:
        n = int(getattr(env, "num_envs", 1))
        if hasattr(self.cfg, "num_envs") and int(self.cfg.num_envs) != n:
            self.cfg = dataclasses.replace(self.cfg, num_envs=n)

    def _attach_env(self, env: Any) -> None:
        super()._attach_env(env)
        if env is not None:
            self.set_action_bounds_from_env(env)

    def _on_env_changed(self) -> None:
        if self.env is None or not hasattr(self.cfg, "num_envs"):
            return
        n = int(getattr(self.env, "num_envs", 1))
        if int(self.cfg.num_envs) != n:
            self.cfg = dataclasses.replace(self.cfg, num_envs=n)
            if getattr(self, "buffer", None) is not None:
                self.buffer = self._build_rollout_buffer()

    def _build_rollout_buffer(self):
        from srl.core.rollout_buffer import RolloutBuffer

        return RolloutBuffer(
            capacity=self.cfg.n_steps,
            num_envs=getattr(self.cfg, "num_envs", 1),
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
            device=self.device,
        )

    # -- acting -------------------------------------------------------
    def _preprocess_obs(self, obs: dict) -> dict:
        return obs

    def act(self, obs, hidden=None, deterministic=False):
        from srl.algorithms._onpolicy_common import select_action

        self.model.eval()
        obs = self._preprocess_obs(obs)
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden)
        # Stochastic: the raw (unclipped) sample -- callers store it for the
        # log-prob and clip with self.clip_action() only at env.step().
        # Deterministic: the distribution mode clipped to the action bounds.
        action, log_prob = select_action(result["actor_out"], deterministic)
        if deterministic:
            action = self.clip_action(action)
        return action, log_prob, result["value"], result["new_hidden"]

    # -- action bounds (see srl.algorithms._onpolicy_common) -------------
    def set_action_bounds(self, low: Any, high: Any) -> None:
        """Set the action range that env-bound actions are clipped to."""
        if low is None or high is None:
            self._action_low = self._action_high = None
            return
        self._action_low = torch.as_tensor(np.asarray(low, dtype=np.float32), device=self.device)
        self._action_high = torch.as_tensor(np.asarray(high, dtype=np.float32), device=self.device)

    def set_action_bounds_from_env(self, env: Any) -> bool:
        from srl.algorithms._onpolicy_common import env_action_bounds

        bounds = env_action_bounds(env)
        if bounds is None:
            return False
        self.set_action_bounds(*bounds)
        return True

    @property
    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._action_low is None or self._action_high is None:
            return None
        return self._action_low, self._action_high

    def clip_action(self, action: Any) -> Any:
        """Clip an action (tensor or ndarray) to the configured bounds (no-op without)."""
        if self._action_low is None or self._action_high is None or action is None:
            return action
        if isinstance(action, torch.Tensor):
            low = self._action_low.to(action.device, action.dtype)
            high = self._action_high.to(action.device, action.dtype)
            return torch.max(torch.min(action, high), low)
        return np.clip(action, self._action_low.cpu().numpy(), self._action_high.cpu().numpy())

    def _action_bounds_payload(self) -> dict[str, Any] | None:
        if self.action_bounds is None:
            return None
        return {
            "low": self._action_low.detach().cpu().numpy(),
            "high": self._action_high.detach().cpu().numpy(),
        }

    def _load_action_bounds_payload(self, payload: dict[str, Any] | None) -> None:
        if payload is not None:
            self.set_action_bounds(payload.get("low"), payload.get("high"))

    # -- payload helpers ------------------------------------------------
    def _base_payload(self) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "algo_step": self._global_step,
            "action_bounds": self._action_bounds_payload(),
        }

    def _load_base_payload(self, payload: dict[str, Any]) -> None:
        model_state = payload.get("model_state", payload.get("model"))
        if model_state is not None:
            self.model.load_state_dict(model_state)
        self._global_step = int(payload.get("algo_step", payload.get("step", 0)))
        self._load_action_bounds_payload(payload.get("action_bounds"))

    # -- learn ----------------------------------------------------------
    def learn(
        self,
        total_timesteps: int,
        env: Any = None,
        eval_env: Any = None,
        callbacks: Any = None,
        log_dir: str | None = None,
        progress: bool = False,
        *,
        eval_freq: int | None = None,
        n_eval_episodes: int = 5,
        seed: int | None = None,
        reset_num_timesteps: bool = True,
        logger: Any = None,
        evaluator: Any = None,
        env_type: str | None = None,
    ) -> OnPolicyAgent:
        from srl.training.on_policy import OnPolicyTrainer

        env = self._resolve_learn_env(env)
        setup = self._build_learn_setup(
            total_timesteps,
            eval_env=eval_env,
            callbacks=callbacks,
            log_dir=log_dir,
            progress=progress,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
            logger=logger,
            evaluator=evaluator,
            reset_num_timesteps=reset_num_timesteps,
        )
        OnPolicyTrainer(self, env, env_type=env_type or self._env_type, **setup).run()
        return self

    learn.__doc__ = _LEARN_DOC


class OffPolicyAgent(BaseAgent):
    """Shared code of the off-policy actor-critic agents (SAC, TD3, DDPG).

    Owns the target network (frozen, permanently ``eval()``; Polyak-averaged
    parameters + hard-copied BatchNorm buffers), the model-mode lock that
    makes every forward set train/eval mode atomically (the async runner's
    collector thread calls :meth:`act` concurrently with :meth:`update`),
    the head/encoder optimizers (critic-reachable encoders train off the
    critic loss, actor-only encoders off the actor loss), the replay buffer
    and the optional observation normalizer.
    """

    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module | None = None,
        config: Any = None,
        device: str | torch.device = "cpu",
        *,
        env: Any = None,
    ) -> None:
        from srl.algorithms._common import build_encoder_optimizers, maybe_build_obs_normalizer
        from srl.training.envs import wrap_env

        if target_model is not None and not isinstance(target_model, nn.Module):
            # SAC(model, config[, device]) -- the target network is optional
            # now (deep-copied from *model*); shift the positional args.
            if isinstance(config, (str, torch.device)):
                device, config = config, None
            elif config is not None:
                raise TypeError("pass either target_model or config positionally, not both")
            target_model, config = None, target_model
        if target_model is None:
            target_model = copy.deepcopy(model)
        self.model = model
        self.target_model = target_model
        self.cfg = config if config is not None else self.config_cls()
        self._device = torch.device(device)
        self.num_timesteps = 0

        env = wrap_env(env, device=str(self._device))
        if env is not None:
            self._fit_config_to_env(env)

        self.model.to(self._device)
        self.target_model.to(self._device)
        # Copy weights to target, freeze. target_model is only ever touched
        # from inside update(), so (unlike self.model) there is no
        # cross-thread mode race: eval() once is correct for its whole life
        # -- a train()-mode target with BatchNorm would normalize with each
        # sampled batch's statistics and make the bootstrap target depend on
        # whatever next-states happened to be sampled.
        self.target_model.load_state_dict(self.model.state_dict())
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()
        self._has_batchnorm: bool = any(
            isinstance(m, nn.modules.batchnorm._BatchNorm) for m in self.model.modules()
        )
        self._model_lock = threading.Lock()

        # Head optimizers exclude encoders; encoders are split by
        # reachability from the critic (see _common._partition_encoder_params).
        encoder_lr = getattr(self.cfg, "encoder_lr", self.cfg.lr_critic)
        (
            self._encoder_param_list,
            self._actor_only_encoder_param_list,
            self._actor_only_encoder_modules,
            self.encoder_optimizer,
            self.actor_only_encoder_optimizer,
        ) = build_encoder_optimizers(self.model, encoder_lr)
        self.actor_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters()), lr=self.cfg.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.critic.parameters()), lr=self.cfg.lr_critic
        )
        self._encoder_update_counter: int = 0
        self.obs_normalizer = maybe_build_obs_normalizer(self.cfg)
        self._global_step = 0
        self._build_exploration()
        self.buffer = self._build_buffer()
        if env is not None:
            self._attach_env(env)

    # -- construction hooks -------------------------------------------
    def _build_buffer(self):
        from srl.core.prioritized_replay_buffer import build_replay_buffer

        return build_replay_buffer(self.cfg)

    def _build_exploration(self) -> None:  # noqa: B027 -- optional hook
        """Build action-dim-dependent exploration state (noise, entropy target)."""

    def _fit_config_to_env(self, env: Any) -> None:
        from srl.training.envs import env_action_dim

        changes: dict[str, Any] = {}
        dim = env_action_dim(env)
        if hasattr(self.cfg, "action_dim") and int(self.cfg.action_dim or 0) != dim:
            if self.cfg.action_dim:
                warnings.warn(
                    f"{type(self).__name__}: config action_dim={self.cfg.action_dim} does not "
                    f"match the env ({dim}); using the env's.",
                    stacklevel=3,
                )
            changes["action_dim"] = dim
        n = int(getattr(env, "num_envs", 1))
        if hasattr(self.cfg, "replay_num_envs") and int(self.cfg.replay_num_envs) != n:
            changes["replay_num_envs"] = n
        if changes:
            self.cfg = dataclasses.replace(self.cfg, **changes)

    def _on_env_changed(self) -> None:
        if self.env is None:
            return
        old = self.cfg
        self._fit_config_to_env(self.env)
        if self.cfg is old:
            return
        if old.action_dim != self.cfg.action_dim:
            self._build_exploration()
        if len(self.buffer) == 0:
            self.buffer = self._build_buffer()
        else:
            warnings.warn(
                f"{type(self).__name__}: env shape changed but the replay buffer is not "
                "empty; keeping the existing buffer.",
                stacklevel=3,
            )

    # -- model plumbing -------------------------------------------------
    def _forward_model(self, mode: str, *args, **kwargs):
        """``self.model(...)`` with train/eval mode set atomically under the lock.

        See :func:`srl.algorithms._common.forward_with_mode`: toggling mode as
        a separate statement races against :meth:`act` on the async runner's
        collector thread (a genuine concurrent write hazard for BatchNorm
        buffers), and guarantees each forward runs in its intended mode.
        """
        from srl.algorithms._common import forward_with_mode

        return forward_with_mode(self.model, self._model_lock, mode, *args, **kwargs)

    def _soft_update_target(self) -> None:
        """Polyak-average parameters into the target; hard-copy BN buffers.

        Buffers (BatchNorm running stats) are not parameters, so without the
        copy the frozen, eval-mode target would keep its step-0 statistics
        forever; they are copied rather than EMA'd again because BatchNorm's
        own momentum already is an EMA.
        """
        from srl.algorithms._common import _soft_update, _sync_target_buffers

        _soft_update(self.model, self.target_model, self.cfg.tau)
        if self._has_batchnorm:
            _sync_target_buffers(self.model, self.target_model)

    def _can_update(self) -> bool:
        # HERReplayBuffer is episode-indexed (its __len__ counts episodes);
        # it exposes can_sample() for exactly this reason.
        can_sample = getattr(self.buffer, "can_sample", None)
        if can_sample is not None:
            return bool(can_sample(self.cfg.batch_size))
        return len(self.buffer) >= self.cfg.batch_size

    def _sample_batch(self):
        """Sample, move to device and normalize a training batch.

        Returns ``(batch, obs, next_obs, actions, rewards, dones, discount,
        is_weights)``.
        """
        from srl.algorithms._common import batch_discounts, prepare_obs

        batch = self.buffer.sample(self.cfg.batch_size)
        obs = {k: v.to(self.device) for k, v in batch.obs.items()}
        next_obs = {k: v.to(self.device) for k, v in batch.next_obs.items()}
        obs = prepare_obs(self.model, self.obs_normalizer, obs, update_stats=True)
        next_obs = prepare_obs(self.model, self.obs_normalizer, next_obs)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        dones = batch.dones.to(self.device)
        discount = batch_discounts(batch, self.cfg.gamma, self.device)
        is_weights = batch.weights.to(self.device) if batch.weights is not None else None
        return batch, obs, next_obs, actions, rewards, dones, discount, is_weights

    # -- acting ---------------------------------------------------------
    def _select_action(self, actor_out: Any, deterministic: bool):
        """``(action, log_prob)`` from the actor head output (exploration noise here)."""
        action = actor_out.get("action") if isinstance(actor_out, dict) else actor_out
        return action, None

    def act(self, obs, hidden=None, deterministic=False):
        from srl.algorithms._common import prepare_obs

        with torch.no_grad():
            # compute_critic=False: the collection path never reads the value.
            obs = prepare_obs(self.model, self.obs_normalizer, obs)
            result = self._forward_model("eval", obs, hidden_states=hidden, compute_critic=False)
            action, log_prob = self._select_action(result["actor_out"], deterministic)
        return action, log_prob, None, result["new_hidden"]

    # -- payload helpers ------------------------------------------------
    def _base_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "replay_buffer_state": self.buffer.state_dict(),
            "algo_step": self._global_step,
            "encoder_update_counter": self._encoder_update_counter,
        }
        # Optional states: absent in older checkpoints, skipped on load.
        if self.encoder_optimizer is not None:
            payload["encoder_optimizer_state"] = self.encoder_optimizer.state_dict()
        if self.actor_only_encoder_optimizer is not None:
            payload["actor_only_encoder_optimizer_state"] = (
                self.actor_only_encoder_optimizer.state_dict()
            )
        if self.obs_normalizer is not None:
            payload["obs_normalizer_state"] = self.obs_normalizer.state_dict()
        return payload

    def _load_base_payload(self, payload: dict[str, Any]) -> None:
        model_state = payload.get("model_state", payload.get("model"))
        if model_state is not None:
            self.model.load_state_dict(model_state)
        target_state = payload.get("target_model_state", payload.get("target"))
        if target_state is not None:
            self.target_model.load_state_dict(target_state)
        for attr, key in (
            ("actor_optimizer", "actor_optimizer_state"),
            ("critic_optimizer", "critic_optimizer_state"),
            ("encoder_optimizer", "encoder_optimizer_state"),
            ("actor_only_encoder_optimizer", "actor_only_encoder_optimizer_state"),
        ):
            opt, state = getattr(self, attr, None), payload.get(key)
            if opt is not None and state is not None:
                opt.load_state_dict(state)
        buf = payload.get("replay_buffer_state")
        if buf is not None:
            self.buffer.load_state_dict(buf)
        self._global_step = int(payload.get("algo_step", payload.get("step", 0)))
        self._encoder_update_counter = int(payload.get("encoder_update_counter", 0))
        norm_state = payload.get("obs_normalizer_state")
        if self.obs_normalizer is not None and norm_state is not None:
            self.obs_normalizer.load_state_dict(norm_state)

    def _algo_payload(self) -> dict[str, Any]:
        return self._base_payload()

    def _load_algo_payload(self, payload: dict[str, Any]) -> None:
        self._load_base_payload(payload)

    # -- learn ----------------------------------------------------------
    def learn(
        self,
        total_timesteps: int,
        env: Any = None,
        eval_env: Any = None,
        callbacks: Any = None,
        log_dir: str | None = None,
        progress: bool = False,
        *,
        eval_freq: int | None = None,
        n_eval_episodes: int = 5,
        seed: int | None = None,
        reset_num_timesteps: bool = True,
        logger: Any = None,
        evaluator: Any = None,
        env_type: str | None = None,
        runner_cfg: Any = None,
    ) -> OffPolicyAgent:
        from srl.training.off_policy import OffPolicyTrainer

        env = self._resolve_learn_env(env)
        setup = self._build_learn_setup(
            total_timesteps,
            eval_env=eval_env,
            callbacks=callbacks,
            log_dir=log_dir,
            progress=progress,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            seed=seed,
            logger=logger,
            evaluator=evaluator,
            reset_num_timesteps=reset_num_timesteps,
        )
        OffPolicyTrainer(
            self, env, env_type=env_type or self._env_type, runner_cfg=runner_cfg, **setup
        ).run()
        return self

    learn.__doc__ = _LEARN_DOC.rstrip() + """
        runner_cfg:
            :class:`~srl.core.config.AsyncRunnerConfig` enabling the
            asynchronous / GPU-buffer runner.
        """
