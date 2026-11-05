"""Agent persistence (``agent.save`` / ``Cls.load``) and ``from_config``.

All on-disk format decisions live in this module so a new checkpoint format
can replace it without touching the algorithms.

Directory format ``srl-agent`` v0 (written by :func:`save_agent`)::

    <dir>/
      agent.pt     torch.save(agent.checkpoint_payload())  -- the source of truth
      meta.json    human-readable summary (format, algo, config, versions)
      model.yaml   the model spec the agent was built from (when known)
      model.pt     torch.save(model) -- only when no model spec is known
      config.yaml  srl.config.dump_resolved_config() of the experiment the agent
                   was built from (from_config / srl-train agents only)

``agent.checkpoint_payload()`` embeds an ``"srl_agent"`` metadata block
(algo, config class + JSON-able config fields, model spec, action rescale
map, obs layout), so a single-file ``srl-train`` checkpoint
(``ckpt_*.pt``/``final_*.pt``/``best_*.pt``) is loadable by
:func:`load_agent` as well.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "agent_from_config",
    "build_agent_from_experiment",
    "load_agent",
    "read_payload",
    "save_agent",
]

FORMAT_NAME = "srl-agent"
FORMAT_VERSION = 0
_PAYLOAD_FILE = "agent.pt"
_CONFIG_FILE = "config.yaml"


def _srl_version() -> str:
    try:
        from srl import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def save_agent(agent, path: str | Path, *, include_replay_buffer: bool = False) -> Path:
    """Write *agent* to directory *path*; returns the directory."""
    path = Path(path)
    payload = agent.checkpoint_payload()
    if not include_replay_buffer:
        payload.pop("replay_buffer_state", None)
    if path.suffix in (".pt", ".pth"):
        warnings.warn(
            "agent.save('<file>.pt') is deprecated; agents are saved to a directory now "
            "(agent.save('runs/my_agent')). Writing the single-file payload.",
            DeprecationWarning,
            stacklevel=3,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path

    path.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path / _PAYLOAD_FILE)
    meta = dict(payload.get("srl_agent") or {})
    model_spec = meta.pop("model_spec", None)
    meta.pop("env_action_rescale", None)
    summary = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "srl_version": _srl_version(),
        "includes_replay_buffer": bool(include_replay_buffer and "replay_buffer_state" in payload),
        **meta,
    }
    (path / "meta.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if model_spec is not None:
        import yaml

        (path / "model.yaml").write_text(yaml.safe_dump(model_spec, sort_keys=False))
        stale = path / "model.pt"
        if stale.exists():
            stale.unlink()
    else:
        torch.save(agent.model, path / "model.pt")
    _write_resolved_config(agent, path / _CONFIG_FILE)
    return path


def _write_resolved_config(agent, target: Path) -> None:
    """``config.yaml``: the fully resolved experiment the agent was built from.

    Written with :func:`srl.config.dump_resolved_config` (explicit dims, every
    hyperparameter including defaults, the agent's *current* algorithm
    config), so ``srl.config.load_experiment(<dir>/config.yaml)`` reproduces
    the run's configuration. Skipped for agents not built from a config.
    """
    import dataclasses

    exp = getattr(agent, "_experiment", None)
    if exp is None:
        return
    from srl.config import dump_resolved_config

    if dataclasses.is_dataclass(agent.cfg) and type(agent.cfg) is type(exp.algo_config):
        exp = dataclasses.replace(exp, algo_config=agent.cfg)
    dump_resolved_config(exp, target)


def read_payload(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """Read an agent payload from a :func:`save_agent` directory or a checkpoint file."""
    path = Path(path)
    if path.is_dir():
        path = path / _PAYLOAD_FILE
    if not path.exists():
        raise FileNotFoundError(f"No SRL agent found at {path}")
    if path.suffix == ".safetensors":
        meta_path = path.with_suffix(".meta.pt")
        raise ValueError(
            f"{path} is a bare-model safetensors checkpoint (no agent state); "
            f"load it with CheckpointManager.load(model, ...) instead (meta: {meta_path})."
        )
    return torch.load(path, map_location=map_location, weights_only=False)


def _resolve_device(device: Any) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_agent_class(cls, algo: str | None):
    """The agent class for *algo* via :data:`srl.registry.AlgorithmRegistry`.

    A concrete *cls* (``SAC.load``) wins but must match *algo* when one is
    recorded; ``BaseAgent`` (or ``None``) dispatches on *algo*.
    """
    from srl.core.base_agent import BaseAgent

    if cls is None or cls is BaseAgent or getattr(cls, "algo_name", "") == "":
        if not algo:
            raise ValueError(
                "Cannot tell which algorithm this is: no 'algo' recorded. "
                "Call the concrete class (e.g. SAC.load(...)) instead."
            )
        from srl.config.algos import algorithm_spec

        return algorithm_spec(algo).agent_class()
    if algo and cls.algo_name and algo != cls.algo_name:
        raise TypeError(
            f"{cls.__name__}: this agent/config is for {algo.upper()}; use "
            f"{algo.upper()} (or BaseAgent, which dispatches on it) instead."
        )
    return cls


def _build_config(agent_cls, meta: dict) -> Any:
    import dataclasses

    import srl.core.config as config_mod

    cfg_cls = getattr(config_mod, meta.get("config_class") or "", None) or agent_cls.config_cls
    known = {f.name for f in dataclasses.fields(cfg_cls)}
    values = {k: v for k, v in (meta.get("algo_config") or {}).items() if k in known}
    return cfg_cls(**values)


def load_agent(
    path: str | Path,
    *,
    cls=None,
    env: Any = None,
    device: Any = None,
    model: Any = None,
):
    """Rebuild an agent saved by :func:`save_agent` (or an ``srl-train`` checkpoint)."""
    from srl.training.envs import wrap_env

    device = _resolve_device(device)
    path = Path(path)
    payload = read_payload(path, map_location=device)
    meta = payload.get("srl_agent") or {}
    agent_cls = _resolve_agent_class(cls, meta.get("algo"))

    if model is None:
        spec = meta.get("model_spec")
        if spec is not None:
            from srl.registry.builder import ModelBuilder

            model = ModelBuilder.from_dict(spec)
        elif path.is_dir() and (path / "model.pt").exists():
            model = torch.load(path / "model.pt", map_location=device, weights_only=False)
        else:
            raise ValueError(
                f"{path}: no model spec stored with this agent; pass model=... to load()."
            )

    cfg = _build_config(agent_cls, meta)
    env = wrap_env(env, device=str(device))
    if env is not None and getattr(cfg, "use_her", False):
        from srl.training.algo_config import configure_her_from_env

        configure_her_from_env(cfg, env, "goal")
    agent = agent_cls(model, config=cfg, device=device, env=env)
    agent._model_spec = meta.get("model_spec")
    agent.load_checkpoint_payload(payload)
    if path.is_dir() and (path / _CONFIG_FILE).exists():
        agent._experiment = _read_resolved_config(path / _CONFIG_FILE)
    if env is not None:
        # The live env wins over the stored action map.
        agent._attach_env(env)
    return agent


def _read_resolved_config(path: Path) -> Any:
    """The saved ``config.yaml`` as an ExperimentConfig, or ``None`` if it no longer validates."""
    from srl.config import ConfigError, load_experiment

    try:
        return load_experiment(path)
    except (ConfigError, OSError) as exc:
        warnings.warn(f"{path}: could not re-read the saved config ({exc})", stacklevel=3)
        return None


def _load_raw_config(config: str | Path | dict) -> dict:
    if isinstance(config, dict):
        return dict(config)
    import yaml

    with open(config, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_agent_from_experiment(
    exp: Any,
    env: Any,
    *,
    cls: Any = None,
    device: Any = "cpu",
    n_envs: int | None = None,
    source: str | None = None,
):
    """Build the agent described by a validated :class:`~srl.config.ExperimentConfig`.

    The one construction path shared by :meth:`BaseAgent.from_config` and
    ``srl-train``. *env* (an SRL env, or ``None``) is built *before* the
    model, so ``input_dim: auto`` / ``input_shape: auto`` / ``action_dim:
    auto`` are filled from its per-env spaces (explicit values are
    cross-checked against them), and the env-derived algorithm fields
    (``num_envs``, ``action_dim``, ``replay_num_envs``, A3C ``n_workers``,
    HER's goal split) are set. The fully resolved experiment is kept on the
    agent (``agent._experiment``) and written as ``config.yaml`` by
    :func:`save_agent`.
    """
    import dataclasses

    from srl.config import ConfigError, resolve_auto_dims_from_env
    from srl.registry.builder import ModelBuilder
    from srl.training.algo_config import (
        configure_her_from_env,
        env_derived_overrides,
        validate_algo_model_compatibility,
    )

    if not exp.algo or exp.algo_config is None:
        where = source or exp.source or "<dict>"
        raise ConfigError(
            f"Config '{where}' does not declare an algorithm: add `algo: <ppo|sac|ddpg|"
            "td3|a2c|a3c>` to the YAML or pass --algo."
        )
    agent_cls = _resolve_agent_class(cls, exp.algo)
    error = validate_algo_model_compatibility(exp.model, exp.algo, source or exp.source or "<dict>")
    if error is not None:
        raise ConfigError(error)

    model_cfg = resolve_auto_dims_from_env(exp.model, env) if env is not None else exp.model
    model = ModelBuilder.from_config(model_cfg)

    overrides = env_derived_overrides(exp.algo, env, n_envs=n_envs, seed=int(exp.trainer.seed))
    known = {f.name for f in dataclasses.fields(exp.algo_config)}
    algo_cfg = dataclasses.replace(
        exp.algo_config, **{k: v for k, v in overrides.items() if k in known}
    )
    if env is not None:
        # HER needs the goal split and the env's sparse reward fn, neither of
        # which can come from YAML -- resolve them before the buffer is built.
        configure_her_from_env(algo_cfg, env, exp.env.env_type)

    device = _resolve_device(device)
    agent = agent_cls(model, config=algo_cfg, device=device, env=env)
    resolved = dataclasses.replace(exp, model=model_cfg, algo_config=algo_cfg)
    agent._experiment = resolved
    # Explicit dims: load() rebuilds the model from this without an env.
    agent._model_spec = _experiment_to_dict(resolved)
    agent._env_type = exp.env.env_type
    return agent


def _experiment_to_dict(exp: Any) -> dict:
    from srl.config import experiment_to_dict

    return experiment_to_dict(exp)


def agent_from_config(cls, config: str | Path | dict, *, env=None, device="cpu", **overrides):
    """Implementation of :meth:`BaseAgent.from_config`.

    ``config`` is strictly validated with :func:`srl.config.load_experiment`
    (*overrides* are merged into its ``train:`` block first, so they are
    validated too).
    """
    import copy

    from srl.config import ConfigError, infer_algo_name, load_experiment
    from srl.training.algo_config import validate_algo_model_compatibility
    from srl.training.envs import build_training_env, normalize_env_name, wrap_env

    raw = copy.deepcopy(_load_raw_config(config))
    source = None if isinstance(config, dict) else str(config)
    if overrides:
        train = dict(raw.get("train") or {})
        train.update(overrides)
        raw["train"] = train
    declared = infer_algo_name(raw, source)
    agent_cls = _resolve_agent_class(cls, declared)
    # Head/algorithm mismatch first: it explains the strict train: errors
    # (another algorithm's hyperparameters) that load_experiment would report.
    error = validate_algo_model_compatibility(raw, agent_cls.algo_name, source or "<dict>")
    if error is not None:
        raise ConfigError(error)
    exp = load_experiment(raw, algo=agent_cls.algo_name)
    if source is not None:
        exp.source = source

    device = _resolve_device(device)
    env_type = exp.env.env_type
    n_envs = int(exp.trainer.n_envs)
    if env is None and exp.env.env_id:
        env_name = normalize_env_name(exp.env.env_id, env_type)
        env = build_training_env(
            env_name,
            device=str(device),
            n_envs=1 if agent_cls.algo_name == "a3c" else n_envs,
            env_type=env_type,
            seed=int(exp.trainer.seed),
        )
    else:
        env = wrap_env(env, device=str(device), env_type=env_type)
    return build_agent_from_experiment(
        exp,
        env,
        cls=agent_cls,
        device=device,
        n_envs=n_envs if agent_cls.algo_name == "a3c" else None,
        source=source,
    )
