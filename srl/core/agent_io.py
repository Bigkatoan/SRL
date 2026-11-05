"""Agent persistence (``agent.save`` / ``Cls.load``) and ``from_config``.

All on-disk format decisions live in this module so a new checkpoint format
can replace it without touching the algorithms.

Directory format ``srl-agent`` v0 (written by :func:`save_agent`)::

    <dir>/
      agent.pt     torch.save(agent.checkpoint_payload())  -- the source of truth
      meta.json    human-readable summary (format, algo, config, versions)
      model.yaml   the model spec the agent was built from (when known)
      model.pt     torch.save(model) -- only when no model spec is known

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
    "load_agent",
    "read_payload",
    "save_agent",
]

FORMAT_NAME = "srl-agent"
FORMAT_VERSION = 0
_PAYLOAD_FILE = "agent.pt"


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
    return path


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
    from srl.algorithms import get_algorithm
    from srl.core.base_agent import BaseAgent

    if cls is None or cls is BaseAgent or getattr(cls, "algo_name", "") == "":
        if not algo:
            raise ValueError(
                "Cannot tell which algorithm this is: no 'algo' recorded. "
                "Call the concrete class (e.g. SAC.load(...)) instead."
            )
        return get_algorithm(algo)[0]
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
    if env is not None:
        # The live env wins over the stored action map.
        agent._attach_env(env)
    return agent


def _load_raw_config(config: str | Path | dict) -> dict:
    if isinstance(config, dict):
        return dict(config)
    import yaml

    with open(config, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def agent_from_config(cls, config: str | Path | dict, *, env=None, device="cpu", **overrides):
    """Implementation of :meth:`BaseAgent.from_config`."""
    from srl.registry.builder import ModelBuilder
    from srl.training.algo_config import (
        ConfigurationError,
        make_algo_config,
        validate_algo_model_compatibility,
    )
    from srl.training.envs import build_training_env, resolve_env_spec, resolve_env_type, wrap_env

    raw = _load_raw_config(config)
    declared = str(raw.get("algo") or "").strip().lower() or None
    agent_cls = _resolve_agent_class(cls, declared)
    algo = agent_cls.algo_name
    source = str(config) if not isinstance(config, dict) else "<dict>"
    error = validate_algo_model_compatibility(raw, algo, source)
    if error is not None:
        raise ConfigurationError(error)

    device = _resolve_device(device)
    env_type = resolve_env_type(raw)
    if env is None and (raw.get("env_id") or (raw.get("train") or {}).get("env_id")):
        env_name, env_type = resolve_env_spec(None, raw)
        n_envs = int((raw.get("train") or {}).get("n_envs", 1)) if algo != "a3c" else 1
        env = build_training_env(env_name, device=str(device), n_envs=n_envs, env_type=env_type)
    else:
        env = wrap_env(env, device=str(device), env_type=env_type)

    # The env is built before the model on purpose (env-derived model dims).
    model = ModelBuilder.from_dict(raw)
    n_envs = None if algo != "a3c" else int((raw.get("train") or {}).get("n_envs", 1))
    algo_cfg = make_algo_config(
        algo, raw, env=env, env_type=env_type, n_envs=n_envs, overrides=overrides
    )
    agent = agent_cls(model, config=algo_cfg, device=device, env=env)
    agent._model_spec = raw
    agent._env_type = env_type
    return agent
