"""Agent persistence (``agent.save`` / ``Cls.load`` / ``srl.load_policy``) and ``from_config``.

All on-disk format decisions live in this module so a new checkpoint format
can replace it without touching the algorithms.

Directory format ``srl-agent`` v1 (written by :func:`save_agent`)::

    <dir>/
      meta.json           format, format_version, versions, algo, obs/action spec,
                          normalizers present, counters (written last)
      config.yaml         resolved experiment config -- the single source of truth
                          for the model and the algorithm config; re-validated with
                          srl.config.load_experiment() on load
      model.yaml          (instead of config.yaml) the model spec, for agents not
                          built from an experiment config
      model.pt            (instead of either) the pickled model -- custom models only
      model.safetensors   model weights (inference; strict key check on load)
      normalizers.json    observation / reward running statistics
      train_state.pt      optimizers, target network, RNG, counters, alpha, ...
                          (optional; loaded with torch.load(weights_only=True))
      replay_buffer/      optional (include_replay_buffer=True)
        state.pt          also weights_only=True

Everything except ``model.pt`` loads without pickle. Older checkpoints still
load: the v0 directory format (``agent.pt`` + ``meta.json`` + ``model.yaml``)
and single-file ``srl-train`` checkpoints (``ckpt_*.pt`` / ``final_*.pt`` /
``best_*.pt``) are read with ``torch.load(weights_only=False)`` after a
:class:`~srl.utils.safe_serialization.LegacyCheckpointWarning`.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from srl.utils.safe_serialization import (
    LegacyCheckpointWarning,
    from_jsonable,
    load_trusted_or_legacy,
    load_weights_only,
    pack,
    to_jsonable,
)

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "Checkpoint",
    "CheckpointFormatError",
    "agent_from_config",
    "build_agent_from_experiment",
    "find_checkpoint",
    "is_agent_dir",
    "load_agent",
    "read_checkpoint",
    "read_payload",
    "save_agent",
]

FORMAT_NAME = "srl-agent"
FORMAT_VERSION = 1

META_FILE = "meta.json"
CONFIG_FILE = "config.yaml"
MODEL_SPEC_FILE = "model.yaml"
PICKLED_MODEL_FILE = "model.pt"
WEIGHTS_FILE = "model.safetensors"
NORMALIZERS_FILE = "normalizers.json"
TRAIN_STATE_FILE = "train_state.pt"
REPLAY_DIR = "replay_buffer"
REPLAY_FILE = "state.pt"
LEGACY_PAYLOAD_FILE = "agent.pt"  # format v0

#: Payload keys holding running statistics (off-policy ObsNormalizer / PPO).
_NORMALIZER_KEYS = ("obs_normalizer_state", "normalizers")
_ON_POLICY = ("ppo", "a2c", "a3c")
_FORMAT_FILES = (
    META_FILE,
    CONFIG_FILE,
    MODEL_SPEC_FILE,
    PICKLED_MODEL_FILE,
    WEIGHTS_FILE,
    NORMALIZERS_FILE,
    TRAIN_STATE_FILE,
    LEGACY_PAYLOAD_FILE,
)


class CheckpointFormatError(ValueError):
    """A checkpoint is missing files, has an unknown format, or does not match its model."""


def _srl_version() -> str:
    try:
        from srl import __version__

        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────


def save_agent(
    agent,
    path: str | Path,
    *,
    include_replay_buffer: bool = False,
    include_train_state: bool = True,
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
    extra_train_state: dict[str, Any] | None = None,
) -> Path:
    """Write *agent* to directory *path* in format v1; returns the directory.

    ``include_train_state=False`` writes an inference-only checkpoint (no
    optimizers / target network / RNG). *step*, *metrics* and
    *extra_train_state* (e.g. RNG state) are what ``srl-train`` checkpoints
    add on top of the agent itself.
    """
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
        torch.save(pack(payload), path)
        return path

    path.mkdir(parents=True, exist_ok=True)
    _remove_format_files(path)
    agent_meta = dict(payload.pop("srl_agent", None) or {})
    model_state = payload.pop("model_state", None)
    if model_state is None:
        model_state = agent.model.state_dict()
    _save_weights(model_state, path / WEIGHTS_FILE)

    normalizers = {k: payload.pop(k) for k in _NORMALIZER_KEYS if k in payload}
    normalizers = {k: v for k, v in normalizers.items() if v is not None}
    if normalizers:
        (path / NORMALIZERS_FILE).write_text(
            json.dumps({"format_version": FORMAT_VERSION, **to_jsonable(normalizers)}),
            encoding="utf-8",
        )

    action_bounds = payload.pop("action_bounds", None)
    replay = payload.pop("replay_buffer_state", None)
    if include_replay_buffer and replay is not None:
        (path / REPLAY_DIR).mkdir(exist_ok=True)
        torch.save(pack(replay), path / REPLAY_DIR / REPLAY_FILE)

    train_state = dict(payload)
    train_state.update(extra_train_state or {})
    if step is not None:
        train_state["step"] = int(step)
    if metrics is not None:
        train_state["metrics"] = dict(metrics)
    if include_train_state:
        torch.save(pack(train_state), path / TRAIN_STATE_FILE)

    model_source = _write_model_source(agent, agent_meta.get("model_spec"), path)
    meta = _build_meta(
        agent,
        agent_meta,
        normalizers=normalizers,
        action_bounds=action_bounds,
        model_source=model_source,
        include_train_state=include_train_state,
        include_replay_buffer=bool(include_replay_buffer and replay is not None),
        step=step,
        metrics=metrics,
    )
    # meta.json last: a directory without it is an incomplete write.
    (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _remove_format_files(path: Path) -> None:
    for name in _FORMAT_FILES:
        (path / name).unlink(missing_ok=True)
    if (path / REPLAY_DIR).is_dir():
        shutil.rmtree(path / REPLAY_DIR)


def _save_weights(state: dict[str, Any], target: Path) -> None:
    from safetensors.torch import save_file

    # clone(): safetensors refuses tensors that share storage (tied weights).
    tensors = {k: v.detach().cpu().contiguous().clone() for k, v in state.items()}
    save_file(
        tensors,
        str(target),
        metadata={"format": FORMAT_NAME, "format_version": str(FORMAT_VERSION)},
    )


def _write_model_source(agent, model_spec: Any, path: Path) -> str:
    """``config.yaml`` (experiment agents) > ``model.yaml`` (spec) > ``model.pt`` (pickle)."""
    if getattr(agent, "_experiment", None) is not None:
        _write_resolved_config(agent, path / CONFIG_FILE)
        return CONFIG_FILE
    if model_spec is not None:
        import yaml

        (path / MODEL_SPEC_FILE).write_text(
            yaml.safe_dump(model_spec, sort_keys=False), encoding="utf-8"
        )
        return MODEL_SPEC_FILE
    warnings.warn(
        f"{type(agent).__name__}: no model spec is known (the model was not built from a "
        "config), so the model is pickled to model.pt and loading it needs "
        "weights_only=False. Build agents with from_config() to get a pickle-free "
        "checkpoint.",
        stacklevel=4,
    )
    torch.save(agent.model, path / PICKLED_MODEL_FILE)
    return PICKLED_MODEL_FILE


def _write_resolved_config(agent, target: Path) -> None:
    """``config.yaml``: the fully resolved experiment the agent was built from.

    Written with :func:`srl.config.dump_resolved_config` (explicit dims, every
    hyperparameter including defaults, the agent's *current* algorithm
    config), so ``srl.config.load_experiment(<dir>/config.yaml)`` reproduces
    the run's configuration.
    """
    from srl.config import dump_resolved_config

    exp = agent._experiment
    if dataclasses.is_dataclass(agent.cfg) and type(agent.cfg) is type(exp.algo_config):
        exp = dataclasses.replace(exp, algo_config=agent.cfg)
    dump_resolved_config(exp, target)


def _model_config_of(agent) -> Any:
    """The agent's :class:`AgentModelConfig` (explicit dims), if known."""
    exp = getattr(agent, "_experiment", None)
    if exp is not None:
        return exp.model
    spec = getattr(agent, "_model_spec", None)
    if spec is None:
        return None
    from srl.registry.config_schema import AgentModelConfig

    try:
        return AgentModelConfig.from_dict(spec)
    except Exception:  # noqa: BLE001 -- the spec is informational here
        return None


def obs_spec_from_model_config(model_cfg: Any, obs_key: str, obs_shape: Any) -> dict[str, Any]:
    """``meta.json`` ``obs_spec``: the env obs key/shape and each encoder's input."""
    encoders: dict[str, Any] = {}
    for enc in getattr(model_cfg, "encoders", None) or []:
        if enc.name in encoders:
            continue
        if enc.input_shape is not None:
            shape = [int(d) for d in enc.input_shape]
        elif enc.input_dim is not None:
            shape = [int(enc.input_dim)]
        else:
            shape = None
        encoders[enc.name] = {
            "type": enc.type,
            "input_name": enc.input_name,
            "shape": shape,
            "dtype": "float32",
            "recurrent": bool(enc.recurrent or enc.type == "lstm"),
        }
    return {
        "obs_key": obs_key,
        "obs_shape": list(obs_shape) if obs_shape is not None else None,
        "encoders": encoders,
    }


def _as_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    import numpy as np

    return np.asarray(value).tolist()


def _action_spec(agent, agent_meta: dict, action_bounds: Any) -> dict[str, Any]:
    rescale = agent_meta.get("env_action_rescale")
    clip = None
    if action_bounds is not None:
        clip = {"low": _as_list(action_bounds["low"]), "high": _as_list(action_bounds["high"])}
    if rescale is not None:
        rescale = {k: _as_list(rescale[k]) for k in ("low", "high", "finite")}
    shape = None
    for source in (rescale, clip):
        if source is not None:
            shape = [len(source["low"])]
            break
    if shape is None:
        dim = getattr(agent.cfg, "action_dim", None)
        model_cfg = _model_config_of(agent)
        actor = getattr(model_cfg, "actor", None)
        dim = dim or getattr(actor, "action_dim", None)
        shape = [int(dim)] if dim else None
    # Bounds in env units: the rescale target when actions are rescaled,
    # otherwise the clip range (or the model's [-1, 1] for squashed heads).
    low = high = None
    if rescale is not None:
        low, high = rescale["low"], rescale["high"]
    elif clip is not None:
        low, high = clip["low"], clip["high"]
    return {"shape": shape, "low": low, "high": high, "rescale": rescale, "clip": clip}


def _normalizers_present(normalizers: dict[str, Any]) -> list[str]:
    present: list[str] = []
    off = normalizers.get("obs_normalizer_state")
    if off and off.get("mean"):
        present.append("obs")
    ppo = normalizers.get("normalizers") or {}
    if ppo.get("obs") and "obs" not in present:
        present.append("obs")
    if ppo.get("ret") is not None:
        present.append("reward")
    return present


def _jsonable_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (metrics or {}).items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _build_meta(
    agent,
    agent_meta: dict,
    *,
    normalizers: dict,
    action_bounds: Any,
    model_source: str,
    include_train_state: bool,
    include_replay_buffer: bool,
    step: int | None,
    metrics: dict | None,
) -> dict[str, Any]:
    model_cfg = _model_config_of(agent)
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "srl_version": _srl_version(),
        "torch_version": str(torch.__version__),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algo": agent_meta.get("algo") or getattr(agent, "algo_name", ""),
        "agent_class": agent_meta.get("agent_class"),
        "config_class": agent_meta.get("config_class"),
        "env_type": getattr(agent, "_env_type", None),
        "num_timesteps": int(agent_meta.get("num_timesteps", 0)),
        "step": int(step) if step is not None else None,
        "metrics": _jsonable_metrics(metrics),
        "obs_spec": obs_spec_from_model_config(
            model_cfg, agent_meta.get("obs_key", "state"), agent_meta.get("obs_shape")
        ),
        "action_spec": _action_spec(agent, agent_meta, action_bounds),
        "normalizers": _normalizers_present(normalizers),
        "model_source": model_source,
        "includes_train_state": bool(include_train_state),
        "includes_replay_buffer": bool(include_replay_buffer),
        # The JSON-able algorithm config fields (config.yaml, when present,
        # is authoritative; this copy serves model.yaml / model.pt agents).
        "algo_config": agent_meta.get("algo_config") or {},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Read
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Checkpoint:
    """A checkpoint read from disk, whatever its format.

    ``payload`` is the :meth:`BaseAgent.checkpoint_payload`-shaped dict
    (``model_state``, optimizer states, ``srl_agent`` metadata, ...);
    ``experiment`` the validated ``config.yaml`` (when present).
    """

    path: Path
    format_version: int
    payload: dict[str, Any]
    meta: dict[str, Any]
    experiment: Any = None
    model_spec: dict | None = None
    pickled_model: Path | None = None

    @property
    def agent_meta(self) -> dict[str, Any]:
        return self.payload.get("srl_agent") or {}

    @property
    def algo(self) -> str | None:
        return self.agent_meta.get("algo") or None

    @property
    def model_config(self) -> Any:
        """The :class:`AgentModelConfig` the model is built from (``None`` for a pickled model)."""
        if self.experiment is not None:
            return self.experiment.model
        if self.model_spec is not None:
            from srl.registry.config_schema import AgentModelConfig

            return AgentModelConfig.from_dict(self.model_spec)
        return None

    def build_model(self, device: Any = "cpu") -> torch.nn.Module:
        """A fresh model with this checkpoint's architecture (weights not loaded)."""
        if self.experiment is not None:
            from srl.registry.builder import ModelBuilder

            return ModelBuilder.from_config(self.experiment.model)
        if self.model_spec is not None:
            from srl.registry.builder import ModelBuilder

            return ModelBuilder.from_dict(self.model_spec)
        if self.pickled_model is not None:
            warnings.warn(
                f"{self.pickled_model}: pickled model; loading it with "
                "torch.load(weights_only=False), which can execute code from the file -- "
                "only load checkpoints you trust.",
                LegacyCheckpointWarning,
                stacklevel=3,
            )
            return torch.load(self.pickled_model, map_location=device, weights_only=False)
        raise CheckpointFormatError(
            f"{self.path}: no model spec stored with this agent; pass model=... to load()."
        )

    def load_weights(self, model: torch.nn.Module) -> None:
        """``model.load_state_dict(weights)`` with a clear error on a mismatch (strict)."""
        state = self.payload.get("model_state", self.payload.get("model"))
        if state is None:
            raise CheckpointFormatError(f"{self.path}: checkpoint holds no model weights")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise CheckpointFormatError(
                f"{self.path}: the stored weights do not match the model built from its "
                f"config ({exc})"
            ) from exc


def is_agent_dir(path: str | Path) -> bool:
    """Whether *path* is a directory written by :func:`save_agent` (any format version)."""
    path = Path(path)
    return path.is_dir() and ((path / META_FILE).exists() or (path / LEGACY_PAYLOAD_FILE).exists())


def _read_meta(path: Path) -> dict[str, Any]:
    try:
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointFormatError(f"{path / META_FILE}: not valid JSON ({exc})") from exc
    if not isinstance(meta, dict) or meta.get("format") != FORMAT_NAME:
        raise CheckpointFormatError(
            f"{path / META_FILE}: not an SRL agent checkpoint (expected format: "
            f"{FORMAT_NAME!r}, got {meta.get('format') if isinstance(meta, dict) else meta!r})"
        )
    version = meta.get("format_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version < 0:
        raise CheckpointFormatError(f"{path / META_FILE}: invalid format_version {version!r}")
    if version > FORMAT_VERSION:
        raise CheckpointFormatError(
            f"{path}: written in srl-agent format_version {version} by SRL "
            f"{meta.get('srl_version', '?')}; this SRL ({_srl_version()}) reads format_version "
            f"<= {FORMAT_VERSION}. Upgrade SRL to load it."
        )
    return meta


def _arrays(d: Any) -> Any:
    if d is None:
        return None
    import numpy as np

    out = {k: np.asarray(d[k], dtype=np.float64) for k in ("low", "high")}
    if "finite" in d:
        out["finite"] = np.asarray(d["finite"], dtype=bool)
    return out


def _read_v1(path: Path, meta: dict, map_location: Any, inference_only: bool) -> Checkpoint:
    from safetensors.torch import load_file

    weights = path / WEIGHTS_FILE
    if not weights.exists():
        raise CheckpointFormatError(
            f"{path}: incomplete srl-agent v1 checkpoint: {WEIGHTS_FILE} is missing"
        )
    source = meta.get("model_source")
    if (
        source not in (CONFIG_FILE, MODEL_SPEC_FILE, PICKLED_MODEL_FILE)
        or not (path / source).exists()
    ):
        raise CheckpointFormatError(
            f"{path}: incomplete srl-agent v1 checkpoint: the model description "
            f"({source or 'config.yaml / model.yaml'}) is missing"
        )

    experiment = model_spec = pickled = None
    if source == CONFIG_FILE:
        from srl.config import ConfigError, experiment_to_dict, load_experiment

        try:
            experiment = load_experiment(path / CONFIG_FILE)
        except ConfigError as exc:
            raise CheckpointFormatError(f"{path / CONFIG_FILE}: does not validate: {exc}") from exc
        model_spec = experiment_to_dict(experiment)
    elif source == MODEL_SPEC_FILE:
        import yaml

        model_spec = yaml.safe_load((path / MODEL_SPEC_FILE).read_text(encoding="utf-8")) or {}
    else:
        pickled = path / PICKLED_MODEL_FILE

    payload: dict[str, Any] = {}
    if not inference_only and (path / TRAIN_STATE_FILE).exists():
        payload.update(load_weights_only(path / TRAIN_STATE_FILE, map_location))
    payload["model_state"] = load_file(str(weights), device=str(map_location))
    if (path / NORMALIZERS_FILE).exists():
        stats = from_jsonable(json.loads((path / NORMALIZERS_FILE).read_text(encoding="utf-8")))
        for key in _NORMALIZER_KEYS:
            if stats.get(key) is not None:
                payload[key] = stats[key]
    elif meta.get("normalizers"):
        raise CheckpointFormatError(
            f"{path}: meta.json lists normalizers {meta['normalizers']} but "
            f"{NORMALIZERS_FILE} is missing"
        )
    replay = path / REPLAY_DIR / REPLAY_FILE
    if not inference_only and replay.exists():
        payload["replay_buffer_state"] = load_weights_only(replay, map_location)
    action_spec = meta.get("action_spec") or {}
    if action_spec.get("clip") is not None:
        payload["action_bounds"] = _arrays(action_spec["clip"])
    obs_spec = meta.get("obs_spec") or {}
    payload["srl_agent"] = {
        "algo": meta.get("algo"),
        "agent_class": meta.get("agent_class"),
        "config_class": meta.get("config_class"),
        "algo_config": meta.get("algo_config") or {},
        "model_spec": model_spec,
        "env_action_rescale": _arrays(action_spec.get("rescale")),
        "obs_key": obs_spec.get("obs_key", "state"),
        "obs_shape": obs_spec.get("obs_shape"),
        "num_timesteps": int(meta.get("num_timesteps") or 0),
    }
    if meta.get("step") is not None:
        payload.setdefault("step", int(meta["step"]))
    payload.setdefault("metrics", meta.get("metrics") or {})
    return Checkpoint(
        path=path,
        format_version=int(meta["format_version"]),
        payload=payload,
        meta=meta,
        experiment=experiment,
        model_spec=model_spec,
        pickled_model=pickled,
    )


def _read_legacy(path: Path, map_location: Any) -> Checkpoint:
    """Format v0 directory (``agent.pt``) or a single-file ``.pt`` checkpoint."""
    directory = path if path.is_dir() else None
    file = path / LEGACY_PAYLOAD_FILE if directory is not None else path
    if not file.exists():
        raise CheckpointFormatError(f"{path}: srl-agent v0 directory without {LEGACY_PAYLOAD_FILE}")
    payload = load_trusted_or_legacy(file, map_location)
    if not isinstance(payload, dict):
        raise CheckpointFormatError(f"{file}: not an SRL checkpoint (got {type(payload).__name__})")
    meta: dict[str, Any] = {}
    experiment = pickled = None
    if directory is not None:
        meta = (
            json.loads((directory / META_FILE).read_text(encoding="utf-8"))
            if (directory / META_FILE).exists()
            else {}
        )
        if (directory / CONFIG_FILE).exists():
            experiment = _read_resolved_config(directory / CONFIG_FILE)
        if (directory / PICKLED_MODEL_FILE).exists():
            pickled = directory / PICKLED_MODEL_FILE
    model_spec = (payload.get("srl_agent") or {}).get("model_spec")
    if model_spec is None and directory is not None and (directory / MODEL_SPEC_FILE).exists():
        import yaml

        model_spec = yaml.safe_load((directory / MODEL_SPEC_FILE).read_text(encoding="utf-8"))
    return Checkpoint(
        path=path,
        format_version=0,
        payload=payload,
        meta=meta,
        experiment=experiment,
        model_spec=model_spec,
        pickled_model=pickled,
    )


def read_checkpoint(
    path: str | Path, map_location: Any = "cpu", *, inference_only: bool = False
) -> Checkpoint:
    """Read any SRL agent checkpoint: a v1/v0 directory or a legacy ``.pt`` file.

    ``inference_only=True`` skips ``train_state.pt`` and the replay buffer.
    """
    path = Path(path)
    if path.is_dir():
        if (path / META_FILE).exists():
            meta = _read_meta(path)
            if int(meta["format_version"]) >= 1:
                return _read_v1(path, meta, map_location, inference_only)
        if (path / LEGACY_PAYLOAD_FILE).exists():
            return _read_legacy(path, map_location)
        if (path / META_FILE).exists():
            raise CheckpointFormatError(
                f"{path}: srl-agent v0 directory without {LEGACY_PAYLOAD_FILE}"
            )
        hint = ""
        found = find_checkpoint(path, "auto", missing_ok=True)
        if found is not None:
            hint = f" It looks like an srl-train checkpoint directory; load {found} instead."
        raise CheckpointFormatError(f"{path} is not an SRL agent directory (no {META_FILE}).{hint}")
    if not path.exists():
        raise FileNotFoundError(f"No SRL agent found at {path}")
    if path.suffix == ".safetensors":
        meta_path = path.with_suffix(".meta.pt")
        raise CheckpointFormatError(
            f"{path} is a bare-model safetensors checkpoint (no agent state); "
            f"load it with CheckpointManager.load(model, ...) instead (meta: {meta_path})."
        )
    return _read_legacy(path, map_location)


def read_payload(path: str | Path, map_location: Any = "cpu") -> dict[str, Any]:
    """The agent payload of any checkpoint (see :func:`read_checkpoint`)."""
    return read_checkpoint(path, map_location).payload


_CHECKPOINT_TAGS = ("best", "final", "ckpt")


def find_checkpoint(directory: str | Path, which: str = "auto", *, missing_ok: bool = False):
    """The checkpoint to load from an ``srl-train`` checkpoint directory.

    *which*: ``"best"``, ``"final"``, ``"latest"`` (newest periodic
    ``ckpt_*``) or ``"auto"`` (best, else final, else latest). Both v1
    sub-directories and legacy ``.pt`` files are considered; the highest step
    wins within a tag.
    """
    directory = Path(directory)
    if which not in ("auto", "best", "final", "latest"):
        raise ValueError(f"which must be auto/best/final/latest, got {which!r}")
    order = {"auto": _CHECKPOINT_TAGS, "best": ("best",), "final": ("final",), "latest": ("ckpt",)}
    for tag in order[which]:
        candidates = [
            p
            for p in directory.glob(f"{tag}_*")
            if (p.is_dir() and is_agent_dir(p)) or (p.is_file() and p.suffix == ".pt")
        ]
        if candidates:
            return max(candidates, key=lambda p: (_step_of(p), p.is_dir()))
    if missing_ok:
        return None
    raise FileNotFoundError(f"No {which} SRL checkpoint in {directory}")


def _step_of(path: Path) -> int:
    stem = path.name.split(".", 1)[0]
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


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
    """The algorithm config from the JSON-able fields stored in the checkpoint.

    The class comes from the algorithm registry (``SACConfig`` or its
    ``Visual*`` variant), never from an arbitrary name in the file; fields the
    class does not have are dropped with a warning.
    """
    from srl.config.algos import algorithm_spec

    cfg_cls = agent_cls.config_cls
    name = meta.get("config_class")
    if name and agent_cls.algo_name:
        spec = algorithm_spec(agent_cls.algo_name)
        for candidate in (spec.config_cls, getattr(spec, "visual_config_cls", None)):
            if candidate is not None and candidate.__name__ == name:
                cfg_cls = candidate
                break
        else:
            if cfg_cls is not None and name != cfg_cls.__name__:
                warnings.warn(
                    f"checkpoint config class {name!r} is not a {agent_cls.algo_name.upper()} "
                    f"config; using {cfg_cls.__name__}",
                    stacklevel=3,
                )
    known = {f.name for f in dataclasses.fields(cfg_cls)}
    stored = meta.get("algo_config") or {}
    unknown = sorted(set(stored) - known)
    if unknown:
        warnings.warn(
            f"checkpoint algorithm config has fields {cfg_cls.__name__} does not know "
            f"(ignored): {unknown}",
            stacklevel=3,
        )
    return cfg_cls(**{k: v for k, v in stored.items() if k in known})


def _config_for(agent_cls, ckpt: Checkpoint) -> Any:
    exp = ckpt.experiment
    if exp is not None and exp.algo_config is not None:
        if agent_cls.algo_name and exp.algo and exp.algo != agent_cls.algo_name:
            raise CheckpointFormatError(
                f"{ckpt.path}: config.yaml is for {exp.algo.upper()}, the checkpoint was "
                f"loaded as {agent_cls.algo_name.upper()}"
            )
        return exp.algo_config
    return _build_config(agent_cls, ckpt.agent_meta)


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
    if path.is_dir() and not is_agent_dir(path):
        found = find_checkpoint(path, "auto", missing_ok=True)
        if found is not None:
            path = found
    ckpt = read_checkpoint(path, map_location=device)
    agent_cls = _resolve_agent_class(cls, ckpt.algo)
    if model is None:
        model = ckpt.build_model(device)
    cfg = _config_for(agent_cls, ckpt)
    env = wrap_env(env, device=str(device))
    if env is not None and getattr(cfg, "use_her", False):
        from srl.training.algo_config import configure_her_from_env

        configure_her_from_env(cfg, env, "goal")
    agent = agent_cls(model, config=cfg, device=device, env=env)
    agent._model_spec = ckpt.model_spec
    ckpt.load_weights(agent.model)
    agent.load_checkpoint_payload(ckpt.payload)
    target = getattr(agent, "target_model", None)
    if target is not None and "target_model_state" not in ckpt.payload:
        # Inference-only checkpoint (no train_state.pt): start the target
        # network from the trained weights, not the random init.
        target.load_state_dict(agent.model.state_dict())
    if ckpt.experiment is not None:
        agent._experiment = ckpt.experiment
        agent._env_type = ckpt.experiment.env.env_type
    if env is not None:
        # The live env wins over the stored action map.
        agent._attach_env(env)
    return agent


def _read_resolved_config(path: Path) -> Any:
    """A v0 ``config.yaml`` as an ExperimentConfig, or ``None`` if it no longer validates."""
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
