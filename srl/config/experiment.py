"""Whole-file experiment config: load, validate, dump."""

from __future__ import annotations

import dataclasses
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from srl.config._validate import (
    ConfigDeprecationWarning,
    ConfigError,
    coerce,
    from_mapping,
    suggest,
    yaml_fields,
)
from srl.config.algos import (
    TRAIN_ENV_KEYS,
    TrainerConfig,
    build_algo_config,
    build_runner_config,
    split_train_section,
)

__all__ = [
    "SCHEMA_VERSION",
    "EnvConfig",
    "ExperimentConfig",
    "infer_algo_name",
    "load_experiment",
    "experiment_to_dict",
    "dump_resolved_config",
]

#: Current config schema version. A file without ``schema_version`` is
#: treated as version 1.
SCHEMA_VERSION = 1

_ALGO_FILENAME_HINTS = ("td3", "sac", "ddpg", "a3c", "a2c", "ppo")


@dataclass
class EnvConfig:
    """Environment keys (top level of the YAML)."""

    env_id: str | None = None
    #: ``flat`` | ``goal`` | ``racecar`` | ``isaaclab`` | ``mjlab`` | a plugin env type
    env_type: str = "flat"
    #: GoalEnvWrapper: append the desired goal to the flat observation.
    #: NOTE: srl-train does not forward this to GoalEnvWrapper yet.
    include_goal: bool = True


@dataclass
class ExperimentConfig:
    """A fully validated experiment YAML.

    ``algo_config`` is the algorithm dataclass (``PPOConfig``,
    ``VisualSACConfig``...) built strictly from the ``train:`` block, or
    ``None`` when no algorithm could be determined. Values the trainer
    derives from the env (``num_envs``, ``action_dim``, ``replay_num_envs``,
    HER fields) are *not* filled in here -- use
    ``dataclasses.replace(exp, algo_config=...)`` before
    :func:`dump_resolved_config` so the dump records what actually ran.
    """

    schema_version: int = SCHEMA_VERSION
    algo: str | None = None
    env: EnvConfig = field(default_factory=EnvConfig)
    model: Any = None  # AgentModelConfig
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    runner: Any = None  # AsyncRunnerConfig
    algo_config: Any = None
    source: str | None = field(default=None, compare=False)


def infer_algo_name(raw: Mapping[str, Any], path: str | os.PathLike | None = None) -> str | None:
    """``raw['algo']``, else the first algorithm name found in the file name (srl-train's rule)."""
    algo = raw.get("algo")
    if algo:
        return str(algo).lower()
    if path is not None:
        stem = os.path.basename(str(path)).lower()
        for name in _ALGO_FILENAME_HINTS:
            if name in stem:
                return name
    return None


def _load_raw(path_or_dict: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(path_or_dict, Mapping):
        return dict(path_or_dict), None
    path = Path(path_or_dict)
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: expected a YAML mapping at the top level")
    return dict(data), str(path)


def load_experiment(
    path_or_dict: str | os.PathLike | Mapping[str, Any],
    *,
    algo: str | None = None,
) -> ExperimentConfig:
    """Load and strictly validate a whole experiment config.

    Validates the model section (encoders / flows / heads / visualization /
    ros2), the env keys (``env_id``, ``env_type``, ``include_goal``), and the
    ``train:`` block (trainer keys, async-runner keys and the algorithm's
    hyperparameters). Every problem is collected and raised together as one
    :class:`~srl.config.ConfigError`; unknown keys come with a "did you mean"
    suggestion.

    ``algo`` overrides the file's ``algo:`` (like ``srl-train --algo``);
    otherwise it is taken from ``algo:`` or inferred from the file name. When
    no algorithm can be determined, ``train:`` keys are checked against every
    registered algorithm and ``algo_config`` is ``None``.
    """
    from srl.registry.config_schema import AgentModelConfig
    from srl.registry.registry import AlgorithmRegistry, EnvRegistry

    raw, source = _load_raw(path_or_dict)
    errors: list[str] = []

    # ── schema version ────────────────────────────────────────────────────
    version = raw.get("schema_version", SCHEMA_VERSION)
    version = coerce(version, int, "schema_version", errors)
    if isinstance(version, int) and version != SCHEMA_VERSION:
        errors.append(
            f"schema_version: unsupported version {version}; this SRL version reads "
            f"schema_version {SCHEMA_VERSION}"
        )
        version = SCHEMA_VERSION

    # ── model section (also rejects unknown top-level keys) ───────────────
    model = AgentModelConfig._parse(raw, errors)

    # ── algorithm ─────────────────────────────────────────────────────────
    algo_name = (algo or "").lower() or infer_algo_name(raw, source)
    if algo_name and algo_name not in AlgorithmRegistry:
        available = AlgorithmRegistry.available()
        errors.append(
            f"algo: unknown algorithm '{algo_name}'{suggest(algo_name, available)}. "
            f"Available: {available}"
        )
        algo_name = None

    # ── train block ───────────────────────────────────────────────────────
    train = raw.get("train") or {}
    if not isinstance(train, Mapping):
        errors.append(f"train: expected a mapping, got {type(train).__name__}")
        train = {}
    train = dict(train)
    parts = split_train_section(train)

    trainer = from_mapping(TrainerConfig, parts["trainer"], "train", errors)

    runner = None
    try:
        runner = build_runner_config(train)
    except ConfigError as exc:
        errors.extend(exc.errors)

    algo_config = None
    if algo_name:
        try:
            algo_config = build_algo_config(algo_name, train, model=raw)
        except ConfigError as exc:
            errors.extend(exc.errors)
    else:
        _check_train_keys_against_all_algorithms(parts["algo"], errors)

    # ── env ───────────────────────────────────────────────────────────────
    env_raw: dict[str, Any] = {k: raw[k] for k in TRAIN_ENV_KEYS if k in raw}
    for key, value in parts["env"].items():
        if key in env_raw and env_raw[key] != value:
            errors.append(
                f"train.{key}: {value!r} conflicts with top-level {key}: {env_raw[key]!r}; "
                "keep only the top-level one"
            )
            continue
        warnings.warn(
            f"train.{key} is deprecated; move it to the top level of the config.",
            ConfigDeprecationWarning,
            stacklevel=2,
        )
        env_raw[key] = value
    env = from_mapping(EnvConfig, env_raw, "", errors)
    if env is not None:
        env.env_type = env.env_type.strip().lower()
        if env.env_type not in EnvRegistry:
            available = EnvRegistry.available()
            errors.append(
                f"env_type: unknown env type '{env.env_type}'"
                f"{suggest(env.env_type, available)}. Available: {available}"
            )

    if errors:
        where = f"{source}: " if source else ""
        if len(errors) == 1:
            raise ConfigError(f"{where}{errors[0]}")
        error = ConfigError(errors)
        if source:
            error.args = (f"{source}: {error.args[0]}",)
        raise error

    return ExperimentConfig(
        schema_version=int(version),
        algo=algo_name,
        env=env,
        model=model,
        trainer=trainer,
        runner=runner,
        algo_config=algo_config,
        source=source,
    )


def _check_train_keys_against_all_algorithms(keys: Mapping[str, Any], errors: list[str]) -> None:
    from srl.config.algos import _algorithms_with_field, _non_algo_keys
    from srl.registry.registry import AlgorithmRegistry

    for key in keys:
        if key in ("start_steps", "update_every") or _algorithms_with_field(key):
            continue
        valid: set[str] = set(_non_algo_keys())
        for name in AlgorithmRegistry.available():
            spec = AlgorithmRegistry.get(name)
            for cls in (spec.config_cls, getattr(spec, "visual_config_cls", None)):
                if cls is not None:
                    valid |= set(yaml_fields(cls))
        hint = suggest(str(key), valid)
        errors.append(f"train.{key}: unknown key{hint}")


# ──────────────────────────────────────────────────────────────────────────────
# Dump
# ──────────────────────────────────────────────────────────────────────────────


def _dataclass_to_yaml(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        if not f.init or not f.metadata.get("yaml", True):
            continue
        value = getattr(obj, f.name)
        if value is None and f.default is None:
            continue
        if f.metadata.get("deprecated_alias") and value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        out[f.name] = value
    return out


def experiment_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Plain-dict YAML form of ``cfg``; :func:`load_experiment` parses it back to an equal config.

    Every algorithm/trainer field is written out explicitly (defaults
    included), so the dump stays meaningful even if a later SRL version
    changes a default. Fields that cannot come from YAML (``her_reward_fn``,
    ``her_obs_dim``...) are omitted.
    """
    out: dict[str, Any] = {"schema_version": cfg.schema_version}
    if cfg.algo:
        out["algo"] = cfg.algo
    env = cfg.env or EnvConfig()
    if env.env_id is not None:
        out["env_id"] = env.env_id
    out["env_type"] = env.env_type
    if env.include_goal is not True:
        out["include_goal"] = env.include_goal
    if cfg.model is not None:
        out.update(cfg.model.to_dict())
    train: dict[str, Any] = {}
    train.update(_dataclass_to_yaml(cfg.trainer or TrainerConfig()))
    if cfg.runner is not None:
        train.update(_dataclass_to_yaml(cfg.runner))
    if cfg.algo_config is not None:
        algo_fields = _dataclass_to_yaml(cfg.algo_config)
        clash = set(algo_fields) & set(train)
        if clash:  # pragma: no cover - schema invariant
            raise ConfigError(f"algorithm fields clash with trainer/runner keys: {sorted(clash)}")
        train.update(algo_fields)
    out["train"] = train
    return out


class _Dumper(yaml.SafeDumper):
    pass


def _represent_float(dumper: yaml.SafeDumper, value: float) -> Any:
    # repr() round-trips exactly; make sure YAML 1.1 reads it back as a float
    # (PyYAML loads `1e-05` -- no dot -- as a string).
    text = repr(float(value))
    if text in ("inf", "-inf", "nan"):
        return yaml.SafeDumper.represent_float(dumper, value)
    if "e" in text and "." not in text.split("e")[0]:
        mantissa, exp = text.split("e")
        text = f"{mantissa}.0e{exp}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


_Dumper.add_representer(float, _represent_float)


def dump_resolved_config(cfg: ExperimentConfig, path: str | os.PathLike | None = None) -> str:
    """Write the fully resolved config as YAML (and return the text).

    ``load_experiment(path) == cfg`` holds for the result (``source`` is not
    compared). With ``path=None`` only the text is returned. Parent
    directories are created.
    """
    text = (
        "# Resolved SRL experiment config (written by srl.config.dump_resolved_config).\n"
        + yaml.dump(experiment_to_dict(cfg), Dumper=_Dumper, sort_keys=False, allow_unicode=True)
    )
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return text
