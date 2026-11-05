"""Strict construction of algorithm configs from a YAML ``train:`` block."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from srl.config._validate import (
    ConfigDeprecationWarning,
    ConfigError,
    from_mapping,
    suggest,
    yaml_fields,
)

__all__ = [
    "TrainerConfig",
    "OffPolicySchedule",
    "AUX_TYPE_TO_LOSS",
    "TRAIN_ENV_KEYS",
    "DEPRECATED_TRAIN_KEYS",
    "algorithm_spec",
    "resolve_algo_config_cls",
    "build_algo_config",
    "build_runner_config",
    "build_trainer_config",
    "normalize_offpolicy_schedule",
    "split_train_section",
]


@dataclass
class TrainerConfig:
    """Trainer/CLI-level keys of the ``train:`` block (not algorithm hyperparameters).

    Most are also ``srl-train`` flags; the CLI flag wins when passed.
    """

    total_steps: int = 1_000_000
    n_envs: int = 1
    device: str = "auto"
    seed: int = 0
    log_interval: int = 2048
    eval_freq: int = 50_000
    eval_episodes: int = 10
    deterministic: bool = False
    save_best: bool = False
    #: env steps between periodic ``srl-train`` checkpoints (``ckpt_<step>/``)
    save_interval: int = 100_000
    #: store the off-policy replay buffer in every srl-train checkpoint
    #: (exact ``--resume``; can be gigabytes per checkpoint)
    checkpoint_replay_buffer: bool = False
    #: off-policy only: average training metrics over this many env steps
    #: before logging them (1 = log every gradient step; no CLI flag)
    metrics_log_interval: int = 1000

    def __post_init__(self) -> None:
        from srl.config._validate import validate_constraints

        validate_constraints(self)


@dataclass(frozen=True)
class OffPolicySchedule:
    """Canonical off-policy step schedule (see :func:`normalize_offpolicy_schedule`)."""

    learning_starts: int  # env steps of uniform-random actions before the policy acts
    update_after: int  # env steps collected before the first gradient update
    train_freq: int  # env steps between update rounds
    gradient_steps: int  # gradient steps per update round


# Encoder-level ``aux_type`` -> algorithm-level ``aux_loss_type``.
AUX_TYPE_TO_LOSS: dict[str, str] = {"autoencoder": "ae", "contrastive": "curl", "byol": "byol"}

# Env keys historically accepted inside ``train:`` (srl-train reads
# ``train.env_id`` / ``train.env_type`` as a fallback); prefer the top level.
TRAIN_ENV_KEYS = ("env_id", "env_type", "include_goal")

# Keys removed from the schema: accepted with a warning, then ignored.
DEPRECATED_TRAIN_KEYS = {
    "prefill_steps": "it was never read by AsyncOffPolicyRunner; use learning_starts",
    "queue_maxsize": "AsyncOffPolicyRunner has no collector/trainer queue",
}

# Legacy alias -> canonical name.
_SCHEDULE_ALIASES = {"start_steps": "learning_starts", "update_every": "train_freq"}


def algorithm_spec(algo_name: str) -> Any:
    """The registered :class:`~srl.registry.registry.AlgorithmSpec` for ``algo_name``."""
    from srl.registry.registry import AlgorithmRegistry, AlgorithmSpec

    spec = AlgorithmRegistry.get(str(algo_name).lower())
    if not isinstance(spec, AlgorithmSpec):
        raise ConfigError(
            f"algorithm plugin '{algo_name}' must register an srl.registry.AlgorithmSpec, "
            f"got {type(spec).__name__}"
        )
    return spec


def _runner_fields() -> set[str]:
    from srl.core.config import AsyncRunnerConfig

    return set(yaml_fields(AsyncRunnerConfig))


def _non_algo_keys() -> set[str]:
    return (
        set(yaml_fields(TrainerConfig))
        | _runner_fields()
        | set(TRAIN_ENV_KEYS)
        | set(DEPRECATED_TRAIN_KEYS)
    )


def split_train_section(train: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Partition a raw ``train:`` mapping into trainer / runner / env / deprecated / algo keys."""
    trainer_keys = set(yaml_fields(TrainerConfig))
    runner_keys = _runner_fields()
    parts: dict[str, dict[str, Any]] = {
        "trainer": {},
        "runner": {},
        "env": {},
        "deprecated": {},
        "algo": {},
    }
    for key, value in train.items():
        if key in trainer_keys:
            parts["trainer"][key] = value
        elif key in runner_keys:
            parts["runner"][key] = value
        elif key in TRAIN_ENV_KEYS:
            parts["env"][key] = value
        elif key in DEPRECATED_TRAIN_KEYS:
            parts["deprecated"][key] = value
        else:
            parts["algo"][key] = value
    return parts


def _recognized_aux_types(model_dict: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    encoders = (model_dict or {}).get("encoders") if isinstance(model_dict, Mapping) else None
    if encoders is None and model_dict is not None and hasattr(model_dict, "encoders"):
        encoders = [dataclasses.asdict(e) for e in model_dict.encoders]
    for index, enc in enumerate(encoders or []):
        if not isinstance(enc, Mapping) or enc.get("aux_type") is None:
            continue
        aux_type = str(enc["aux_type"]).strip().lower()
        if aux_type in AUX_TYPE_TO_LOSS:
            out.append((str(enc.get("name") or f"encoder[{index}]"), aux_type))
    return out


def resolve_algo_config_cls(
    algo_name: str, train_dict: Mapping[str, Any] | None = None, model: Any = None
) -> tuple[type, dict[str, Any]]:
    """Pick the plain or the ``Visual*`` config class for ``algo_name``.

    Same rules as ``srl-train``: the Visual variant is used when an encoder of
    ``model`` (raw dict or ``AgentModelConfig``) declares a recognized
    ``aux_type``, or ``train_dict`` sets a Visual-only field (``encoder_lr``,
    ``aux_weight``...). Returns ``(config_cls, extra_defaults)`` where
    ``extra_defaults`` carries ``aux_loss_type`` inferred from the encoder's
    ``aux_type`` when the user did not set it. Unlike the CLI helper this
    prints nothing.
    """
    spec = algorithm_spec(algo_name)
    base, visual = spec.config_cls, spec.visual_config_cls
    if visual is None:
        return base, {}
    train_dict = train_dict or {}
    visual_only = set(yaml_fields(visual)) - set(yaml_fields(base))
    recognized = _recognized_aux_types(model)
    if not recognized and not (visual_only & set(train_dict)):
        return base, {}
    extra: dict[str, Any] = {}
    if recognized and "aux_loss_type" not in train_dict:
        extra["aux_loss_type"] = AUX_TYPE_TO_LOSS[recognized[0][1]]
    return visual, extra


def _fold_schedule_aliases(kwargs: dict[str, Any], config_cls: type, path: str, errors: list[str]):
    """Rewrite legacy start_steps / update_every onto canonical names, preserving behaviour."""
    field_defaults = {f.name: f.default for f in dataclasses.fields(config_cls)}
    if kwargs.get("start_steps") is not None:
        warnings.warn(
            f"{path}.start_steps is deprecated; use learning_starts (random-action warmup "
            "steps). update_after keeps its previous default (the old learning_starts).",
            ConfigDeprecationWarning,
            stacklevel=4,
        )
        old_learning_starts = kwargs.get("learning_starts", field_defaults.get("learning_starts"))
        kwargs["learning_starts"] = kwargs.pop("start_steps")
        if kwargs.get("update_after") is None and old_learning_starts is not None:
            kwargs["update_after"] = old_learning_starts
    else:
        kwargs.pop("start_steps", None)
    if kwargs.get("update_every") is not None:
        warnings.warn(
            f"{path}.update_every is deprecated; use train_freq.",
            ConfigDeprecationWarning,
            stacklevel=4,
        )
        value = kwargs.pop("update_every")
        if "train_freq" in kwargs and kwargs["train_freq"] != value:
            errors.append(
                f"{path}: both train_freq ({kwargs['train_freq']}) and its deprecated alias "
                f"update_every ({value}) are set to different values; keep only train_freq"
            )
        elif "train_freq" in field_defaults:
            kwargs["train_freq"] = value
    else:
        kwargs.pop("update_every", None)


def build_algo_config(
    algo_name: str,
    train_dict: Mapping[str, Any] | None = None,
    *,
    config_cls: type | None = None,
    model: Any = None,
    overrides: Mapping[str, Any] | None = None,
    path: str = "train",
) -> Any:
    """Strictly build ``algo_name``'s config dataclass from a ``train:`` mapping.

    Drop-in strict replacement for ``srl.cli.train._build_algo_config``:

    * keys that are neither algorithm fields nor trainer/runner/env keys of
      the ``train:`` block raise :class:`~srl.config.ConfigError` with a
      "did you mean" suggestion (``lrr`` -> ``lr``); keys belonging to a
      *different* algorithm are reported as such;
    * values are type-coerced (``"3e-4"`` -> ``0.0003``) and checked against
      the field's enum/range;
    * legacy ``start_steps`` / ``update_every`` are folded into
      ``learning_starts`` / ``train_freq`` with a deprecation warning;
    * ``config_cls`` forces a class (else :func:`resolve_algo_config_cls`
      picks plain vs ``Visual*`` from ``train_dict`` and ``model``);
    * ``overrides`` (e.g. ``{"action_dim": 6, "num_envs": 8}``) are applied
      last; ``None`` values and keys the class does not have are skipped,
      matching the CLI's ``extra_overrides`` behaviour.
    """
    train_dict = dict(train_dict or {})
    errors: list[str] = []
    extra_defaults: dict[str, Any] = {}
    if config_cls is None:
        config_cls, extra_defaults = resolve_algo_config_cls(algo_name, train_dict, model)

    algo_keys = split_train_section(train_dict)["algo"]
    settable = yaml_fields(config_cls)
    kwargs: dict[str, Any] = {}
    for key, value in algo_keys.items():
        if key in settable:
            kwargs[key] = value
            continue
        owners = _algorithms_with_field(key)
        if owners:
            errors.append(
                f"{path}.{key}: not a {algo_name.upper()} hyperparameter ({config_cls.__name__}); "
                f"it belongs to {', '.join(o.upper() for o in owners)}. Remove it from "
                f"train: (or run with the algorithm it belongs to) to use {algo_name.upper()}"
            )
            continue
        valid = set(settable) | _non_algo_keys()
        hint = suggest(str(key), valid)
        errors.append(
            f"{path}.{key}: unknown key{hint}" + ("" if hint else f" for {config_cls.__name__}")
        )

    for key, value in extra_defaults.items():
        kwargs.setdefault(key, value)
    _fold_schedule_aliases(kwargs, config_cls, path, errors)
    for key, value in (overrides or {}).items():
        if value is not None and key in {f.name for f in dataclasses.fields(config_cls)}:
            kwargs[key] = value

    if errors:
        raise ConfigError(errors)
    # Overrides may carry non-YAML fields (her_reward_fn...): set those after.
    non_yaml = {k: kwargs.pop(k) for k in list(kwargs) if k not in settable}
    cfg = from_mapping(config_cls, kwargs, path, errors)
    if errors or cfg is None:
        raise ConfigError(errors)
    if non_yaml:
        cfg = dataclasses.replace(cfg, **non_yaml)
    return cfg


def _algorithms_with_field(key: str) -> list[str]:
    from srl.registry.registry import AlgorithmRegistry

    owners = []
    for name in AlgorithmRegistry.available():
        try:
            spec = algorithm_spec(name)
        except Exception:
            continue
        classes = [spec.config_cls] + ([spec.visual_config_cls] if spec.visual_config_cls else [])
        if any(key in yaml_fields(c) for c in classes):
            owners.append(name)
    return owners


def build_trainer_config(
    train_dict: Mapping[str, Any] | None, path: str = "train"
) -> TrainerConfig:
    """Strictly build the trainer-level keys of a ``train:`` block (others ignored)."""
    errors: list[str] = []
    cfg = from_mapping(
        TrainerConfig, split_train_section(train_dict or {})["trainer"], path, errors
    )
    if errors or cfg is None:
        raise ConfigError(errors)
    return cfg


def build_runner_config(train_dict: Mapping[str, Any] | None, path: str = "train") -> Any:
    """``AsyncRunnerConfig`` from the ``use_async`` / ``use_gpu_buffer`` keys of ``train:``.

    Deprecated ``prefill_steps`` / ``queue_maxsize`` warn and are ignored.
    """
    from srl.core.config import AsyncRunnerConfig

    parts = split_train_section(train_dict or {})
    for key in parts["deprecated"]:
        warnings.warn(
            f"{path}.{key} is deprecated and ignored: {DEPRECATED_TRAIN_KEYS[key]}.",
            ConfigDeprecationWarning,
            stacklevel=2,
        )
    errors: list[str] = []
    cfg = from_mapping(AsyncRunnerConfig, parts["runner"], path, errors)
    if errors or cfg is None:
        raise ConfigError(errors)
    return cfg


def normalize_offpolicy_schedule(cfg: Any) -> OffPolicySchedule:
    """The ONE place that resolves the off-policy schedule aliases.

    Works on any off-policy config object (SAC/DDPG/TD3 or a plugin's), with
    the exact semantics ``srl-train`` has always used:

    * ``learning_starts`` (random-action warmup) = ``start_steps`` if set,
      else ``learning_starts`` (default 10 000);
    * ``update_after`` = ``update_after`` if set, else ``learning_starts``
      (the config's own field, *not* ``start_steps``);
    * ``train_freq`` = ``update_every`` if set, else ``train_freq`` (default 1);
    * ``gradient_steps`` = ``gradient_steps`` (default 1).

    Values are clamped to ``>= 0`` (warmup / update_after) and ``>= 1``
    (train_freq / gradient_steps). Setting a legacy alias emits a
    :class:`~srl.config.ConfigDeprecationWarning`. Configs built with
    :func:`build_algo_config` never carry aliases (they are folded at load).
    """
    learning_starts = getattr(cfg, "learning_starts", None)
    if learning_starts is None:
        learning_starts = 10_000
    start_steps = getattr(cfg, "start_steps", None)
    if start_steps is not None:
        warnings.warn(
            "start_steps is deprecated; use learning_starts.",
            ConfigDeprecationWarning,
            stacklevel=2,
        )
    warmup = start_steps if start_steps is not None else learning_starts
    update_after = getattr(cfg, "update_after", None)
    if update_after is None:
        update_after = learning_starts
    update_every = getattr(cfg, "update_every", None)
    if update_every is not None:
        warnings.warn(
            "update_every is deprecated; use train_freq.",
            ConfigDeprecationWarning,
            stacklevel=2,
        )
    train_freq = getattr(cfg, "train_freq", None)
    if update_every is not None:
        train_freq = update_every
    if train_freq is None:
        train_freq = 1
    gradient_steps = getattr(cfg, "gradient_steps", None) or 1
    return OffPolicySchedule(
        learning_starts=max(int(warmup), 0),
        update_after=max(int(update_after), 0),
        train_freq=max(int(train_freq), 1),
        gradient_steps=max(int(gradient_steps), 1),
    )
