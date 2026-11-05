"""A small, dependency-free strict validator for SRL's config dataclasses.

Why not pydantic: every SRL config (``PPOConfig``, ``SACConfig``,
``EncoderConfig``, ...) is a plain :mod:`dataclasses` dataclass that the
algorithms, the CLI and user code construct and introspect directly
(``dataclasses.fields``, ``dataclasses.replace``, ``isinstance`` checks on
``VisualSACConfig``...). Moving them to ``pydantic.BaseModel`` would change
all of those call sites and add a hard runtime dependency for what is a
~300-line job: walk the type hints of a dataclass, coerce YAML scalars,
reject unknown keys with a "did you mean" suggestion, and check a handful of
enum (``Literal``) and numeric-range constraints.

Two entry points:

* :func:`from_mapping` -- build a dataclass from an untrusted mapping (YAML),
  with unknown-key detection, type coercion and nested dataclasses.
* :func:`validate_constraints` -- the enum / range checks, called from each
  config dataclass's ``__post_init__`` so they also apply to configs built
  directly in Python (``PPOConfig(gamma=1.5)`` raises).
"""

from __future__ import annotations

import dataclasses
import difflib
import math
import types
import typing
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Union, get_args, get_origin

__all__ = [
    "ConfigError",
    "ConfigDeprecationWarning",
    "Range",
    "suggest",
    "from_mapping",
    "coerce",
    "validate_constraints",
    "field_hints",
    "yaml_fields",
]


class ConfigError(ValueError):
    """Raised for any invalid configuration value.

    Subclasses :class:`ValueError` so pre-existing ``except ValueError``
    handlers (the model builder used to raise plain ``ValueError``) keep
    working. ``errors`` holds every individual problem found, so a config
    with several typos reports all of them at once.
    """

    def __init__(self, errors: str | Iterable[str]) -> None:
        if isinstance(errors, str):
            errors = [errors]
        self.errors: list[str] = list(errors)
        # (field, message) pairs when raised by validate_constraints(), so
        # from_mapping() can re-key them onto the YAML path ("train.gamma").
        self.field_errors: list[tuple[str, str]] = []
        if len(self.errors) == 1:
            msg = self.errors[0]
        else:
            msg = f"{len(self.errors)} configuration errors:\n" + "\n".join(
                f"  - {e}" for e in self.errors
            )
        super().__init__(msg)


class ConfigDeprecationWarning(FutureWarning):
    """A deprecated config key or value was used.

    A :class:`FutureWarning` subclass (not ``DeprecationWarning``) on purpose:
    these warnings target the person who wrote the YAML file, and Python only
    shows ``DeprecationWarning`` by default for code in ``__main__``.
    """


# ──────────────────────────────────────────────────────────────────────────────
# Suggestions
# ──────────────────────────────────────────────────────────────────────────────


def suggest(name: str, candidates: Iterable[str]) -> str:
    """Return ``" (did you mean 'x'?)"`` for the closest candidate, or ``""``."""
    matches = difflib.get_close_matches(str(name), list(candidates), n=1, cutoff=0.6)
    return f" (did you mean '{matches[0]}'?)" if matches else ""


def _join(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else str(key)


# ──────────────────────────────────────────────────────────────────────────────
# Type hints
# ──────────────────────────────────────────────────────────────────────────────


_HINTS: dict[type, dict[str, Any]] = {}


def field_hints(cls: type) -> dict[str, Any]:
    """Resolved type hints of a dataclass (handles ``from __future__ import annotations``)."""
    hints = _HINTS.get(cls)
    if hints is None:
        hints = _HINTS[cls] = typing.get_type_hints(cls)
    return hints


def yaml_fields(cls: type) -> dict[str, dataclasses.Field]:
    """Dataclass fields that may be set from YAML (``metadata={"yaml": False}`` opts out)."""
    return {f.name: f for f in dataclasses.fields(cls) if f.init and f.metadata.get("yaml", True)}


def _type_name(hint: Any) -> str:
    origin = get_origin(hint)
    if hint is type(None):
        return "null"
    if origin is Literal:
        return " | ".join(repr(a) for a in get_args(hint))
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(a) for a in get_args(hint))
    if origin is not None:
        args = ", ".join(_type_name(a) for a in get_args(hint))
        return f"{getattr(origin, '__name__', str(origin))}[{args}]"
    return getattr(hint, "__name__", str(hint))


def _describe(value: Any) -> str:
    text = repr(value)
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{text} ({type(value).__name__})"


# ──────────────────────────────────────────────────────────────────────────────
# Coercion
# ──────────────────────────────────────────────────────────────────────────────

_INVALID = object()


def _parse_number(text: str) -> float | int | None:
    # PyYAML (YAML 1.1) loads `1e-3` / `3e6` (no dot) as *strings*; accept them.
    stripped = text.strip().replace("_", "")
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return None


def coerce(value: Any, hint: Any, path: str, errors: list[str]) -> Any:
    """Coerce a YAML value to ``hint``; append an error and return a sentinel on failure."""
    if hint is Any or hint is object:
        return value

    origin = get_origin(hint)

    if origin in (Union, types.UnionType):
        args = get_args(hint)
        if value is None:
            if type(None) in args:
                return None
            errors.append(f"{path}: must not be null (expected {_type_name(hint)})")
            return _INVALID
        # Literal first (so "auto" is matched before a str/float fallback),
        # then the others in declaration order.
        ordered = sorted(
            (a for a in args if a is not type(None)),
            key=lambda a: 0 if get_origin(a) is Literal else 1,
        )
        for arg in ordered:
            trial: list[str] = []
            result = coerce(value, arg, path, trial)
            if not trial:
                return result
        errors.append(f"{path}: expected {_type_name(hint)}, got {_describe(value)}")
        return _INVALID

    if origin is Literal:
        choices = get_args(hint)
        if value in choices and not (isinstance(value, bool) and True not in choices):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            for choice in choices:
                if isinstance(choice, str) and choice.lower() == lowered:
                    return choice
        str_choices = [str(c) for c in choices]
        errors.append(
            f"{path}: invalid value {value!r}{suggest(str(value), str_choices)}; "
            f"expected one of {', '.join(repr(c) for c in choices)}"
        )
        return _INVALID

    if value is None:
        errors.append(f"{path}: must not be null (expected {_type_name(hint)})")
        return _INVALID

    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        errors.append(f"{path}: expected a boolean (true/false), got {_describe(value)}")
        return _INVALID

    if hint is int:
        if isinstance(value, bool):
            errors.append(f"{path}: expected an integer, got {_describe(value)}")
            return _INVALID
        if isinstance(value, int):
            return value
        number = _parse_number(value) if isinstance(value, str) else value
        if isinstance(number, float) and math.isfinite(number) and number.is_integer():
            return int(number)
        if isinstance(number, int) and not isinstance(number, bool):
            return number
        errors.append(f"{path}: expected an integer, got {_describe(value)}")
        return _INVALID

    if hint is float:
        if isinstance(value, bool):
            errors.append(f"{path}: expected a number, got {_describe(value)}")
            return _INVALID
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            number = _parse_number(value)
            if number is not None:
                return float(number)
        errors.append(f"{path}: expected a number, got {_describe(value)}")
        return _INVALID

    if hint is str:
        if isinstance(value, str):
            return value
        errors.append(f"{path}: expected a string, got {_describe(value)}")
        return _INVALID

    if origin is list or hint is list:
        if not isinstance(value, (list, tuple)):
            errors.append(f"{path}: expected a list, got {_describe(value)}")
            return _INVALID
        (item_hint,) = get_args(hint) or (Any,)
        out = [coerce(v, item_hint, _join(path, i), errors) for i, v in enumerate(value)]
        return _INVALID if any(v is _INVALID for v in out) else out

    if origin is tuple or hint is tuple:
        if not isinstance(value, (list, tuple)):
            errors.append(f"{path}: expected a list, got {_describe(value)}")
            return _INVALID
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            hints = [args[0]] * len(value)
        elif args:
            if len(args) != len(value):
                errors.append(f"{path}: expected {len(args)} items, got {len(value)}")
                return _INVALID
            hints = list(args)
        else:
            hints = [Any] * len(value)
        out = [
            coerce(v, h, _join(path, i), errors)
            for i, (v, h) in enumerate(zip(value, hints, strict=True))
        ]
        return _INVALID if any(v is _INVALID for v in out) else tuple(out)

    if origin is dict or hint is dict:
        if not isinstance(value, Mapping):
            errors.append(f"{path}: expected a mapping, got {_describe(value)}")
            return _INVALID
        key_hint, val_hint = get_args(hint) or (Any, Any)
        out_d = {}
        for k, v in value.items():
            ck = coerce(k, key_hint, _join(path, str(k)), errors)
            cv = coerce(v, val_hint, _join(path, str(k)), errors)
            out_d[ck] = cv
        if any(v is _INVALID for v in out_d.values()) or _INVALID in out_d:
            return _INVALID
        return out_d

    if origin is Callable or hint is Callable or origin is type(Callable):
        errors.append(f"{path}: cannot be set from a config file (Python callables only)")
        return _INVALID

    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        if isinstance(value, hint):
            return value
        return from_mapping(hint, value, path, errors)

    if isinstance(hint, type):
        if isinstance(value, hint):
            return value
        errors.append(f"{path}: expected {_type_name(hint)}, got {_describe(value)}")
        return _INVALID

    return value


def from_mapping(
    cls: type,
    data: Any,
    path: str,
    errors: list[str],
    *,
    ignore: Iterable[str] = (),
    extra_ok: bool = False,
) -> Any:
    """Build dataclass ``cls`` from ``data`` strictly.

    Unknown keys are errors with a difflib suggestion; missing required keys
    are errors; values are coerced via :func:`coerce`. ``ignore`` names keys
    the caller handles itself; ``extra_ok`` silently skips unknown keys
    (the caller must then handle them). Returns the instance, or ``None``
    when anything went wrong (the problems are appended to ``errors``).
    """
    label = path or cls.__name__
    if not isinstance(data, Mapping):
        errors.append(f"{label}: expected a mapping, got {_describe(data)}")
        return None

    settable = yaml_fields(cls)
    hints = field_hints(cls)
    ignore = set(ignore)
    n_errors = len(errors)
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key in ignore:
            continue
        if key not in settable:
            if extra_ok:
                continue
            all_fields = {f.name for f in dataclasses.fields(cls)}
            if key in all_fields:
                errors.append(f"{_join(path, key)}: cannot be set from a config file")
                continue
            hint_text = suggest(str(key), settable)
            if not hint_text:
                hint_text = f"; valid keys: {', '.join(sorted(settable))}"
            errors.append(f"{_join(path, key)}: unknown key{hint_text}")
            continue
        coerced = coerce(value, hints[key], _join(path, key), errors)
        if coerced is not _INVALID:
            kwargs[key] = coerced

    for name, f in settable.items():
        if (
            name not in kwargs
            and name not in data
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            errors.append(f"{_join(path, name)}: required key is missing")

    if len(errors) != n_errors:
        return None
    try:
        return cls(**kwargs)
    except ConfigError as exc:
        if exc.field_errors:
            errors.extend(f"{_join(path, f) if path else f}: {m}" for f, m in exc.field_errors)
        else:
            errors.extend(f"{label}: {e}" if path else e for e in exc.errors)
    except (TypeError, ValueError) as exc:
        errors.append(f"{label}: {exc}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Constraints (enums + ranges), run from __post_init__
# ──────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Range:
    """Numeric bound: ``lo`` / ``hi`` are inclusive unless ``*_open``."""

    lo: float | None = None
    hi: float | None = None
    lo_open: bool = False
    hi_open: bool = False

    def check(self, value: float) -> bool:
        if isinstance(value, float) and math.isnan(value):
            return False
        if self.lo is not None and (value <= self.lo if self.lo_open else value < self.lo):
            return False
        if self.hi is not None and (value >= self.hi if self.hi_open else value > self.hi):
            return False
        return True

    def describe(self) -> str:
        if self.lo is not None and self.hi is not None:
            left = "(" if self.lo_open else "["
            right = ")" if self.hi_open else "]"
            return f"in {left}{self.lo:g}, {self.hi:g}{right}"
        if self.lo is not None:
            return f"{'>' if self.lo_open else '>='} {self.lo:g}"
        if self.hi is not None:
            return f"{'<' if self.hi_open else '<='} {self.hi:g}"
        return "any value"


POSITIVE = Range(lo=0, lo_open=True)
NON_NEGATIVE = Range(lo=0)
AT_LEAST_ONE = Range(lo=1)
UNIT_OPEN_LOW = Range(lo=0, hi=1, lo_open=True)  # (0, 1]
UNIT = Range(lo=0, hi=1)  # [0, 1]

# Field-name based rules shared by every config dataclass; a class can add or
# override entries with a ``__srl_ranges__`` class attribute. ``None`` values
# are always accepted (they mean "unset"/"auto").
DEFAULT_RANGES: dict[str, Range] = {
    # learning rates / strictly positive scalars
    **{
        name: POSITIVE
        for name in (
            "lr",
            "lr_actor",
            "lr_critic",
            "lr_alpha",
            "encoder_lr",
            "entropy_coef_lr",
            "min_lr",
            "max_lr",
            "clip_range",
            "clip_range_vf",
            "max_grad_norm",
            "target_kl",
            "desired_kl",
            "noise_clip",
            "init_alpha",
            "clip_obs",
            "clip_reward",
            "norm_epsilon",
            "obs_norm_clip",
            "rms_prop_eps",
            "per_eps",
        )
    },
    "kl_lr_factor": Range(lo=1, lo_open=True),
    # discount-like factors
    "gamma": UNIT_OPEN_LOW,
    "tau": UNIT_OPEN_LOW,
    "momentum_tau": UNIT,
    "gae_lambda": UNIT,
    "per_alpha": UNIT,
    "per_beta_start": UNIT,
    "her_ratio": UNIT,
    # non-negative scalars
    **{
        name: NON_NEGATIVE
        for name in (
            "entropy_coef",
            "entropy_coef_final",
            "vf_coef",
            "aux_weight",
            "noise_sigma",
            "policy_noise",
            "alpha",
            "min_alpha",
            "min_entropy_coef",
            "max_entropy_coef",
            "learning_starts",
            "start_steps",
            "update_after",
            "eval_freq",
            "action_dim",
            "her_obs_dim",
            "her_goal_dim",
        )
    },
    # counts
    **{
        name: AT_LEAST_ONE
        for name in (
            "n_steps",
            "num_envs",
            "n_envs",
            "batch_size",
            "n_epochs",
            "buffer_size",
            "n_workers",
            "target_update_interval",
            "train_freq",
            "update_every",
            "gradient_steps",
            "policy_delay",
            "replay_n_step",
            "replay_num_envs",
            "encoder_update_freq",
            "her_max_episode_len",
            "per_beta_steps",
            "entropy_coef_anneal_steps",
            "total_steps",
            "log_interval",
            "save_interval",
            "metrics_log_interval",
            "eval_episodes",
            "latent_dim",
            "aux_latent_dim",
            "lstm_hidden",
            "input_dim",
        )
    },
}


def _ranges_for(cls: type) -> dict[str, Range]:
    merged = dict(DEFAULT_RANGES)
    for klass in reversed(cls.__mro__):
        merged.update(getattr(klass, "__dict__", {}).get("__srl_ranges__", {}) or {})
    return merged


def _literal_choices(hint: Any) -> tuple[Any, ...] | None:
    """Literal values allowed by ``hint`` (``None`` if it has no Literal part)."""
    origin = get_origin(hint)
    if origin is Literal:
        return get_args(hint)
    if origin in (Union, types.UnionType):
        lits = [a for a in get_args(hint) if get_origin(a) is Literal]
        if lits:
            return tuple(v for a in lits for v in get_args(a))
    return None


def _accepts_non_literal(hint: Any) -> tuple[type, ...]:
    """Non-Literal, non-None types of a Union hint (e.g. float in ``Literal['auto'] | float``)."""
    if get_origin(hint) in (Union, types.UnionType):
        return tuple(
            a
            for a in get_args(hint)
            if get_origin(a) is not Literal and a is not type(None) and isinstance(a, type)
        )
    return ()


def validate_constraints(obj: Any) -> None:
    """Check Literal enums and numeric ranges of a config dataclass instance.

    Raises :class:`ConfigError` listing every violation. Called from the
    ``__post_init__`` of SRL's config dataclasses.
    """
    cls = type(obj)
    hints = field_hints(cls)
    ranges = _ranges_for(cls)
    errors: list[tuple[str, str]] = []
    for f in dataclasses.fields(cls):
        value = getattr(obj, f.name)
        if value is None:
            continue
        hint = hints.get(f.name)
        choices = _literal_choices(hint)
        if choices is not None:
            others = _accepts_non_literal(hint)
            is_other = bool(others) and isinstance(value, others) and not isinstance(value, str)
            if value not in choices and not is_other:
                errors.append(
                    (
                        f.name,
                        f"invalid value {value!r}{suggest(str(value), map(str, choices))}; "
                        f"expected one of {', '.join(repr(c) for c in choices)}"
                        + (f" or a {'/'.join(t.__name__ for t in others)}" if others else ""),
                    )
                )
                continue
        rng = f.metadata.get("range") or ranges.get(f.name)
        if rng is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
            if not rng.check(value):
                errors.append((f.name, f"must be {rng.describe()}, got {value!r}"))
    # Optional cross-field checks: a method returning [(field, message), ...].
    check = getattr(obj, "__srl_check__", None)
    if check is not None:
        errors.extend(check())
    if errors:
        exc = ConfigError([f"{cls.__name__}.{f}: {m}" for f, m in errors])
        exc.field_errors = errors
        raise exc
