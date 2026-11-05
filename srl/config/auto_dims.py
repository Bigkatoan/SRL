"""Resolve ``input_dim: auto`` / ``input_shape: auto`` / ``action_dim: auto`` from env spaces."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from srl.config._validate import ConfigError

__all__ = ["resolve_auto_dims", "resolve_auto_dims_from_env", "space_shapes", "action_dim_of"]


def _shape_of(space: Any, path: str) -> tuple[int, ...]:
    if isinstance(space, (int, np.integer)) and not isinstance(space, bool):
        return (int(space),)
    if isinstance(space, (tuple, list)) and all(isinstance(d, (int, np.integer)) for d in space):
        return tuple(int(d) for d in space)
    shape = getattr(space, "shape", None)
    if shape is not None:
        return tuple(int(d) for d in shape)
    raise ConfigError(f"cannot infer a shape from {path} ({type(space).__name__}: {space!r})")


def space_shapes(obs_space: Any) -> dict[str, tuple[int, ...]]:
    """Per-key observation shapes from a space, a dict of spaces/shapes/arrays, or one space.

    A single (non-dict) space is published under the key ``"state"``, the key
    SRL's Gymnasium/Goal wrappers use.
    """
    if obs_space is None:
        return {}
    spaces = getattr(obs_space, "spaces", None)
    if isinstance(spaces, Mapping):
        return {str(k): _shape_of(v, f"obs_space[{k!r}]") for k, v in spaces.items()}
    if isinstance(obs_space, Mapping):
        return {str(k): _shape_of(v, f"obs_space[{k!r}]") for k, v in obs_space.items()}
    return {"state": _shape_of(obs_space, "obs_space")}


def action_dim_of(act_space: Any) -> int:
    """Flat action dimensionality (``prod(shape)``; ``n`` for a Discrete space)."""
    shape = getattr(act_space, "shape", None)
    if shape:
        return int(math.prod(int(d) for d in shape))
    n = getattr(act_space, "n", None)
    if n is not None:
        return int(n)
    if isinstance(act_space, (int, np.integer)):
        return int(act_space)
    if isinstance(act_space, (tuple, list)):
        return int(math.prod(int(d) for d in act_space))
    raise ConfigError(f"cannot infer action_dim from act_space {act_space!r}")


def _to_chw(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Mirror the wrappers' HWC -> CHW image transpose (see srl.envs._image_obs)."""
    from srl.envs._image_obs import maybe_hwc_to_chw

    return tuple(maybe_hwc_to_chw(np.empty(shape, dtype=np.uint8)).shape)


def resolve_auto_dims(
    model_cfg: Any,
    obs_space: Any = None,
    act_space: Any = None,
) -> Any:
    """Fill ``auto``/omitted dims of a model config from per-env spaces.

    Parameters
    ----------
    model_cfg:
        :class:`~srl.registry.config_schema.AgentModelConfig` (or a raw dict,
        which is parsed strictly first). Not modified; a resolved copy is
        returned.
    obs_space:
        Per-env observation space: a ``gymnasium.spaces.Dict``, a mapping of
        obs key -> space / shape / array, or a single Box-like space (treated
        as the ``"state"`` key). Pass the *single-env* space for batched
        simulators (see :func:`resolve_auto_dims_from_env`).
    act_space:
        Per-env action space (Box-like with ``shape``, or Discrete-like ``n``).

    Notes
    -----
    Encoders are matched to observation keys with exactly the runtime routing
    rules of :func:`srl.utils.obs_remap.apply_obs_remap` (explicit
    ``input_name`` first, then name match, single-obs broadcast, order zip).
    ``mlp``/``lstm``-style encoders get ``input_dim = prod(shape)``; encoders
    whose type requires ``input_shape`` (``cnn``) get the CHW shape. Heads that
    need an action dimension get ``action_dim`` from ``act_space``.

    Explicit values are kept but cross-checked: a mismatch with the env raises
    :class:`~srl.config.ConfigError` naming the encoder/head and both values.
    """
    from srl.registry.config_schema import AgentModelConfig
    from srl.registry.registry import EncoderRegistry, HeadRegistry
    from srl.utils.obs_remap import apply_obs_remap

    if isinstance(model_cfg, Mapping):
        model_cfg = AgentModelConfig.from_dict(dict(model_cfg))
    cfg = copy.deepcopy(model_cfg)
    errors: list[str] = []

    if obs_space is not None and cfg.encoders:
        shapes = space_shapes(obs_space)
        names = list(dict.fromkeys(e.name for e in cfg.encoders))
        input_names = {e.name: e.input_name for e in cfg.encoders}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                routed = apply_obs_remap(dict(shapes), names, input_names)
        except KeyError as exc:
            raise ConfigError(
                f"cannot resolve encoder inputs: {exc.args[0]} Available observation keys: "
                f"{sorted(shapes)}"
            ) from exc

        for i, enc in enumerate(cfg.encoders):
            path = f"encoders[{i}] ('{enc.name}')"
            shape = routed.get(enc.name)
            if shape is None:
                if enc.input_dim is None and enc.input_shape is None:
                    errors.append(
                        f"{path}: no observation key routes to this encoder (obs keys: "
                        f"{sorted(shapes)}); set input_name or an explicit input_dim/input_shape"
                    )
                continue
            try:
                entry = EncoderRegistry.entry(enc.type)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            if "input_shape" in entry.requires:
                env_shape = list(_to_chw(shape)) if len(shape) == 3 else list(shape)
                if enc.input_shape is None:
                    enc.input_shape = env_shape
                elif list(enc.input_shape) != env_shape:
                    errors.append(
                        f"{path}: input_shape {list(enc.input_shape)} does not match the "
                        f"environment observation shape {env_shape} (CHW). Fix it or set "
                        "input_shape: auto"
                    )
            elif "input_dim" in entry.requires:
                flat = int(math.prod(shape)) if shape else 1
                if enc.input_dim is None:
                    enc.input_dim = flat
                elif int(enc.input_dim) != flat:
                    errors.append(
                        f"{path}: input_dim {enc.input_dim} does not match the environment "
                        f"observation size {flat} (shape {list(shape)}). Fix it or set "
                        "input_dim: auto"
                    )

    if act_space is not None:
        env_action_dim = action_dim_of(act_space)
        for role in ("actor", "critic"):
            head = getattr(cfg, role)
            if head is None:
                continue
            try:
                entry = HeadRegistry.entry(head.type)
            except Exception as exc:
                errors.append(f"{role}: {exc}")
                continue
            if not entry.requires_action_dim and head.action_dim is None:
                continue  # e.g. a value head: no action input
            if head.action_dim is None:
                head.action_dim = env_action_dim
            elif int(head.action_dim) != env_action_dim:
                errors.append(
                    f"{role}.action_dim {head.action_dim} does not match the environment "
                    f"action dimension {env_action_dim}. Fix it or set action_dim: auto"
                )

    if errors:
        raise ConfigError(errors)
    return cfg


def resolve_auto_dims_from_env(model_cfg: Any, env: Any) -> Any:
    """:func:`resolve_auto_dims` using an SRL env's per-env spaces.

    Uses ``single_obs_space`` / ``single_act_space`` when the env exposes them
    (vector / Isaac Lab / mjlab envs batch ``obs_space``/``act_space``) and
    otherwise strips a leading ``num_envs`` axis from batched shapes.
    """
    num_envs = int(getattr(env, "num_envs", 1) or 1)

    def _single(space: Any) -> Any:
        if space is None or num_envs <= 1:
            return space
        shapes = space_shapes(space)
        if all(len(s) >= 2 and s[0] == num_envs for s in shapes.values()):
            stripped = {k: s[1:] for k, s in shapes.items()}
            if getattr(space, "spaces", None) is None and not isinstance(space, Mapping):
                return stripped["state"]
            return stripped
        return space

    obs_space = getattr(env, "single_obs_space", None) or _single(getattr(env, "obs_space", None))
    act_space = getattr(env, "single_act_space", None) or _single(getattr(env, "act_space", None))
    return resolve_auto_dims(model_cfg, obs_space, act_space)
