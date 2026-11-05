"""Env-dependent glue between a validated experiment config and an agent.

The strict YAML -> dataclass construction lives in :mod:`srl.config`
(:func:`~srl.config.load_experiment`, :func:`~srl.config.build_algo_config`).
This module only holds what needs the *environment* or the algorithm
registry on top of that:

* :func:`env_derived_overrides` -- ``num_envs`` / ``n_workers`` /
  ``action_dim`` / ``replay_num_envs`` taken from the built env;
* :func:`configure_her_from_env` / :func:`her_goal_obs_parts` -- HER's goal split;
* :func:`validate_algo_model_compatibility` -- actor/critic head types vs. the
  algorithm's :class:`~srl.registry.AlgorithmSpec`;
* :func:`report_aux_config` -- human-readable warnings about auxiliary
  encoder losses that would silently do nothing.

Nothing here exits the process: problems raise
:class:`~srl.config.ConfigError` (``ConfigurationError`` is its old name).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from srl.config._validate import ConfigError

__all__ = [
    "ConfigurationError",
    "configure_her_from_env",
    "env_derived_overrides",
    "her_goal_obs_parts",
    "report_aux_config",
    "validate_algo_model_compatibility",
]

#: Backwards-compatible name of :class:`srl.config.ConfigError`.
ConfigurationError = ConfigError

_AUX_TYPE_TO_LOSS: dict[str, str] = {"autoencoder": "ae", "contrastive": "curl", "byol": "byol"}


def _spec(algo_name: str):
    from srl.config.algos import algorithm_spec

    return algorithm_spec(algo_name)


# ──────────────────────────────────────────────────────────────────────────────
# Env-derived algorithm fields
# ──────────────────────────────────────────────────────────────────────────────


def env_derived_overrides(
    algo_name: str, env: Any, *, n_envs: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Algorithm-config fields that come from the built env, not the YAML.

    Pass the result as ``overrides=`` to :func:`srl.config.build_algo_config`
    (keys the config class does not have are skipped there). ``n_envs`` is
    the requested env count; it becomes A3C's ``n_workers`` (A3C trains on
    one env per worker, so its own env is never vectorized). ``seed`` (the
    trainer-level ``train.seed``) fills algorithm configs that carry their
    own ``seed`` field (A3C's per-worker base seed).
    """
    from srl.training.envs import env_action_dim

    out: dict[str, Any] = {}
    if env is not None:
        num_envs = int(getattr(env, "num_envs", None) or 1)
        out["num_envs"] = num_envs
        out["replay_num_envs"] = num_envs
        out["action_dim"] = env_action_dim(env)
    elif n_envs is not None:
        out["num_envs"] = out["replay_num_envs"] = int(n_envs)
    if n_envs is not None and str(algo_name).lower() == "a3c":
        out["n_workers"] = int(n_envs)
    if seed is not None:
        out["seed"] = int(seed)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm / model-head compatibility
# ──────────────────────────────────────────────────────────────────────────────


def _head_type(cfg: Any, role: str) -> str:
    head = cfg.get(role) if isinstance(cfg, Mapping) else getattr(cfg, role, None)
    if isinstance(head, Mapping):
        value = head.get("type")
    else:
        value = getattr(head, "type", None)
    return str(value or "").lower()


def validate_algo_model_compatibility(raw_cfg: Any, algo_name: str, config_path: str) -> str | None:
    """An error message when the YAML's actor/critic heads cannot run *algo_name*.

    *raw_cfg* is the YAML mapping or a parsed ``AgentModelConfig``. The
    accepted head types come from the algorithm's registered
    :class:`~srl.registry.AlgorithmSpec` (``actor_types`` / ``critic_types``);
    a spec that declares none (e.g. a plugin) accepts anything.
    """
    spec = _spec(algo_name)
    valid_actor_types = set(spec.actor_types)
    valid_critic_types = set(spec.critic_types)
    actor_type = _head_type(raw_cfg, "actor")
    critic_type = _head_type(raw_cfg, "critic")
    actor_ok = not valid_actor_types or actor_type in valid_actor_types
    critic_ok = not valid_critic_types or critic_type in valid_critic_types
    if actor_ok and critic_ok:
        return None
    configured_algo = str(
        (raw_cfg.get("algo") if isinstance(raw_cfg, Mapping) else None) or ""
    ).lower()
    configured_msg = f" config declares algo '{configured_algo}' and" if configured_algo else ""
    return (
        f"Config '{config_path}' is not compatible with --algo {algo_name}:"
        f"{configured_msg} uses actor='{actor_type or 'missing'}', "
        f"critic='{critic_type or 'missing'}'. "
        f"Expected actor in {sorted(valid_actor_types)} and "
        f"critic in {sorted(valid_critic_types)}. "
        "Use a matching YAML config for the selected algorithm or omit "
        "--algo to use the config's declared algorithm."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Auxiliary encoder loss diagnostics
# ──────────────────────────────────────────────────────────────────────────────
#
# srl.config.resolve_algo_config_cls picks VisualPPOConfig / VisualSACConfig
# (the classes that switch PPO/SAC onto their aux-loss code paths) when an
# encoder declares a recognized aux_type or train: sets a Visual-only field.
# It prints nothing; srl-train calls report_aux_config() so the cases where an
# aux request silently does nothing are visible on the console.


def _encoder_aux_types(raw_cfg: Any) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Declared encoder ``aux_type``s split into (recognized, unrecognized)."""
    recognized: list[tuple[str, str]] = []
    unrecognized: list[tuple[str, str]] = []
    encoders = raw_cfg.get("encoders") if isinstance(raw_cfg, Mapping) else None
    for index, enc in enumerate(encoders or []):
        if not isinstance(enc, Mapping) or enc.get("aux_type") is None:
            continue
        name = str(enc.get("name") or f"encoder[{index}]")
        aux_type = str(enc["aux_type"]).strip().lower()
        (recognized if aux_type in _AUX_TYPE_TO_LOSS else unrecognized).append((name, aux_type))
    return recognized, unrecognized


def _warn(message: str) -> None:
    print(f"[srl-train] Warning: {message}", file=sys.stderr)


def report_aux_config(algo_name: str, raw_cfg: Mapping[str, Any], algo_cfg: Any = None) -> None:
    """Print warnings for auxiliary-loss requests that will not do anything.

    * an unrecognized encoder ``aux_type`` (no head is built);
    * an ``aux_type`` on an algorithm without a ``Visual*`` config (the head
      is built but nothing trains it);
    * several distinct ``aux_type``s (the first one wins);
    * PPO with a projection head (PPO's aux loss is reconstruction-only);
    * CURL/BYOL without a momentum encoder.

    With *algo_cfg* (the built config), also prints which ``Visual*`` class
    and ``aux_loss_type`` were selected.
    """
    algo_name = str(algo_name).lower()
    recognized, unrecognized = _encoder_aux_types(raw_cfg)
    for name, aux_type in unrecognized:
        _warn(
            f"encoder '{name}' sets aux_type: {aux_type!r}, which is not a known aux head "
            f"({', '.join(sorted(_AUX_TYPE_TO_LOSS))}). No auxiliary head will be built "
            "and no auxiliary loss will run."
        )
    if not recognized:
        return
    names = ", ".join(f"'{n}' (aux_type: {t})" for n, t in recognized)
    if _spec(algo_name).visual_config_cls is None:
        _warn(
            f"{names} requests an auxiliary encoder loss, but {algo_name.upper()} has no "
            "visual config variant, so the auxiliary loss will not be trained. Use PPO or "
            "SAC for auxiliary encoder losses."
        )
        return
    distinct = {t for _, t in recognized}
    if len(distinct) > 1:
        _warn(
            f"multiple encoder aux_types declared ({sorted(distinct)}); using "
            f"'{recognized[0][1]}' from encoder '{recognized[0][0]}'. Set aux_loss_type in "
            "the train: block to choose explicitly."
        )
    if algo_name == "ppo":
        non_ae = [(n, t) for n, t in recognized if t != "autoencoder"]
        if non_ae:
            _warn(
                ", ".join(f"'{n}' (aux_type: {t})" for n, t in non_ae)
                + " -- PPO's auxiliary loss only implements reconstruction, so this head "
                "will be skipped at runtime. Use aux_type: autoencoder with PPO, or switch "
                "to SAC for contrastive/BYOL objectives."
            )
    resolved_loss = getattr(algo_cfg, "aux_loss_type", None)
    if resolved_loss in ("curl", "byol"):
        encoders = raw_cfg.get("encoders") or []
        if not any(isinstance(e, Mapping) and e.get("use_momentum") for e in encoders):
            _warn(
                f"aux_loss_type '{resolved_loss}' needs a momentum encoder, but no encoder "
                "sets use_momentum: true. The auxiliary loss will not run. Add "
                "use_momentum: true to the visual encoder."
            )
    if algo_cfg is not None:
        print(
            f"[srl-train] Auxiliary encoder loss detected ({names}) -> using "
            f"{type(algo_cfg).__name__}"
            + (f" (aux_loss_type={resolved_loss!r})" if resolved_loss else "")
        )


# ──────────────────────────────────────────────────────────────────────────────
# HER
# ──────────────────────────────────────────────────────────────────────────────


def her_goal_obs_parts(goal_obs: dict):
    """Split a raw GoalEnv observation dict into HER's (obs, ag, dg) vectors.

    ``GoalEnvWrapper``'s flat observation is
    ``[observation | achieved_goal | desired_goal]``, and ``HERReplayBuffer``
    re-appends the (possibly relabelled) desired goal at sample time. So the
    part stored as "obs" is ``[observation | achieved_goal]`` -- that way the
    sampled batch has exactly the layout, and dimension, the encoders were
    built for.
    """
    import numpy as np

    obs = np.asarray(goal_obs["observation"], dtype=np.float32).ravel()
    ag = np.asarray(goal_obs["achieved_goal"], dtype=np.float32).ravel()
    dg = np.asarray(goal_obs["desired_goal"], dtype=np.float32).ravel()
    return np.concatenate([obs, ag]), ag, dg


def configure_her_from_env(algo_cfg, env, env_type: str) -> None:
    """Fill the env-derived HER fields on an algo config, in place.

    No-op unless ``use_her`` is set. Raises if the run is not goal-conditioned:
    HER needs ``achieved_goal``/``desired_goal`` and the env's own sparse
    reward function, which only a GoalEnv exposes.
    """
    import numpy as np

    if not getattr(algo_cfg, "use_her", False):
        return

    if env_type != "goal":
        raise ConfigurationError(
            'use_her=True requires env_type: "goal" (a '
            "gymnasium-robotics GoalEnv such as FetchReach-v4); got "
            f'env_type: "{env_type}".'
        )
    if not getattr(env, "include_goal", True):
        raise ConfigurationError(
            "use_her=True is incompatible with include_goal: false -- "
            "HER re-appends the desired goal to every sampled observation, which "
            "only matches the model input when the flat obs includes the goal."
        )
    if getattr(env, "num_envs", 1) != 1:
        raise ConfigurationError(
            "use_her=True currently supports a single environment "
            f"only (got num_envs={getattr(env, 'num_envs', 1)}); HER stores whole "
            "episodes and the CLI collects them from one un-vectorized env."
        )

    unwrapped = getattr(env, "unwrapped", env)
    space = getattr(unwrapped, "observation_space", None)
    try:
        obs_dim = int(np.prod(space["observation"].shape))
        ag_dim = int(np.prod(space["achieved_goal"].shape))
        dg_dim = int(np.prod(space["desired_goal"].shape))
    except (TypeError, KeyError) as exc:
        raise ConfigurationError(
            "use_her=True but the environment's observation space is "
            "not a GoalEnv Dict with 'observation'/'achieved_goal'/'desired_goal' "
            f"keys: {exc}"
        ) from exc

    reward_fn = getattr(unwrapped, "compute_reward", None)
    if not callable(reward_fn):
        raise ConfigurationError(
            "use_her=True but the environment exposes no "
            "compute_reward(achieved_goal, desired_goal, info); HER cannot "
            "recompute rewards for relabelled goals without it."
        )

    algo_cfg.her_obs_dim = obs_dim + ag_dim
    algo_cfg.her_goal_dim = dg_dim
    algo_cfg.her_reward_fn = reward_fn
    print(
        f"[srl-train] HER enabled: strategy={algo_cfg.her_strategy} "
        f"ratio={algo_cfg.her_ratio} obs_dim={algo_cfg.her_obs_dim} "
        f"goal_dim={dg_dim} max_episode_len={algo_cfg.her_max_episode_len}"
    )
