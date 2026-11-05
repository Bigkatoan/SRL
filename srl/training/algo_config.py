"""Algorithm-config construction from a YAML dict (shared by CLI and Python API).

Moved out of ``srl/cli/train.py``. The functions here never exit the
process: invalid configurations raise :class:`ConfigurationError`, and only
``srl-train``'s ``main()`` turns that into an exit code.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from typing import Any

__all__ = [
    "ConfigurationError",
    "build_algo_config",
    "configure_her_from_env",
    "her_goal_obs_parts",
    "make_algo_config",
    "resolve_algo_config_cls",
    "validate_algo_model_compatibility",
]


class ConfigurationError(ValueError):
    """An algorithm/env/model configuration that cannot be trained."""


def coerce_config_value(value):
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in lowered for ch in (".", "e")):
            return float(lowered)
        return int(lowered)
    except ValueError:
        return value


def build_algo_config(config_cls, train_cfg: dict, **extra_overrides):
    kwargs = {}
    field_names = {field.name for field in fields(config_cls)}
    for key, value in train_cfg.items():
        if key in field_names:
            kwargs[key] = coerce_config_value(value)
    for key, value in extra_overrides.items():
        if key in field_names and value is not None:
            kwargs[key] = value
    return config_cls(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Visual (auxiliary-encoder-loss) config auto-detection
# ──────────────────────────────────────────────────────────────────────────────
#
# ``VisualPPOConfig`` / ``VisualSACConfig`` are what switch the algorithms onto
# their auxiliary-encoder-loss code paths (separate encoder optimizer, aux loss
# on every update).  Nothing in the YAML schema names those classes, so the CLI
# picks one by looking at what the config actually asks for:
#
#   1. any encoder declares an ``aux_type`` that ``srl.registry.builder``
#      actually builds an aux head for, or
#   2. the ``train:`` block sets a field that only exists on the Visual variant
#      (``encoder_lr``, ``aux_loss_type``, ``aux_weight``, ...).
#
# Both signals are already-valid YAML, so existing configs light up with no
# edits and no new syntax to learn.

# Encoder-level ``aux_type`` -> algorithm-level ``aux_loss_type``.  Keys are the
# authoritative set from ``srl.registry.builder``'s aux-head dispatch; anything
# else builds no aux head at all.
_AUX_TYPE_TO_LOSS: dict[str, str] = {
    "autoencoder": "ae",
    "contrastive": "curl",
    "byol": "byol",
}


def _visual_config_for_algo(algo_name: str):
    """Return the Visual* config class for *algo_name*, or None if it has none."""
    from srl.core.config import VisualPPOConfig, VisualSACConfig

    return {"ppo": VisualPPOConfig, "sac": VisualSACConfig}.get(algo_name.lower())


def _visual_only_fields(visual_cls, base_cls) -> set[str]:
    """Fields that exist on the Visual variant but not on the plain config."""
    return {f.name for f in fields(visual_cls)} - {f.name for f in fields(base_cls)}


def _encoder_aux_types(raw_cfg: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split declared encoder ``aux_type``s into (recognized, unrecognized).

    Each entry is ``(encoder_name, aux_type)``.  Unrecognized values build no
    aux head in the model builder, so they must not be treated as a signal --
    they are reported to the user instead (usually a typo).
    """
    recognized: list[tuple[str, str]] = []
    unrecognized: list[tuple[str, str]] = []
    for index, enc in enumerate(raw_cfg.get("encoders") or []):
        if not isinstance(enc, dict):
            continue
        aux_type = enc.get("aux_type")
        if aux_type is None:
            continue
        name = str(enc.get("name") or f"encoder[{index}]")
        aux_type = str(aux_type).strip().lower()
        if aux_type in _AUX_TYPE_TO_LOSS:
            recognized.append((name, aux_type))
        else:
            unrecognized.append((name, aux_type))
    return recognized, unrecognized


def _warn_unusable_aux(algo_name: str, raw_cfg: dict) -> list[tuple[str, str]]:
    """Report aux requests that will not do anything; return the usable ones.

    Covers the two ways an ``aux_type`` silently does nothing: an unrecognized
    value (no head is built at all), and an algorithm with no Visual config
    variant (head is built, but nothing trains it).  Both warn rather than
    fail -- the run is otherwise valid.
    """
    recognized, unrecognized = _encoder_aux_types(raw_cfg)

    for name, aux_type in unrecognized:
        print(
            f"[srl-train] Warning: encoder '{name}' sets aux_type: {aux_type!r}, which is "
            f"not a known aux head ({', '.join(sorted(_AUX_TYPE_TO_LOSS))}). No auxiliary "
            "head will be built and no auxiliary loss will run.",
            file=sys.stderr,
        )

    if recognized and _visual_config_for_algo(algo_name) is None:
        names = ", ".join(f"'{n}' (aux_type: {t})" for n, t in recognized)
        print(
            f"[srl-train] Warning: {names} requests an auxiliary encoder loss, but "
            f"{algo_name.upper()} has no visual config variant, so the auxiliary loss "
            "will not be trained. Use PPO or SAC for auxiliary encoder losses.",
            file=sys.stderr,
        )
    return recognized


def _warn_aux_loss_prerequisites(
    algo_name: str,
    raw_cfg: dict,
    recognized: list[tuple[str, str]],
    resolved_loss: str | None,
) -> None:
    """Warn when the selected aux loss cannot run against the built model.

    Both cases below produce a silently-zero aux loss at runtime rather than an
    error, so they are worth catching while the config is still on screen.
    """
    # PPO's aux path is reconstruction-only -- it ignores aux_loss_type and
    # feeds every aux head into reconstruction_loss(). A projection head
    # (contrastive/byol) fails the shape check there and gets skipped.
    if algo_name.lower() == "ppo":
        non_ae = [(n, t) for n, t in recognized if t != "autoencoder"]
        if non_ae:
            names = ", ".join(f"'{n}' (aux_type: {t})" for n, t in non_ae)
            print(
                f"[srl-train] Warning: {names} -- PPO's auxiliary loss only implements "
                "reconstruction, so this head will be skipped at runtime. Use "
                "aux_type: autoencoder with PPO, or switch to SAC for "
                "contrastive/BYOL objectives.",
                file=sys.stderr,
            )
        return

    # CURL/BYOL encode the positive view through an EMA target copy, which only
    # exists when the encoder sets use_momentum: true.
    if resolved_loss in ("curl", "byol"):
        has_momentum = any(
            isinstance(enc, dict) and enc.get("use_momentum")
            for enc in (raw_cfg.get("encoders") or [])
        )
        if not has_momentum:
            print(
                f"[srl-train] Warning: aux_loss_type '{resolved_loss}' needs a momentum "
                "encoder, but no encoder sets use_momentum: true. The auxiliary loss will "
                "not run. Add use_momentum: true to the visual encoder.",
                file=sys.stderr,
            )


def resolve_algo_config_cls(
    algo_name: str,
    base_cls,
    raw_cfg: dict,
    train_cfg: dict,
) -> tuple[type, dict]:
    """Choose between the plain and the Visual algorithm config.

    Returns ``(config_cls, extra_defaults)``.  ``extra_defaults`` carries values
    inferred from the model config (currently ``aux_loss_type``) and is only
    ever populated with keys the user did not set explicitly in ``train:``.
    """
    recognized = _warn_unusable_aux(algo_name, raw_cfg)

    visual_cls = _visual_config_for_algo(algo_name)
    if visual_cls is None:
        return base_cls, {}

    visual_fields = _visual_only_fields(visual_cls, base_cls)
    explicit_fields = sorted(visual_fields & set(train_cfg))

    if not recognized and not explicit_fields:
        return base_cls, {}

    extra_defaults: dict = {}
    if recognized and "aux_loss_type" not in train_cfg:
        # Keep the algorithm's aux loss consistent with the head that was
        # actually built: an "ae" loss needs a decoder, "curl"/"byol" need a
        # projection head.  Defaulting to the config-class default ("curl")
        # against an autoencoder head silently computes nothing.
        distinct = {t for _, t in recognized}
        if len(distinct) > 1:
            print(
                f"[srl-train] Warning: multiple encoder aux_types declared ({sorted(distinct)}); "
                f"using '{recognized[0][1]}' from encoder '{recognized[0][0]}'. Set "
                "aux_loss_type in the train: block to choose explicitly.",
                file=sys.stderr,
            )
        extra_defaults["aux_loss_type"] = _AUX_TYPE_TO_LOSS[recognized[0][1]]

    resolved_loss = extra_defaults.get("aux_loss_type", train_cfg.get("aux_loss_type"))
    _warn_aux_loss_prerequisites(algo_name, raw_cfg, recognized, resolved_loss)

    reasons = []
    if recognized:
        reasons.append("encoder aux_type: " + ", ".join(f"{n}={t}" for n, t in recognized))
    if explicit_fields:
        reasons.append("train: " + ", ".join(explicit_fields))
    print(
        f"[srl-train] Auxiliary encoder loss detected ({'; '.join(reasons)}) "
        f"-> using {visual_cls.__name__}"
        + (f" (aux_loss_type={resolved_loss!r})" if resolved_loss else "")
    )
    return visual_cls, extra_defaults


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


def validate_algo_model_compatibility(
    raw_cfg: dict, algo_name: str, config_path: str
) -> str | None:
    actor_type = ((raw_cfg.get("actor") or {}).get("type") or "").lower()
    critic_type = ((raw_cfg.get("critic") or {}).get("type") or "").lower()
    configured_algo = (raw_cfg.get("algo") or "").lower()

    compatible_heads = {
        "ppo": ({"gaussian"}, {"value"}),
        "a2c": ({"gaussian"}, {"value"}),
        "a3c": ({"gaussian"}, {"value"}),
        "sac": ({"squashed_gaussian"}, {"twin_q"}),
        "ddpg": ({"deterministic"}, {"q", "q_function", "twin_q"}),
        "td3": ({"deterministic"}, {"twin_q"}),
    }

    expected = compatible_heads.get(algo_name.lower())
    if expected is None:
        return None

    valid_actor_types, valid_critic_types = expected
    if actor_type in valid_actor_types and critic_type in valid_critic_types:
        return None

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
# One-call construction
# ──────────────────────────────────────────────────────────────────────────────

_ON_POLICY_NUM_ENVS = {"ppo", "a2c"}


def make_algo_config(
    algo_name: str,
    raw_cfg: dict,
    *,
    env: Any = None,
    env_type: str = "flat",
    n_envs: int | None = None,
    overrides: dict | None = None,
):
    """Build the algorithm config dataclass for *algo_name* from a YAML dict.

    Reads the ``train:`` block (unknown keys are ignored, as before), picks
    the Visual* config variant when the model asks for auxiliary encoder
    losses, and fills in the env-derived fields: ``num_envs`` (PPO/A2C),
    ``n_workers`` (A3C), ``action_dim``/``replay_num_envs`` (SAC/TD3/DDPG)
    and the HER goal split (``use_her``). *overrides* win over the YAML.
    """
    from srl.algorithms import get_algorithm

    algo_name = algo_name.lower()
    _, base_cls = get_algorithm(algo_name)
    train_cfg = dict(raw_cfg.get("train") or {})
    if overrides:
        train_cfg.update({k: v for k, v in overrides.items() if v is not None})

    if _visual_config_for_algo(algo_name) is None:
        _warn_unusable_aux(algo_name, raw_cfg)
        cfg_cls, aux_defaults = base_cls, {}
    else:
        cfg_cls, aux_defaults = resolve_algo_config_cls(algo_name, base_cls, raw_cfg, train_cfg)

    num_envs = getattr(env, "num_envs", None) if env is not None else None
    if num_envs is None:
        num_envs = n_envs if n_envs is not None else int(train_cfg.get("n_envs", 1))
    extra: dict[str, Any] = dict(aux_defaults)
    if algo_name in _ON_POLICY_NUM_ENVS:
        extra["num_envs"] = int(num_envs)
    elif algo_name == "a3c":
        extra["n_workers"] = int(n_envs if n_envs is not None else train_cfg.get("n_envs", 1))
    elif env is not None:
        from srl.training.envs import env_action_dim

        extra["action_dim"] = env_action_dim(env)
        extra["replay_num_envs"] = int(num_envs)
    algo_cfg = build_algo_config(cfg_cls, train_cfg, **extra)
    if env is not None:
        # HER needs the goal split and the env's sparse reward fn, neither of
        # which can come from YAML -- resolve them before the buffer is built.
        configure_her_from_env(algo_cfg, env, env_type)
    return algo_cfg
