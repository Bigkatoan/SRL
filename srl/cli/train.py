"""CLI entry point: srl-train --config path/to/config.yaml [options]"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import fields

from srl.utils.spaces import sample_action_space as _sample_action_space


def _make_cli_env(
    env_name: str, device: str, n_envs: int, env_type: str = "flat", render: bool = False
):
    """Build an env for training/eval, or (``render=True``) for the
    ``--visualize`` live viewer's plain-Gymnasium path (flat/goal/racecar
    only -- isaaclab/mjlab visualization goes through
    ``srl.utils.live_viewer`` instead, since they need a differently-built
    raw env; ``render`` is a no-op for those two branches here).
    """
    import gymnasium as gym

    from srl.envs.goal_env_wrapper import GoalEnvWrapper
    from srl.envs.gymnasium_wrapper import GymnasiumWrapper
    from srl.envs.racecar_wrapper import RacecarWrapper

    normalized_env_type = (env_type or "flat").strip().lower()
    normalized_env_name = _normalize_env_name(env_name, normalized_env_type)
    make_kwargs = {"render_mode": "human"} if render else {}

    if normalized_env_type == "isaaclab" or normalized_env_name.startswith("isaaclab:"):
        from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

        task_name = normalized_env_name.split(":", 1)[1]
        import isaaclab_tasks  # noqa: F401
        from isaaclab.envs import ManagerBasedRLEnv

        env_cfg = isaaclab_tasks.utils.parse_env_cfg(task_name, device=device, num_envs=n_envs)
        base_env = ManagerBasedRLEnv(cfg=env_cfg)
        return IsaacLabWrapper(base_env)

    if normalized_env_type == "mjlab" or normalized_env_name.startswith("mjlab:"):
        # mjlab (github.com/mujocolab/mjlab) is a MuJoCo-Warp GPU-batched
        # env stack that mirrors Isaac Lab's ManagerBasedRLEnv API closely
        # enough that IsaacLabWrapper works unchanged -- no real Isaac Sim/
        # isaaclab install needed, which is the whole point of this branch:
        # same "one process, num_envs already batched on GPU" shape, without
        # the multi-GB Omniverse runtime. Task lookup goes through mjlab's
        # own registry (mjlab.tasks.registry), populated by auto-importing
        # every package registered under the "mjlab.tasks" entry-point group
        # -- the same mechanism `import mjlab` itself runs at import time, so
        # any task package installed alongside mjlab (e.g. a project's own
        # `your_pkg.tasks` registered via
        # `[project.entry-points."mjlab.tasks"]` in its pyproject.toml) is
        # already discoverable here with no extra wiring.
        from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

        task_name = normalized_env_name.split(":", 1)[1]
        import mjlab  # noqa: F401  (side effect: discovers installed mjlab.tasks packages)
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        env_cfg = load_env_cfg(task_name)
        env_cfg.scene.num_envs = n_envs
        base_env = ManagerBasedRlEnv(env_cfg, device=device)
        return IsaacLabWrapper(base_env)

    if normalized_env_type == "goal":
        import gymnasium_robotics

        gymnasium_robotics.register_robotics_envs()
        return GoalEnvWrapper(gym.make(normalized_env_name, **make_kwargs))

    if normalized_env_type == "racecar":
        import racecar_gym  # noqa: F401

        return RacecarWrapper(gym.make(normalized_env_name, **make_kwargs))

    base = gym.make(normalized_env_name, **make_kwargs)
    return GymnasiumWrapper(base)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="srl-train",
        description="SRL — train an RL agent from a YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  srl-train --config configs/ppo_state.yaml --env HalfCheetah-v5 --steps 1000000
  srl-train --config configs/sac_state.yaml --env Ant-v5 --steps 3000000 --device cuda
        """,
    )
    p.add_argument("--config", required=True, help="Path to the YAML model config file")
    p.add_argument(
        "--env",
        required=False,
        default=None,
        help="Gymnasium environment id, 'isaaclab:<task>', or 'mjlab:<task>'",
    )
    p.add_argument(
        "--algo",
        default=None,
        help="Algorithm override: ppo|sac|ddpg|td3|a2c|a3c (auto-detected from config)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total environment steps (defaults to train.total_steps or 1M)",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Parallel environments (defaults to train.n_envs or 1)",
    )
    p.add_argument("--device", default="auto", help="PyTorch device: cpu|cuda|auto (default: auto)")
    p.add_argument(
        "--vec-mode",
        choices=["auto", "sync", "async"],
        default="auto",
        help="Vector env backend for n-envs > 1",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    p.add_argument("--logdir", default="runs", help="TensorBoard log dir (default: runs/)")
    p.add_argument(
        "--ckptdir", default="checkpoints", help="Checkpoint directory (default: checkpoints/)"
    )
    p.add_argument(
        "--log-interval", type=int, default=2048, help="Console/logging interval in env steps"
    )
    p.add_argument(
        "--episode-window", type=int, default=20, help="Rolling window for episode summaries"
    )
    p.add_argument(
        "--console-metrics",
        type=int,
        default=8,
        help="Maximum metrics shown in compact terminal summaries",
    )
    p.add_argument(
        "--console-layout",
        choices=["multi_line", "single_line"],
        default="multi_line",
        help="Terminal logging layout",
    )
    p.add_argument(
        "--plot-metrics", default="", help="Comma-separated metric tags to visualize after training"
    )
    p.add_argument(
        "--no-plots", action="store_true", help="Disable plot export at the end of training"
    )
    p.add_argument(
        "--resume", default=None, help="Resume training from a checkpoint created by srl-train"
    )
    p.add_argument(
        "--save-model-pipeline",
        nargs="?",
        const="auto",
        default=None,
        help="Save model pipeline PNG before training",
    )
    p.add_argument(
        "--save-training-pipeline",
        nargs="?",
        const="auto",
        default=None,
        help="Save training pipeline PNG before training",
    )
    p.add_argument(
        "--export-pipeline-only",
        action="store_true",
        help="Render requested pipeline PNGs and exit without training",
    )
    p.add_argument("--eval-freq", type=int, default=50_000, help="Evaluation frequency in steps")
    p.add_argument("--eval-episodes", type=int, default=10, help="Episodes per evaluation")
    p.add_argument("--render", action="store_true", help="Render environment during eval")
    p.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Run one extra single env in the background doing live inference "
            "with the current (training) model, rendered while training runs. "
            "mjlab uses its own interactive viewer; flat/goal/racecar envs "
            "open a render window. Not supported for isaaclab."
        ),
    )
    return p


def _coerce_config_value(value):
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


def _train_section(config_path: str) -> dict:
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("train", {}), data


def _resolve_env_name(cli_env: str | None, raw_cfg: dict) -> str:
    env_name = cli_env or raw_cfg.get("env_id") or (raw_cfg.get("train") or {}).get("env_id")
    if not env_name:
        raise ValueError("Environment id is required. Pass --env or set env_id in the YAML config.")
    return env_name


def _resolve_env_type(raw_cfg: dict) -> str:
    env_type = raw_cfg.get("env_type") or (raw_cfg.get("train") or {}).get("env_type") or "flat"
    return str(env_type).strip().lower()


def _normalize_env_name(env_name: str, env_type: str) -> str:
    if env_type not in ("isaaclab", "mjlab"):
        return env_name
    prefix = f"{env_type}:"
    if env_name.startswith(prefix):
        return env_name
    return f"{prefix}{env_name}"


def _resolve_env_spec(cli_env: str | None, raw_cfg: dict) -> tuple[str, str]:
    env_type = _resolve_env_type(raw_cfg)
    env_name = _resolve_env_name(cli_env, raw_cfg)
    return _normalize_env_name(env_name, env_type), env_type


def _resolve_pipeline_outputs(
    raw_cfg: dict,
    *,
    run_name: str,
    logdir: str,
    cli_model_path: str | None,
    cli_training_path: str | None,
    export_only: bool,
) -> tuple[str | None, str | None]:
    visualization_cfg = raw_cfg.get("visualization") or {}
    model_path = cli_model_path
    training_path = cli_training_path

    if model_path is None and visualization_cfg.get("save_model_pipeline"):
        model_path = visualization_cfg.get("model_pipeline_path") or "auto"
    if training_path is None and visualization_cfg.get("save_training_pipeline"):
        training_path = visualization_cfg.get("training_pipeline_path") or "auto"

    if export_only and model_path is None and training_path is None:
        model_path = "auto"
        training_path = "auto"

    base_dir = os.path.join(logdir, run_name)
    if model_path == "auto":
        model_path = os.path.join(base_dir, "model_pipeline.png")
    if training_path == "auto":
        training_path = os.path.join(base_dir, "training_pipeline.png")
    return model_path, training_path


def _build_algo_config(config_cls, train_cfg: dict, **extra_overrides):
    kwargs = {}
    field_names = {field.name for field in fields(config_cls)}
    for key, value in train_cfg.items():
        if key in field_names:
            kwargs[key] = _coerce_config_value(value)
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


def _resolve_algo_config_cls(
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


def _validate_algo_model_compatibility(
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


def _next_eval_step(start_step: int, eval_freq: int) -> int | None:
    if eval_freq <= 0:
        return None
    if start_step <= 0:
        return eval_freq
    return int(math.floor(start_step / eval_freq) + 1) * eval_freq


def _evaluate_agent(
    agent, *, env_name: str, env_type: str, device: str, seed: int, episodes: int, render: bool
) -> dict[str, float]:
    import numpy as np

    eval_env = _make_cli_env(env_name, device, 1, env_type)
    episode_scores: list[float] = []
    episode_lengths: list[int] = []
    success_values: list[float] = []
    encoder_names = list(agent.model.encoders.keys())

    try:
        for episode_index in range(max(int(episodes), 1)):
            obs, _ = eval_env.reset(seed=seed + episode_index)
            done = False
            truncated = False
            score = 0.0
            length = 0

            while not (done or truncated):
                obs_remapped = _remap_obs_to_encoders(
                    obs,
                    encoder_names,
                    encoder_input_names=getattr(agent.model, "encoder_input_names", None),
                )
                obs_t = _obs_to_tensors(obs_remapped, agent.device, force_batch=True)
                action, _, _, _ = agent.predict(obs_t, deterministic=True)
                action_np = action.detach().cpu().numpy()
                # isaaclab/mjlab envs always expect a batched (num_envs,
                # action_dim) action, even at num_envs=1 -- IsaacLabWrapper
                # never unbatches, unlike a plain gymnasium.Env. Squeezing
                # here for those env types produces a 1-D action and
                # `action.shape[1]` in the env's action manager raises
                # IndexError.
                if (
                    env_type not in ("isaaclab", "mjlab")
                    and action_np.ndim > 1
                    and action_np.shape[0] == 1
                ):
                    action_np = action_np.squeeze(0)
                next_obs, reward, done, truncated, info = eval_env.step(action_np)
                score += float(np.asarray(reward).reshape(-1)[0])
                length += 1
                obs = next_obs
                if render and hasattr(eval_env, "render"):
                    try:
                        eval_env.render()
                    except Exception:
                        pass
                for key in ("is_success", "success"):
                    if isinstance(info, dict) and key in info:
                        try:
                            success_values.append(float(np.asarray(info[key]).reshape(-1)[0]))
                        except Exception:
                            pass
            episode_scores.append(score)
            episode_lengths.append(length)
    finally:
        eval_env.close()

    metrics = {
        "eval/score_mean": float(sum(episode_scores) / len(episode_scores)),
        "eval/score_max": float(max(episode_scores)),
        "eval/episode_length_mean": float(sum(episode_lengths) / len(episode_lengths)),
        "eval/episodes": float(len(episode_scores)),
    }
    if success_values:
        metrics["eval/success_mean"] = float(sum(success_values) / len(success_values))
    return metrics


def _maybe_run_evaluation(
    agent, args, logger, *, device: str, step: int, next_eval_step: int | None
) -> int | None:
    if next_eval_step is None or step < next_eval_step:
        return next_eval_step

    eval_freq = int(getattr(args, "eval_freq", 0))

    eval_metrics = _evaluate_agent(
        agent,
        env_name=args.env,
        env_type=getattr(args, "env_type", "flat"),
        device=device,
        seed=args.seed + 10_000,
        episodes=getattr(args, "eval_episodes", 1),
        render=bool(getattr(args, "render", False)),
    )
    logger.record_metrics(
        eval_metrics, step=step, total_steps=args.steps, prefix=None, console=False
    )
    print(
        f"[eval] step {step} | score_mean={eval_metrics['eval/score_mean']:.4f} "
        f"| episodes={int(eval_metrics['eval/episodes'])}",
        flush=True,
    )
    return next_eval_step + eval_freq


def _final_eval_already_ran_at(next_eval_step: int | None, eval_freq: int, step: int) -> bool:
    """True iff the in-loop periodic eval already ran at exactly `step`.

    `_maybe_run_evaluation` advances `next_eval_step` by exactly `eval_freq`
    when it fires, and leaves it unchanged otherwise -- so "an eval just ran
    at this exact step" is `next_eval_step == step + eval_freq`, and nothing
    else. (`step < next_eval_step` is NOT a safe proxy for this: it's true
    immediately after *any* eval fires, by construction, so using it as an
    alternate "already ran" check here previously produced a duplicate eval
    + duplicate `[eval]` log line whenever `total_steps` was an exact
    multiple of `eval_freq`.)
    """
    return next_eval_step is not None and next_eval_step - eval_freq == step


def _maybe_start_visualizer(agent, args, device: str):
    """Start the ``--visualize`` background viewer, if requested.

    Returns a ``srl.utils.live_viewer.VisualizerHandle`` (or None if
    ``--visualize`` wasn't passed, isn't supported for this env type, or
    failed to start). Callers should ``.stop()`` and ``.thread.join(...)``
    the handle before the process exits -- see that module's docstring for
    why. Never raises: visualization is a nice-to-have and a failure here (a
    missing optional dependency, an env that can't be constructed twice,
    whatever) must not take training down with it.
    """
    if not getattr(args, "visualize", False):
        return None

    encoder_names = list(agent.model.encoders.keys())
    encoder_input_names = getattr(agent.model, "encoder_input_names", None)

    def _remap(obs):
        return _remap_obs_to_encoders(obs, encoder_names, encoder_input_names=encoder_input_names)

    if args.env_type == "isaaclab" or args.env.startswith("isaaclab:"):
        print(
            "[srl-train] --visualize: not supported for isaaclab envs (a single "
            "process hosts one Isaac Sim render context, shared by every env in "
            "it, so a second view-only env can't be added alongside headless "
            "training envs) -- skipping.",
            file=sys.stderr,
        )
        return None

    from srl.utils.live_viewer import start_gym_visualizer, start_mjlab_visualizer

    if args.env_type == "mjlab" or args.env.startswith("mjlab:"):
        task_name = args.env.split(":", 1)[1]
        handle = start_mjlab_visualizer(agent, task_name, device, remap_obs_fn=_remap)
        if handle is not None and int(getattr(args, "eval_freq", 0)) > 0:
            # mjlab's env construction does a one-time CUDA graph capture
            # (mujoco_warp's Simulation.create_graph()). The visualizer
            # thread steps its own env continuously on the GPU for the rest
            # of the run; periodic in-training eval builds a brand-new mjlab
            # env every `--eval-freq` steps. If that new env's graph-capture
            # window overlaps with the visualizer's ongoing stepping, the
            # capture gets corrupted -- confirmed via a real repro: training
            # ran cleanly through 9+ eval-triggered env rebuilds with
            # --visualize off, and crashed (`Warp CUDA error 901: operation
            # failed due to a previous error during capture`, hard SIGABRT)
            # at the second eval cycle with it on. Disabling eval removes
            # the only other mjlab-env-construction call site left once
            # training is running, which removes the race entirely --
            # you're already watching the live policy via the viewer, so
            # scripted eval episodes are redundant anyway.
            args.eval_freq = 0
            print(
                "[srl-train] --visualize: disabling periodic evaluation for this "
                "mjlab run (--eval-freq forced to 0). Evaluating while the "
                "background viewer keeps stepping its own env can corrupt "
                "mjlab's CUDA graph capture and crash training -- watch the "
                "live viewer instead of scripted eval episodes.",
            )
        return handle

    return start_gym_visualizer(
        agent,
        lambda: _make_cli_env(args.env, device, 1, args.env_type, render=True),
        remap_obs_fn=_remap,
        obs_to_tensor_fn=lambda obs, dev: _obs_to_tensors(obs, dev, force_batch=True),
    )


def _stop_visualizer(handle) -> None:
    """Best-effort graceful shutdown of a `--visualize` handle before the
    process exits -- see `srl.utils.live_viewer`'s module docstring for why
    this matters (an abruptly-killed daemon thread mid-GPU-call can
    segfault/core-dump on interpreter shutdown)."""
    if handle is None:
        return
    try:
        handle.stop()
        handle.thread.join(timeout=5.0)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    import os
    import random

    import numpy as np
    import torch

    from srl.core.config import A2CConfig, A3CConfig, DDPGConfig, PPOConfig, SACConfig, TD3Config
    from srl.registry.builder import ModelBuilder
    from srl.utils.pipeline_graph import render_pipeline_bundle

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ── reproducibility ───────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ── build model ───────────────────────────────────────────────────────────
    print(f"[srl-train] Loading config: {args.config}")
    train_cfg, raw_cfg = _train_section(args.config)

    # ── infer algorithm from config filename ─────────────────────────────────
    algo_name = args.algo
    if algo_name is None:
        algo_name = raw_cfg.get("algo")
    if algo_name is None:
        cfg_lower = os.path.basename(args.config).lower()
        for a in ("td3", "sac", "ddpg", "a3c", "a2c", "ppo"):
            if a in cfg_lower:
                algo_name = a
                break
        if algo_name is None:
            algo_name = "ppo"
    args.steps = (
        args.steps if args.steps is not None else int(train_cfg.get("total_steps", 1_000_000))
    )
    args.n_envs = args.n_envs if args.n_envs is not None else int(train_cfg.get("n_envs", 1))
    args.env, args.env_type = _resolve_env_spec(args.env, raw_cfg)

    compatibility_error = _validate_algo_model_compatibility(raw_cfg, algo_name, args.config)
    if compatibility_error is not None:
        print(f"[srl-train] {compatibility_error}", file=sys.stderr)
        return 2

    model = ModelBuilder.from_yaml(args.config)
    print(f"[srl-train] Algorithm: {algo_name.upper()}")

    # PPO/SAC do this inside their own branch (it also selects the config
    # class); the remaining algorithms only need the warning.
    if _visual_config_for_algo(algo_name) is None:
        _warn_unusable_aux(algo_name, raw_cfg)

    run_name = f"{algo_name}_{os.path.splitext(os.path.basename(args.config))[0]}"
    model_pipeline_path, training_pipeline_path = _resolve_pipeline_outputs(
        raw_cfg,
        run_name=run_name,
        logdir=args.logdir,
        cli_model_path=args.save_model_pipeline,
        cli_training_path=args.save_training_pipeline,
        export_only=args.export_pipeline_only,
    )
    pipeline_outputs = render_pipeline_bundle(
        raw_cfg,
        config_path=args.config,
        algo_name=algo_name,
        env_name=args.env,
        model_output_path=model_pipeline_path,
        training_output_path=training_pipeline_path,
    )
    for name, path in pipeline_outputs.items():
        print(f"[srl-train] Saved {name} pipeline: {path}")
    if args.export_pipeline_only:
        return 0

    # ── Isaac Sim bootstrap (must happen before any omni.* import) ───────────
    # SimulationApp is a singleton; initialise it once per process when using
    # any isaaclab env type.  All subsequent isaaclab imports are safe after this.
    if args.env_type == "isaaclab" or args.env.startswith("isaaclab:"):
        try:
            import atexit

            # Use IsaacLab's AppLauncher (not raw SimulationApp) so it loads
            # the headless-rendering kit file which:
            #   1. Sets /isaaclab/cameras_enabled = true  (required by TiledCamera)
            #   2. Activates omni.replicator.core  (required for camera data)
            from isaaclab.app import AppLauncher

            _isaac_app_launcher = AppLauncher(headless=True, enable_cameras=True)
            _isaac_sim_app = _isaac_app_launcher.app
            atexit.register(_isaac_sim_app.close)
            # Set the asset-root BEFORE any isaaclab sub-module is imported; the
            # constant NUCLEUS_ASSET_ROOT_DIR is evaluated at **module load time**
            # in isaaclab/utils/assets.py.  If the carb setting is None at that
            # point the path becomes "None/…" and USD loading fails.
            import carb as _carb

            _carb_settings = _carb.settings.get_settings()
            if not _carb_settings.get("/persistent/isaac/asset_root/cloud"):
                _carb_settings.set(
                    "/persistent/isaac/asset_root/cloud",
                    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1",
                )
            print("[srl-train] Isaac Sim initialized (headless)")
        except Exception as _e:
            print(f"[srl-train] WARNING: Isaac Sim could not be initialized: {_e}", file=sys.stderr)

    # ── build environment ─────────────────────────────────────────────────────
    from srl.envs.async_vector_env import AsyncVectorEnv
    from srl.envs.sync_vector_env import SyncVectorEnv

    def _make_env(seed_offset=0):
        return _make_cli_env(args.env, device, args.n_envs, args.env_type)

    print(f"[srl-train] Creating {args.n_envs} × {args.env}")
    uses_internal_vectorization = (
        args.env_type in ("isaaclab", "mjlab")
        or args.env.startswith("isaaclab:")
        or args.env.startswith("mjlab:")
    )
    if uses_internal_vectorization or args.n_envs == 1:
        env = _make_env()
    else:
        env_fns = [lambda i=i: _make_env(i) for i in range(args.n_envs)]
        if args.vec_mode == "sync":
            env = SyncVectorEnv(env_fns)
        elif args.vec_mode == "async":
            env = AsyncVectorEnv(env_fns)
        else:
            env = AsyncVectorEnv(env_fns) if args.n_envs > 1 else SyncVectorEnv(env_fns)

    # ── build agent ───────────────────────────────────────────────────────────
    from srl.utils.callbacks import CheckpointCallback
    from srl.utils.checkpoint import CheckpointManager
    from srl.utils.logger import Logger, LoggerConfig

    plot_metrics = [metric.strip() for metric in args.plot_metrics.split(",") if metric.strip()]
    logger = Logger(
        log_dir=os.path.join(args.logdir, run_name),
        verbose=True,
        config=LoggerConfig(
            console_interval=args.log_interval,
            episode_window=args.episode_window,
            enable_plots=not args.no_plots,
            plot_metrics=plot_metrics or None,
            max_console_metrics=args.console_metrics,
            console_layout=args.console_layout,
        ),
    )
    logger.set_metadata(
        algorithm=algo_name,
        env=args.env,
        config=args.config,
        device=device,
        total_steps=args.steps,
        seed=args.seed,
        n_envs=args.n_envs,
        vec_mode=args.vec_mode,
        env_type=args.env_type,
    )
    logger.configure_env(getattr(env, "num_envs", args.n_envs))
    cm = CheckpointManager(os.path.join(args.ckptdir, run_name), max_keep=5)

    import copy

    action_dim = int(np.prod(getattr(env.act_space, "shape", ()) or (1,)))

    start_step = 0
    visualizer_handle = None

    if algo_name == "ppo":
        from srl.algorithms.ppo import PPO

        ppo_cfg_cls, ppo_aux_defaults = _resolve_algo_config_cls(
            algo_name, PPOConfig, raw_cfg, train_cfg
        )
        agent = PPO(
            model,
            config=_build_algo_config(
                ppo_cfg_cls,
                train_cfg,
                num_envs=getattr(env, "num_envs", args.n_envs),
                **ppo_aux_defaults,
            ),
            device=device,
        )
        visualizer_handle = _maybe_start_visualizer(agent, args, device)
        callbacks = [CheckpointCallback(cm, save_interval=100_000, model=agent)]
        if args.resume:
            start_step = int(cm.load(agent, args.resume, device=device).get("step", 0))
            print(f"[srl-train] Resuming from step {start_step}: {args.resume}")
        _run_on_policy(agent, env, args, callbacks, logger, start_step=start_step, device=device)

    elif algo_name == "a2c":
        from srl.algorithms.a2c import A2C

        agent = A2C(
            model,
            config=_build_algo_config(
                A2CConfig, train_cfg, num_envs=getattr(env, "num_envs", args.n_envs)
            ),
            device=device,
        )
        visualizer_handle = _maybe_start_visualizer(agent, args, device)
        callbacks = [CheckpointCallback(cm, save_interval=100_000, model=agent)]
        if args.resume:
            start_step = int(cm.load(agent, args.resume, device=device).get("step", 0))
            print(f"[srl-train] Resuming from step {start_step}: {args.resume}")
        _run_on_policy(agent, env, args, callbacks, logger, start_step=start_step, device=device)

    elif algo_name == "a3c":
        from functools import partial

        from srl.algorithms.a3c import A3C

        agent = A3C(
            model,
            config=_build_algo_config(A3CConfig, train_cfg, n_workers=args.n_envs),
            device=device,
        )
        visualizer_handle = _maybe_start_visualizer(agent, args, device)
        agent.train(
            total_timesteps=args.steps,
            env_fn=partial(_make_cli_env, args.env, device, args.n_envs, args.env_type),
            logger=logger,
            log_interval=args.log_interval,
        )

    elif algo_name == "sac":
        from srl.algorithms.sac import SAC

        target = copy.deepcopy(model)
        sac_cfg_cls, sac_aux_defaults = _resolve_algo_config_cls(
            algo_name, SACConfig, raw_cfg, train_cfg
        )
        agent = SAC(
            model,
            target,
            config=_build_algo_config(
                sac_cfg_cls,
                train_cfg,
                action_dim=action_dim,
                replay_num_envs=getattr(env, "num_envs", 1),
                **sac_aux_defaults,
            ),
            device=device,
        )
        visualizer_handle = _maybe_start_visualizer(agent, args, device)
        callbacks = [CheckpointCallback(cm, save_interval=100_000, model=agent)]
        if args.resume:
            start_step = int(cm.load(agent, args.resume, device=device).get("step", 0))
            print(f"[srl-train] Resuming from step {start_step}: {args.resume}")
        _run_off_policy(agent, env, args, callbacks, logger, start_step=start_step, device=device)

    elif algo_name == "ddpg":
        from srl.algorithms.ddpg import DDPG

        target = copy.deepcopy(model)
        agent = DDPG(
            model,
            target,
            config=_build_algo_config(
                DDPGConfig,
                train_cfg,
                action_dim=action_dim,
                replay_num_envs=getattr(env, "num_envs", 1),
            ),
            device=device,
        )
        visualizer_handle = _maybe_start_visualizer(agent, args, device)
        callbacks = [CheckpointCallback(cm, save_interval=100_000, model=agent)]
        if args.resume:
            start_step = int(cm.load(agent, args.resume, device=device).get("step", 0))
            print(f"[srl-train] Resuming from step {start_step}: {args.resume}")
        _run_off_policy(agent, env, args, callbacks, logger, start_step=start_step, device=device)

    elif algo_name == "td3":
        from srl.algorithms.td3 import TD3

        target = copy.deepcopy(model)
        agent = TD3(
            model,
            target,
            config=_build_algo_config(
                TD3Config,
                train_cfg,
                action_dim=action_dim,
                replay_num_envs=getattr(env, "num_envs", 1),
            ),
            device=device,
        )
        visualizer_handle = _maybe_start_visualizer(agent, args, device)
        callbacks = [CheckpointCallback(cm, save_interval=100_000, model=agent)]
        if args.resume:
            start_step = int(cm.load(agent, args.resume, device=device).get("step", 0))
            print(f"[srl-train] Resuming from step {start_step}: {args.resume}")
        _run_off_policy(agent, env, args, callbacks, logger, start_step=start_step, device=device)

    else:
        print(f"[srl-train] Unknown algorithm: {algo_name}", file=sys.stderr)
        return 1

    _stop_visualizer(visualizer_handle)
    cm.save(agent if algo_name != "a3c" else model, step=args.steps, tag="final")
    logger.set_step(args.steps)
    logger.close()
    env.close()
    print("[srl-train] Done.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for mapping obs keys to encoder names
# ──────────────────────────────────────────────────────────────────────────────


def _remap_obs_to_encoders(
    obs_dict: dict,
    encoder_names: list[str],
    encoder_input_names: dict[str, str | None] | None = None,
) -> dict:
    """Map observation dict keys → encoder input names.

    How the model receives multi-modal observations (image + vector)
    ----------------------------------------------------------------
    The obs dict returned by the environment must ultimately have keys that
    match the encoder names defined in the YAML config.  Three cases:

    Case 1 — keys already match (most explicit, always recommended):
        env returns  {"cnn_enc": <(N,3,H,W) image>,  "mlp_enc": <(N,8) state>}
        YAML encoders: cnn_enc (type: cnn), mlp_enc (type: mlp)
        → passthrough, no remapping needed.

    Case 2 — single obs, single encoder (Isaac Lab default for image-only envs):
        env returns  {"policy": <(N,3,H,W) image>}
        YAML encoder: policy_enc (type: cnn)
        → rename "policy" → "policy_enc" automatically.

    Case 3 — multiple obs groups, multiple encoders, same count:
        env returns  {"policy": <image>, "critic":  <state>}         (2 keys)
        YAML encoders: policy_enc (cnn), critic_enc (mlp)            (2 encoders)
        → zip by order: "policy" → policy_enc, "critic" → critic_enc.
        NOTE: order matters here.  Name your encoders so they match
        the env's obs group names to avoid relying on dict order.

    Matching rules applied in order:
      0. Encoder has input_name set       → route by that explicit obs key.
      1. Any obs key already equals an encoder name → passthrough.
      2. 1 obs, 1 encoder                 → rename obs key to encoder name.
      3. N obs, N encoders (N > 1)        → zip obs values to encoder names.
      4. Anything else                    → passthrough (model handles it).

    Validation:
      - Missing explicit input_name key   → KeyError.
      - Unused obs keys after explicit routing → warnings.warn.
    """
    from srl.utils.obs_remap import apply_obs_remap

    return apply_obs_remap(obs_dict, encoder_names, encoder_input_names)


def _obs_to_tensors(obs_dict: dict, device, *, force_batch: bool) -> dict:
    """``force_batch=True`` means: this obs dict came from a single env step
    that *may or may not* already carry an explicit leading batch dim --
    isaaclab/mjlab always pre-batch (even at num_envs=1: vectors come back
    (1, D), images (1, C, H, W)), while a plain gymnasium/goal/racecar env
    never does (vectors (D,), images (C, H, W)).

    Both unbatched shapes have odd ndim (1 for a vector, 3 for an image) and
    both pre-batched shapes have even ndim (2, 4) -- so parity of ndim tells
    us whether to add the batch dim, regardless of whether this is a flat
    vector or an image. (A shape[0]-based heuristic here previously
    mistook an unbatched (C, H, W) image's channel count for an existing
    batch size and skipped expanding it, crashing the first real eval/
    visualize step against any CNN-encoder config.)
    """
    import numpy as np
    import torch

    tensor_obs = {}
    for key, value in obs_dict.items():
        arr = np.asarray(value)
        if force_batch and (arr.ndim == 0 or arr.ndim % 2 == 1):
            arr = np.expand_dims(arr, axis=0)
        tensor_obs[key] = torch.from_numpy(arr).float().to(device)
    return tensor_obs


def _split_vector_transition(
    obs: dict, next_obs: dict, action, reward, done, trunc
) -> list[tuple[dict, dict, object, float, bool, bool]]:
    """Split a batched (vectorized-env) step into per-env transition tuples.

    `done`/`trunc` are kept SEPARATE in the returned tuples (last two fields:
    terminated, truncated) rather than OR'd into one -- off-policy bootstrap
    targets (`(1 - terminated) * next_q` in sac.py/ddpg.py/td3.py) must see
    true termination only. Callers that want "episode ended for any reason"
    (resets, episode-length logging) OR these two themselves at the point
    they need it; do not re-combine them here.
    """
    import numpy as np

    rewards = np.asarray(reward, dtype=np.float32).reshape(-1)
    dones = np.asarray(done, dtype=bool).reshape(-1)
    truncs = np.asarray(trunc, dtype=bool).reshape(-1)
    actions = np.asarray(action)
    if actions.ndim == 1:
        actions = np.expand_dims(actions, axis=0)

    transitions = []
    for index in range(len(rewards)):
        obs_i = {k: np.asarray(v)[index] for k, v in obs.items()}
        next_obs_i = {k: np.asarray(v)[index] for k, v in next_obs.items()}
        transitions.append(
            (
                obs_i,
                next_obs_i,
                np.asarray(actions[index], dtype=np.float32),
                float(rewards[index]),
                bool(dones[index]),
                bool(truncs[index]),
            )
        )
    return transitions


# ──────────────────────────────────────────────────────────────────────────────
# Training loops
# ──────────────────────────────────────────────────────────────────────────────


def _run_on_policy(
    agent, env, args, callbacks, logger, *, start_step: int = 0, device: str = "cpu"
) -> None:
    import numpy as np

    n_steps = agent.cfg.n_steps
    obs, _ = env.reset(seed=args.seed)
    step = start_step
    next_eval_step = _next_eval_step(start_step, int(getattr(args, "eval_freq", 0)))
    encoder_names = list(agent.model.encoders.keys())

    while step < args.steps:
        remaining_steps = max(args.steps - step, 0)
        rollout_steps = min(
            n_steps,
            max(1, math.ceil(remaining_steps / max(getattr(agent.cfg, "num_envs", 1), 1))),
        )
        for _ in range(rollout_steps):
            obs_remapped = _remap_obs_to_encoders(
                obs,
                encoder_names,
                encoder_input_names=getattr(agent.model, "encoder_input_names", None),
            )
            obs_t = _obs_to_tensors(obs_remapped, agent.device, force_batch=False)
            action, log_prob, value, _ = agent.predict(obs_t)
            action_np = action.cpu().numpy()
            next_obs, reward, done, trunc, info = env.step(action_np)
            logger.update_episodes(reward, done, trunc, step=step, info=info)
            agent.buffer.add(
                obs=obs,
                action=action_np,
                reward=np.asarray(reward),
                done=np.asarray(done),
                log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                value=value.cpu().numpy() if value is not None else None,
            )
            obs = next_obs
            step += getattr(agent.cfg, "num_envs", 1)

        obs_remapped_final = _remap_obs_to_encoders(
            obs,
            encoder_names,
            encoder_input_names=getattr(agent.model, "encoder_input_names", None),
        )
        last_t = _obs_to_tensors(obs_remapped_final, agent.device, force_batch=False)
        _, _, last_val, _ = agent.predict(last_t)
        agent.buffer.compute_returns_and_advantages(
            last_value=last_val.cpu().numpy() if last_val is not None else 0.0
        )
        metrics = agent.update()
        logger.set_step(step)
        logger.record_metrics(metrics, step=step, total_steps=args.steps)
        for cb in callbacks:
            cb.on_step_end(step, metrics)
        next_eval_step = _maybe_run_evaluation(
            agent, args, logger, device=device, step=step, next_eval_step=next_eval_step
        )

    eval_freq = int(getattr(args, "eval_freq", 0))
    if next_eval_step is not None and not _final_eval_already_ran_at(
        next_eval_step, eval_freq, step
    ):
        _maybe_run_evaluation(agent, args, logger, device=device, step=step, next_eval_step=step)


def _run_off_policy(
    agent, env, args, callbacks, logger, *, start_step: int = 0, device: str = "cpu"
) -> None:
    import numpy as np

    # ------------------------------------------------------------------
    # Async / GPU-buffer fast path (v0.2.0)
    # Activated when train config carries AsyncRunnerConfig with
    # use_async=True or use_gpu_buffer=True.
    # ------------------------------------------------------------------
    _runner_cfg = getattr(args, "runner_cfg", None)
    if _runner_cfg is None:
        # Check if algo_config dict has runner_cfg keys
        _algo_cfg = getattr(args, "algo_config", {}) or {}
        if isinstance(_algo_cfg, dict) and (
            _algo_cfg.get("use_async") or _algo_cfg.get("use_gpu_buffer")
        ):
            from srl.core.config import AsyncRunnerConfig

            _runner_cfg = AsyncRunnerConfig(
                use_async=bool(_algo_cfg.get("use_async", False)),
                use_gpu_buffer=bool(_algo_cfg.get("use_gpu_buffer", False)),
            )

    if _runner_cfg is not None and (_runner_cfg.use_async or _runner_cfg.use_gpu_buffer):
        from srl.runners import AsyncOffPolicyRunner

        _random_steps = getattr(agent.cfg, "start_steps", None) or getattr(
            agent.cfg, "learning_starts", 10_000
        )
        _update_after = getattr(agent.cfg, "update_after", None) or getattr(
            agent.cfg, "learning_starts", 10_000
        )
        _update_every = getattr(agent.cfg, "update_every", None) or getattr(
            agent.cfg, "train_freq", 1
        )
        _gradient_steps = max(int(getattr(agent.cfg, "gradient_steps", 1)), 1)

        def _log_fn(step: int, metrics: dict) -> None:
            logger.set_step(step)
            logger.record_metrics(metrics, step=step, total_steps=args.steps)
            for cb in callbacks:
                cb.on_step_end(step, metrics)

        runner = AsyncOffPolicyRunner(
            agent=agent,
            env=env,
            total_steps=args.steps - start_step,
            runner_cfg=_runner_cfg,
            log_fn=_log_fn,
            device=device,
            random_steps=int(_random_steps),
            update_after=int(_update_after),
            update_every=int(_update_every),
            gradient_steps=_gradient_steps,
        )
        runner.run()
        return

    random_steps = getattr(agent.cfg, "start_steps", None)
    if random_steps is None:
        random_steps = getattr(agent.cfg, "learning_starts", 10_000)
    update_after = getattr(agent.cfg, "update_after", None)
    if update_after is None:
        update_after = getattr(agent.cfg, "learning_starts", 10_000)
    update_every = getattr(agent.cfg, "update_every", None)
    if update_every is None:
        update_every = getattr(agent.cfg, "train_freq", 1)

    random_steps = max(int(random_steps), 0)
    update_after = max(int(update_after), 0)
    update_every = max(int(update_every), 1)
    obs, _ = env.reset(seed=args.seed)
    encoder_names = list(agent.model.encoders.keys())
    vectorized_env = (
        args.env_type in ("isaaclab", "mjlab")
        or args.env.startswith("isaaclab:")
        or args.env.startswith("mjlab:")
        or getattr(env, "num_envs", 1) > 1
    )
    step_increment = getattr(env, "num_envs", 1) if vectorized_env else 1
    env_step = start_step
    since_last_update = 0
    next_eval_step = _next_eval_step(start_step, int(getattr(args, "eval_freq", 0)))

    while env_step < args.steps:
        remaining_steps = max(args.steps - env_step, 0)
        active_envs = min(step_increment, remaining_steps) if vectorized_env else 1
        obs_remapped = _remap_obs_to_encoders(
            obs,
            encoder_names,
            encoder_input_names=getattr(agent.model, "encoder_input_names", None),
        )
        obs_t = _obs_to_tensors(obs_remapped, agent.device, force_batch=not vectorized_env)
        if env_step < random_steps:
            if vectorized_env:
                # `env.act_space` on isaaclab/mjlab envs is already the
                # BATCHED (num_envs, action_dim) space -- use
                # `single_act_space` (one env's space) per sample, else
                # stacking N samples of the already-batched space produces
                # an (N, num_envs, action_dim) array instead of
                # (num_envs, action_dim).
                per_env_space = getattr(env, "single_act_space", env.act_space)
                action_np = np.stack(
                    [
                        _sample_action_space(per_env_space)
                        for _ in range(getattr(env, "num_envs", 1))
                    ],
                    axis=0,
                )
            else:
                action_np = _sample_action_space(env.act_space)
        else:
            action, _, _, _ = agent.predict(obs_t)
            action_np = action.cpu().numpy()
            if not vectorized_env and action_np.ndim > 1 and action_np.shape[0] == 1:
                action_np = action_np.squeeze(0)

        next_obs, reward, done, trunc, info = env.step(action_np)
        env_step += active_envs
        since_last_update += active_envs
        log_reward = reward
        log_done = done
        log_trunc = trunc
        log_info = info
        if vectorized_env and active_envs < step_increment:
            log_reward = np.asarray(reward)[:active_envs]
            log_done = np.asarray(done)[:active_envs]
            log_trunc = np.asarray(trunc)[:active_envs]
            log_info = list(info)[:active_envs]
        logger.update_episodes(log_reward, log_done, log_trunc, step=env_step, info=log_info)
        if vectorized_env:
            transitions = _split_vector_transition(
                obs,
                next_obs,
                action_np,
                reward,
                done,
                trunc,
            )[:active_envs]
            for env_index, (obs_i, next_obs_i, action_i, reward_i, done_i, trunc_i) in enumerate(
                transitions
            ):
                agent.buffer.add(
                    obs=obs_i,
                    action=action_i,
                    reward=np.array([reward_i], dtype=np.float32),
                    done=np.array([done_i], dtype=bool),
                    truncated=np.array([trunc_i], dtype=bool),
                    next_obs=next_obs_i,
                    env_idx=env_index,
                )
        else:
            agent.buffer.add(
                obs=obs,
                action=action_np,
                reward=np.array([reward], dtype=np.float32),
                done=np.array([done], dtype=bool),
                truncated=np.array([trunc], dtype=bool),
                next_obs=next_obs,
            )
        obs = next_obs
        if not vectorized_env and (done or trunc):
            obs, _ = env.reset()

        gradient_steps = max(int(getattr(agent.cfg, "gradient_steps", 1)), 1)
        if env_step >= update_after and since_last_update >= update_every:
            update_span = since_last_update
            metrics_list = []
            for _ in range(gradient_steps):
                metrics = agent.update()
                if metrics:
                    metrics_list.append(metrics)
            since_last_update = 0
            if metrics_list:
                sums: dict[str, float] = {}
                counts: dict[str, int] = {}
                for metric in metrics_list:
                    for key, value in metric.items():
                        sums[key] = sums.get(key, 0.0) + float(value)
                        counts[key] = counts.get(key, 0) + 1
                merged = {key: sums[key] / counts[key] for key in sums}
                merged["train/utd_ratio"] = gradient_steps / max(update_span, 1)
                logger.set_step(env_step)
                logger.record_metrics(merged, step=env_step, total_steps=args.steps)
                for cb in callbacks:
                    cb.on_step_end(env_step, merged)
        next_eval_step = _maybe_run_evaluation(
            agent, args, logger, device=device, step=env_step, next_eval_step=next_eval_step
        )

    eval_freq = int(getattr(args, "eval_freq", 0))
    if next_eval_step is not None and not _final_eval_already_ran_at(
        next_eval_step, eval_freq, env_step
    ):
        _maybe_run_evaluation(
            agent, args, logger, device=device, step=env_step, next_eval_step=env_step
        )


if __name__ == "__main__":
    sys.exit(main())
