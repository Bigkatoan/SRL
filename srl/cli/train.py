"""CLI entry point: ``srl-train --config path/to/config.yaml [options]``.

A thin layer over the Python API: parse flags, load the YAML, build the env
and the agent, then ``agent.learn()``. The training loops, evaluation,
callbacks and env construction live in :mod:`srl.training`; helpers there
raise exceptions, and only :func:`main` turns them into exit codes.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any

from srl.config._validate import ConfigError
from srl.training import envs as _envs

#: Old name of :class:`srl.config.ConfigError` (kept for imports from here).
ConfigurationError = ConfigError


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
        help="Algorithm: ppo|sac|ddpg|td3|a2c|a3c (default: the config's `algo:` key)",
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
    p.add_argument(
        "--device",
        default=None,
        help="PyTorch device: cpu|cuda|auto (default: train.device, else auto)",
    )
    p.add_argument(
        "--vec-mode",
        choices=["auto", "sync", "async"],
        default="auto",
        help="Vector env backend for n-envs > 1",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (default: train.seed, else 0)"
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Bit-reproducible PyTorch kernels: torch.use_deterministic_algorithms"
            "(True), cudnn.deterministic, no cudnn.benchmark, TF32 off, "
            "CUBLAS_WORKSPACE_CONFIG set. Slower on GPU. (default: "
            "train.deterministic, else False; CPU runs are reproducible from "
            "--seed alone)"
        ),
    )
    p.add_argument("--logdir", default="runs", help="TensorBoard log dir (default: runs/)")
    p.add_argument(
        "--ckptdir", default="checkpoints", help="Checkpoint directory (default: checkpoints/)"
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Console/logging interval in env steps (default: train.log_interval, else 2048)",
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
    p.add_argument(
        "--eval-freq",
        type=int,
        default=None,
        help="Evaluation frequency in steps (default: train.eval_freq, else 50000)",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Episodes per evaluation (default: train.eval_episodes, else 10)",
    )
    p.add_argument("--render", action="store_true", help="Render environment during eval")
    p.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save a best_*.pt checkpoint (alongside, never replacing, the "
            "existing periodic/final checkpoints) whenever eval/score_mean "
            "reaches a new run-best, tracked independently of --eval-freq's "
            "periodic checkpoint rotation. Off by default; can also be set "
            "via train.save_best in the YAML config (default: train.save_best, "
            "else False). Needs --eval-freq > 0 to have any effect, since "
            "eval is what produces eval/score_mean."
        ),
    )
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
    p.add_argument(
        "--visualize-backend",
        choices=["viser", "native"],
        default="viser",
        help=(
            "mjlab only: which of mjlab's own viewer classes '--visualize' uses. "
            "'viser' (default) is browser-based and works headless/remote -- the "
            "console prints a URL to open. 'native' opens a real desktop GLFW "
            "window on the training machine (needs a display reachable from this "
            "process). Ignored for flat/goal/racecar envs, which always open a "
            "render window."
        ),
    )
    return p


def _train_section(config_path: str) -> tuple[dict, dict]:
    """``(train block, whole YAML)`` of *config_path*, unvalidated (pipeline rendering)."""
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("train", {}), data


def _resolve_algo(cli_algo: str | None, raw_cfg: dict, config_path: str) -> str:
    """``--algo`` wins, then the YAML ``algo:`` key; never silently defaults.

    Guessing from the file name (``srl.config.infer_algo_name``) still works
    but warns.
    """
    from srl.config import infer_algo_name

    algo = cli_algo or raw_cfg.get("algo")
    if algo:
        return str(algo).strip().lower()
    guess = infer_algo_name(raw_cfg, config_path)
    if guess:
        message = (
            f"Config '{config_path}' has no `algo:` key; guessing '{guess}' from the "
            "file name is deprecated and will be removed. Add `algo: "
            f"{guess}` to the YAML or pass --algo {guess}."
        )
        warnings.warn(message, DeprecationWarning, stacklevel=2)
        print(f"[srl-train] Warning: {message}", file=sys.stderr)
        return guess
    raise ConfigurationError(
        f"Config '{config_path}' does not declare an algorithm: add `algo: <ppo|sac|ddpg|"
        "td3|a2c|a3c>` to the YAML or pass --algo."
    )


_TRAINER_FLAGS = {
    # args attribute -> TrainerConfig field
    "device": "device",
    "seed": "seed",
    "log_interval": "log_interval",
    "eval_freq": "eval_freq",
    "eval_episodes": "eval_episodes",
    "deterministic": "deterministic",
    "save_best": "save_best",
    "steps": "total_steps",
    "n_envs": "n_envs",
}


def _apply_train_defaults(args, trainer) -> Any:
    """CLI flag wins if passed, else the validated ``train:`` value (or its default).

    Returns the :class:`~srl.config.TrainerConfig` with the flags applied,
    which is what the resolved config dump records.
    """
    import dataclasses

    for name, key in _TRAINER_FLAGS.items():
        if getattr(args, name) is None:
            setattr(args, name, getattr(trainer, key))
    return dataclasses.replace(
        trainer, **{key: getattr(args, name) for name, key in _TRAINER_FLAGS.items()}
    )


def _maybe_start_visualizer(agent, args, device: str):
    """Start the ``--visualize`` background viewer, if requested.

    Returns a ``srl.utils.live_viewer.VisualizerHandle`` or None. Never
    raises: visualization is a nice-to-have and must not take training down.
    """
    if not getattr(args, "visualize", False):
        return None
    from srl.training._obs import encoder_layout, obs_to_tensors, remap_obs_to_encoders

    names, input_names = encoder_layout(agent)

    def _remap(obs):
        return remap_obs_to_encoders(obs, names, encoder_input_names=input_names)

    if args.env_type == "isaaclab" or args.env.startswith("isaaclab:"):
        print(
            "[srl-train] --visualize: not supported for isaaclab envs (one Isaac Sim "
            "render context per process, shared by every env in it) -- skipping.",
            file=sys.stderr,
        )
        return None

    from srl.utils.live_viewer import start_gym_visualizer, start_mjlab_visualizer

    if args.env_type == "mjlab" or args.env.startswith("mjlab:"):
        handle = start_mjlab_visualizer(
            agent,
            args.env.split(":", 1)[1],
            device,
            remap_obs_fn=_remap,
            backend=getattr(args, "visualize_backend", "viser"),
        )
        if handle is not None and int(getattr(args, "eval_freq", 0)) > 0:
            # Building a second mjlab env for eval while the viewer steps its
            # own corrupts mujoco_warp's CUDA graph capture (hard SIGABRT,
            # reproduced on a real run) -- the live viewer replaces eval.
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
        lambda: _envs.make_env(args.env, device, 1, args.env_type, render=True),
        remap_obs_fn=_remap,
        obs_to_tensor_fn=lambda obs, dev: obs_to_tensors(obs, dev, force_batch=True),
    )


def _stop_visualizer(handle) -> None:
    """Best-effort graceful shutdown (see ``srl.utils.live_viewer``)."""
    if handle is None:
        return
    try:
        handle.stop()
        handle.thread.join(timeout=5.0)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _train(args)
    except ConfigError as exc:
        print(f"[srl-train] {exc}", file=sys.stderr)
        return 2


def _train(args) -> int:
    from functools import partial

    import torch

    from srl.config import dump_resolved_config, load_experiment
    from srl.core.agent_io import build_agent_from_experiment
    from srl.core.base_agent import OffPolicyAgent
    from srl.registry.registry import AlgorithmRegistry
    from srl.training.algo_config import report_aux_config, validate_algo_model_compatibility
    from srl.training.callbacks import CheckpointCallback
    from srl.training.evaluation import Evaluator, seed_best_tracker
    from srl.utils.checkpoint import BestCheckpointTracker, CheckpointManager
    from srl.utils.logger import Logger, LoggerConfig
    from srl.utils.pipeline_graph import render_pipeline_bundle, resolve_pipeline_outputs
    from srl.utils.seeding import set_random_seed

    print(f"[srl-train] Loading config: {args.config}")
    _, raw_cfg = _train_section(args.config)
    algo_name = _resolve_algo(args.algo, raw_cfg, args.config)
    if algo_name not in AlgorithmRegistry:
        print(
            f"[srl-train] Unknown algorithm: {algo_name} "
            f"(available: {AlgorithmRegistry.available()})",
            file=sys.stderr,
        )
        return 1
    compatibility_error = validate_algo_model_compatibility(raw_cfg, algo_name, args.config)
    if compatibility_error is not None:
        raise ConfigurationError(compatibility_error)
    exp = load_experiment(args.config, algo=algo_name)
    trainer_cfg = _apply_train_defaults(args, exp.trainer)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(args.seed, deterministic=args.deterministic)
    if device == "cuda" and not args.deterministic:
        # TF32 trades precision for speed; only when bit-reproducibility is off.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    args.env_type = exp.env.env_type
    env_name = args.env or exp.env.env_id
    if not env_name:
        raise ConfigurationError(
            "Environment id is required. Pass --env or set env_id in the YAML config."
        )
    args.env = _envs.normalize_env_name(env_name, args.env_type)

    print(f"[srl-train] Algorithm: {algo_name.upper()}")
    run_name = f"{algo_name}_{os.path.splitext(os.path.basename(args.config))[0]}"
    model_png, training_png = resolve_pipeline_outputs(
        raw_cfg,
        run_name=run_name,
        logdir=args.logdir,
        cli_model_path=args.save_model_pipeline,
        cli_training_path=args.save_training_pipeline,
        export_only=args.export_pipeline_only,
    )
    for name, path in render_pipeline_bundle(
        raw_cfg,
        config_path=args.config,
        algo_name=algo_name,
        env_name=args.env,
        model_output_path=model_png,
        training_output_path=training_png,
    ).items():
        print(f"[srl-train] Saved {name} pipeline: {path}")
    if args.export_pipeline_only:
        return 0

    if args.env_type == "isaaclab" or args.env.startswith("isaaclab:"):
        _envs.bootstrap_isaac_sim()  # must precede any omni.* import
    is_a3c = algo_name == "a3c"
    print(f"[srl-train] Creating {args.n_envs} × {args.env}")
    # A3C's n_envs is its WORKER count; each worker builds its own single env.
    env = _envs.build_training_env(
        args.env,
        device=device,
        n_envs=1 if is_a3c else args.n_envs,
        env_type=args.env_type,
        vec_mode=args.vec_mode,
        seed=args.seed,
    )

    plot_metrics = [m.strip() for m in args.plot_metrics.split(",") if m.strip()]
    run_dir = os.path.join(args.logdir, run_name)
    logger = Logger(
        log_dir=run_dir,
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
    # --save-best: its own CheckpointManager (max_keep=1) so periodic saves
    # never evict the best checkpoint (see BestCheckpointTracker).
    best_cm = best_tracker = None
    if args.save_best:
        best_cm = CheckpointManager(os.path.join(args.ckptdir, run_name), max_keep=1)
        best_tracker = BestCheckpointTracker(mode="max")
        if args.resume:
            seed_best_tracker(best_cm, best_tracker, device)

    # Env before model: `auto` dims are resolved from the env's spaces.
    import dataclasses

    exp = dataclasses.replace(
        exp, trainer=trainer_cfg, env=dataclasses.replace(exp.env, env_id=env_name)
    )
    agent = build_agent_from_experiment(
        exp,
        env,
        device=device,
        n_envs=args.n_envs if is_a3c else None,
        source=args.config,
    )
    report_aux_config(algo_name, raw_cfg, agent.cfg)
    resolved_path = os.path.join(run_dir, "config.yaml")
    dump_resolved_config(agent._experiment, resolved_path)
    print(f"[srl-train] Resolved config: {resolved_path}")
    visualizer_handle = _maybe_start_visualizer(agent, args, device)

    if is_a3c:
        agent.learn(
            args.steps,
            env_fn=partial(_envs.make_env, args.env, device, 1, args.env_type),
            logger=logger,
            log_interval=args.log_interval,
            seed=args.seed,
        )
    else:
        start_step = 0
        if args.resume:
            start_step = int(cm.load(agent, args.resume, device=device).get("step", 0))
            print(f"[srl-train] Resuming from step {start_step}: {args.resume}")
        agent.num_timesteps = start_step
        evaluator = Evaluator(
            env_fn=lambda: _envs.make_env(args.env, device, 1, args.env_type),
            eval_freq=args.eval_freq,
            n_episodes=args.eval_episodes,
            seed=args.seed + 10_000,
            env_type=args.env_type,
            render=args.render,
            best_cm=best_cm,
            best_tracker=best_tracker,
        )
        extra = {"runner_cfg": exp.runner} if isinstance(agent, OffPolicyAgent) else {}
        agent.learn(
            max(args.steps - start_step, 0),
            callbacks=[
                CheckpointCallback(cm, save_interval=trainer_cfg.save_interval, model=agent)
            ],
            logger=logger,
            evaluator=evaluator,
            seed=args.seed,
            reset_num_timesteps=False,
            env_type=args.env_type,
            **extra,
        )

    _stop_visualizer(visualizer_handle)
    cm.save(agent, step=args.steps, tag="final")
    logger.set_step(args.steps)
    logger.close()
    env.close()
    print("[srl-train] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
