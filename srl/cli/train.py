"""CLI entry point: srl-train --config path/to/config.yaml [options]"""

from __future__ import annotations

import argparse
import sys


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
    p.add_argument("--config",  required=True, help="Path to the YAML model config file")
    p.add_argument("--env",     required=True, help="Gymnasium environment id or 'isaaclab:<task>'")
    p.add_argument("--algo",    default=None,  help="Algorithm override: ppo|sac|ddpg|a2c|a3c (auto-detected from config)")
    p.add_argument("--steps",   type=int, default=1_000_000, help="Total environment steps (default: 1M)")
    p.add_argument("--n-envs",  type=int, default=1,          help="Parallel environments (default: 1)")
    p.add_argument("--device",  default="auto", help="PyTorch device: cpu|cuda|auto (default: auto)")
    p.add_argument("--seed",    type=int, default=0,           help="Random seed (default: 0)")
    p.add_argument("--logdir",  default="runs",                help="TensorBoard log dir (default: runs/)")
    p.add_argument("--ckptdir", default="checkpoints",         help="Checkpoint directory (default: checkpoints/)")
    p.add_argument("--eval-freq", type=int, default=50_000,    help="Evaluation frequency in steps")
    p.add_argument("--eval-episodes", type=int, default=10,    help="Episodes per evaluation")
    p.add_argument("--render",  action="store_true",           help="Render environment during eval")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    import os
    import random
    import numpy as np
    import torch
    from srl.registry.builder import ModelBuilder

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

    # ── build model ───────────────────────────────────────────────────────────
    print(f"[srl-train] Loading config: {args.config}")
    model = ModelBuilder.from_yaml(args.config)

    # ── infer algorithm from config filename ─────────────────────────────────
    algo_name = args.algo
    if algo_name is None:
        cfg_lower = os.path.basename(args.config).lower()
        for a in ("sac", "ddpg", "a3c", "a2c", "ppo"):
            if a in cfg_lower:
                algo_name = a
                break
        if algo_name is None:
            algo_name = "ppo"
    print(f"[srl-train] Algorithm: {algo_name.upper()}")

    # ── build environment ─────────────────────────────────────────────────────
    from srl.envs.gymnasium_wrapper import GymnasiumWrapper
    from srl.envs.sync_vector_env import SyncVectorEnv
    import gymnasium as gym

    def _make_env(seed_offset=0):
        if args.env.startswith("isaaclab:"):
            from srl.envs.isaac_lab_wrapper import IsaacLabWrapper
            task_name = args.env.split(":", 1)[1]
            import isaaclab_tasks  # noqa: F401
            from isaaclab.envs import ManagerBasedRLEnv
            env_cfg = isaaclab_tasks.utils.parse_env_cfg(task_name, device=device, num_envs=args.n_envs)
            base_env = ManagerBasedRLEnv(cfg=env_cfg)
            return IsaacLabWrapper(base_env)
        else:
            base = gym.make(args.env)
            return GymnasiumWrapper(base)

    print(f"[srl-train] Creating {args.n_envs} × {args.env}")
    if algo_name in ("sac", "ddpg") or args.n_envs == 1:
        env = _make_env()
    else:
        env = SyncVectorEnv([lambda i=i: _make_env(i) for i in range(args.n_envs)])

    # ── build agent ───────────────────────────────────────────────────────────
    from srl.utils.logger import Logger
    from srl.utils.checkpoint import CheckpointManager
    from srl.utils.callbacks import LogCallback, CheckpointCallback

    run_name = f"{algo_name}_{os.path.splitext(os.path.basename(args.config))[0]}"
    logger = Logger(log_dir=os.path.join(args.logdir, run_name), verbose=True)
    cm = CheckpointManager(os.path.join(args.ckptdir, run_name), max_keep=5)
    callbacks = [
        LogCallback(logger, log_interval=2048),
        CheckpointCallback(cm, save_interval=100_000),
    ]

    import copy

    if algo_name == "ppo":
        from srl.algorithms.ppo import PPO
        from srl.core.config import PPOConfig
        agent = PPO(model, config=PPOConfig(n_steps=2048, num_envs=args.n_envs), device=device)
        _run_on_policy(agent, env, args, callbacks, logger)

    elif algo_name == "a2c":
        from srl.algorithms.a2c import A2C
        from srl.core.config import A2CConfig
        agent = A2C(model, config=A2CConfig(), device=device)
        _run_on_policy(agent, env, args, callbacks, logger)

    elif algo_name == "a3c":
        from srl.algorithms.a3c import A3C
        from srl.core.config import A3CConfig
        agent = A3C(model, config=A3CConfig(n_workers=args.n_envs), device=device)
        agent.train(total_timesteps=args.steps, env_fn=_make_env)

    elif algo_name == "sac":
        from srl.algorithms.sac import SAC
        from srl.core.config import SACConfig
        target = copy.deepcopy(model)
        agent = SAC(model, target, config=SACConfig(), device=device)
        _run_off_policy(agent, env, args, callbacks, logger)

    elif algo_name == "ddpg":
        from srl.algorithms.ddpg import DDPG
        from srl.core.config import DDPGConfig
        target = copy.deepcopy(model)
        agent = DDPG(model, target, config=DDPGConfig(), device=device)
        _run_off_policy(agent, env, args, callbacks, logger)

    else:
        print(f"[srl-train] Unknown algorithm: {algo_name}", file=sys.stderr)
        return 1

    cm.save(model, step=args.steps, tag="final")
    logger.close()
    env.close()
    print("[srl-train] Done.")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for mapping obs keys to encoder names
# ──────────────────────────────────────────────────────────────────────────────

def _remap_obs_to_encoders(obs_dict: dict, encoder_names: list[str]) -> dict:
    """Map observation dict keys to encoder input names.
    
    If obs_dict has keys like {'state'} and encoders expect {'state_enc'},
    automatically map them together.
    """
    # Simple approach: if obs has 'state' key and we have an encoder, 
    # rename it to match the first encoder name
    if len(obs_dict) == 1 and len(encoder_names) == 1:
        obs_key = list(obs_dict.keys())[0]
        enc_name = encoder_names[0]
        if obs_key != enc_name:
            return {enc_name: obs_dict[obs_key]}
    return obs_dict


# ──────────────────────────────────────────────────────────────────────────────
# Training loops
# ──────────────────────────────────────────────────────────────────────────────

def _run_on_policy(agent, env, args, callbacks, logger) -> None:
    import torch, numpy as np
    from srl.core.rollout_buffer import RolloutBuffer

    n_steps = agent.cfg.n_steps
    obs, _ = env.reset()
    step = 0
    encoder_names = list(agent.model.encoders.keys())

    while step < args.steps:
        for _ in range(n_steps):
            obs_remapped = _remap_obs_to_encoders(obs, encoder_names)
            obs_t = {k: torch.from_numpy(np.asarray(v)).float().to(agent.device)
                     for k, v in obs_remapped.items()}
            action, log_prob, value, _ = agent.predict(obs_t)
            action_np = action.cpu().numpy()
            next_obs, reward, done, trunc, _ = env.step(action_np)
            agent.buffer.add(
                obs=obs, action=action_np, reward=np.asarray(reward),
                done=np.asarray(done),
                log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                value=value.cpu().numpy() if value is not None else None,
            )
            obs = next_obs
            step += getattr(agent.cfg, "num_envs", 1)

        obs_remapped_final = _remap_obs_to_encoders(obs, encoder_names)
        last_t = {k: torch.from_numpy(np.asarray(v)).float().to(agent.device)
                  for k, v in obs_remapped_final.items()}
        _, _, last_val, _ = agent.predict(last_t)
        agent.buffer.compute_returns_and_advantages(
            last_value=last_val.cpu().numpy() if last_val is not None else 0.0
        )
        metrics = agent.update()
        metrics["step"] = step
        for cb in callbacks:
            cb.on_step_end(step, metrics)


def _run_off_policy(agent, env, args, callbacks, logger) -> None:
    import torch, numpy as np

    warmup = getattr(agent.cfg, "learning_starts", 10_000)
    obs, _ = env.reset()
    encoder_names = list(agent.model.encoders.keys())

    for step in range(args.steps):
        obs_remapped = _remap_obs_to_encoders(obs, encoder_names)
        obs_t = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(agent.device)
                 for k, v in obs_remapped.items()}
        if step < warmup:
            action_np = env.act_space.sample()
        else:
            action, _, _, _ = agent.predict(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, done, trunc, _ = env.step(action_np)
        agent.buffer.add(
            obs=obs, action=action_np,
            reward=np.array([reward], dtype=np.float32),
            done=np.array([done], dtype=bool),
            next_obs=next_obs,
        )
        obs = next_obs
        if done or trunc:
            obs, _ = env.reset()

        if step >= warmup:
            metrics = agent.update()
            if metrics:
                for cb in callbacks:
                    cb.on_step_end(step, metrics)


if __name__ == "__main__":
    sys.exit(main())
