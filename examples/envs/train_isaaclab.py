"""Train PPO on Isaac Lab environments.

Supported environments (requires IsaacLab + IsaacSim):
  - Isaac-Cartpole-v0       (obs=4,  act=1)   — simplest, good smoke test
  - Isaac-Ant-v0            (obs=60, act=8)
  - Isaac-Humanoid-v0       (obs=87, act=21)

Isaac Lab environments return tensors directly (GPU) and support massive
parallelism (n_envs=512–4096).  This script bridges them via SRL's
IsaacLabWrapper and runs PPO with large batches.

Usage:
  # From within Isaac Lab python environment:
  python examples/envs/train_isaaclab.py --env Isaac-Cartpole-v0
  python examples/envs/train_isaaclab.py --env Isaac-Ant-v0
  python examples/envs/train_isaaclab.py --env Isaac-Humanoid-v0 --n-envs 4096

Prerequisites:
  - IsaacSim installed
  - isaaclab activated (source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh)
  - pip install -e /path/to/SRL

Notes:
  - Run headless: add --headless flag (passed to IsaacLab env kwargs)
  - Isaac Lab handles env vectorization internally (n_envs passed at creation)
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper
from srl.registry.builder import ModelBuilder
from srl.utils.callbacks import CheckpointCallback, LogCallback
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

# ──────────────────────────────────────────────────────────────
# Per-environment hyperparameters
# ──────────────────────────────────────────────────────────────
ENV_CFGS = {
    "Isaac-Cartpole-v0": dict(
        yaml="configs/envs/isaaclab_cartpole_ppo.yaml",
        obs_dim=4, action_dim=1,
        total_steps=500_000,
        n_envs=512,     n_steps=16,  batch_size=8192,
        n_epochs=5,     lr=5e-4,     gamma=0.99,
        gae_lambda=0.95, clip_range=0.2,
        entropy_coef=0.01, vf_coef=1.0, max_grad_norm=1.0,
        target="ep_reward > 400",
    ),
    "Isaac-Ant-v0": dict(
        yaml="configs/envs/isaaclab_ant_ppo.yaml",
        obs_dim=60, action_dim=8,
        total_steps=5_000_000,
        n_envs=4096,    n_steps=32,  batch_size=16384,
        n_epochs=5,     lr=5e-4,     gamma=0.99,
        gae_lambda=0.95, clip_range=0.2,
        entropy_coef=0.005, vf_coef=1.0, max_grad_norm=1.0,
        target="ep_reward > 5000",
    ),
    "Isaac-Humanoid-v0": dict(
        yaml="configs/envs/isaaclab_humanoid_ppo.yaml",
        obs_dim=87, action_dim=21,
        total_steps=10_000_000,
        n_envs=4096,    n_steps=32,  batch_size=16384,
        n_epochs=5,     lr=5e-4,     gamma=0.99,
        gae_lambda=0.95, clip_range=0.2,
        entropy_coef=0.0, vf_coef=1.0, max_grad_norm=1.0,
        target="ep_reward > 5000",
    ),
}


def train(env_id: str, n_envs: int | None, device: str, headless: bool, seed: int):
    if env_id not in ENV_CFGS:
        raise ValueError(f"Unknown env: {env_id}. Choose from {list(ENV_CFGS)}")

    cfg_dict = ENV_CFGS[env_id]
    if n_envs is None:
        n_envs = cfg_dict["n_envs"]

    print(f"\n=== PPO on {env_id} (Isaac Lab) ===")
    print(f"n_envs={n_envs}  obs_dim={cfg_dict['obs_dim']}  action_dim={cfg_dict['action_dim']}")
    print(f"Target: {cfg_dict['target']}")

    # Isaac Lab env creation
    try:
        import isaaclab_tasks  # noqa: F401 — trigger task registration
        from omni.isaac.lab_tasks.utils import parse_env_cfg
        env_kwargs: dict = {"num_envs": n_envs}
        if headless:
            env_kwargs["headless"] = True
        raw_env = IsaacLabWrapper(env_id, **env_kwargs)
    except ImportError as e:
        raise SystemExit(
            f"IsaacLab not available: {e}\n"
            "Activate the IsaacLab environment before running this script."
        ) from e

    model   = ModelBuilder.from_yaml(cfg_dict["yaml"])
    ppo_cfg = PPOConfig(
        n_steps      = cfg_dict["n_steps"],
        num_envs     = n_envs,
        batch_size   = cfg_dict["batch_size"],
        n_epochs     = cfg_dict["n_epochs"],
        gamma        = cfg_dict["gamma"],
        gae_lambda   = cfg_dict["gae_lambda"],
        clip_range   = cfg_dict["clip_range"],
        lr           = cfg_dict["lr"],
        entropy_coef = cfg_dict["entropy_coef"],
        vf_coef      = cfg_dict["vf_coef"],
        max_grad_norm= cfg_dict["max_grad_norm"],
    )

    agent  = PPO(model, config=ppo_cfg, device=device)
    tag    = env_id.lower().replace("-", "_")
    logger = Logger(f"runs/ppo_{tag}")
    cm     = CheckpointManager(f"checkpoints/ppo_{tag}")
    callbacks = [
        LogCallback(logger, log_interval=ppo_cfg.n_steps * n_envs),
        CheckpointCallback(cm, save_interval=500_000),
    ]

    obs, _ = raw_env.reset(seed=seed)
    step   = 0

    while step < cfg_dict["total_steps"]:
        for _ in range(ppo_cfg.n_steps):
            obs_t = {k: torch.from_numpy(v).float().to(device) for k, v in obs.items()}
            action, log_prob, value, _ = agent.predict(obs_t)
            next_obs, reward, done, truncated, info = raw_env.step(action.cpu().numpy())

            agent.buffer.add(
                obs=obs, action=action.cpu().numpy(),
                reward=reward, done=done, truncated=truncated,
                log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                value=value.cpu().numpy()       if value   is not None else None,
            )
            obs   = next_obs
            step += n_envs

        last_obs_t = {k: torch.from_numpy(v).float().to(device) for k, v in obs.items()}
        with torch.no_grad():
            _, _, last_value, _ = agent.predict(last_obs_t)
        agent.buffer.compute_returns_and_advantages(
            last_value=last_value.cpu().numpy() if last_value is not None else None
        )

        metrics = agent.update()
        metrics["step"] = step
        for cb in callbacks:
            cb.on_step_end(step, metrics)

        if step % 500_000 == 0:
            print(f"  [{env_id}] step={step:>10,}")

    print(f"\nTraining complete on {env_id}.")
    raw_env.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Isaac Lab environments.")
    parser.add_argument("--env",      default="Isaac-Cartpole-v0",
                        choices=list(ENV_CFGS))
    parser.add_argument("--n-envs",   type=int, default=None,
                        help="Override number of parallel envs (default: per-env config)")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--headless", action="store_true",
                        help="Run IsaacSim in headless mode (no GUI)")
    parser.add_argument("--seed",     type=int, default=0)
    args = parser.parse_args()
    train(args.env, args.n_envs, args.device, args.headless, args.seed)
