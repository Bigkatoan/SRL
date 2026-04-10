"""Train PPO on classic control environments.

Supported environments:
  - Pendulum-v1          (obs=3,  act=1)
  - MountainCarContinuous-v0 (obs=2, act=1)

Usage:
  python examples/envs/train_classic_control.py --env Pendulum-v1
  python examples/envs/train_classic_control.py --env MountainCarContinuous-v0

All converge within the steps defined in configs/envs/<env>_ppo.yaml.
"""

import argparse
import os
import sys

import gymnasium as gym
import torch

# Make sure SRL is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.sync_vector_env import SyncVectorEnv
from srl.registry.builder import ModelBuilder
from srl.utils.callbacks import CheckpointCallback, LogCallback
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

# ──────────────────────────────────────────────────────────────
# Hyperparameters per environment
# ──────────────────────────────────────────────────────────────
ENV_CFGS = {
    "Pendulum-v1": dict(
        yaml="configs/envs/pendulum_ppo.yaml",
        total_steps=200_000,
        n_envs=4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target="ep_reward > -200",
    ),
    "MountainCarContinuous-v0": dict(
        yaml="configs/envs/mountain_car_continuous_ppo.yaml",
        total_steps=1_000_000,
        n_envs=8,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        lr=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target="ep_reward > 90",
    ),
}


def make_vec_env(env_id: str, n_envs: int):
    def _make():
        return GymnasiumWrapper(gym.make(env_id))
    return SyncVectorEnv([_make for _ in range(n_envs)])


def train(env_id: str, device: str, seed: int = 0):
    if env_id not in ENV_CFGS:
        raise ValueError(f"Unknown env: {env_id}. Choose from {list(ENV_CFGS)}")

    cfg_dict = ENV_CFGS[env_id]
    print(f"\n=== Training PPO on {env_id} ===")
    print(f"Target: {cfg_dict['target']}")
    print(f"Steps:  {cfg_dict['total_steps']:,}")

    model = ModelBuilder.from_yaml(cfg_dict["yaml"])

    cfg = PPOConfig(
        n_steps      = cfg_dict["n_steps"],
        num_envs     = cfg_dict["n_envs"],
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

    n_envs  = cfg_dict["n_envs"]
    env     = make_vec_env(env_id, n_envs)
    agent   = PPO(model, config=cfg, device=device)

    tag     = env_id.lower().replace("-", "_").replace(".", "_")
    logger  = Logger(log_dir=f"runs/ppo_{tag}")
    cm      = CheckpointManager(f"checkpoints/ppo_{tag}")
    callbacks = [
        LogCallback(logger, log_interval=cfg.n_steps * n_envs),
        CheckpointCallback(cm, save_interval=100_000),
    ]

    obs, _  = env.reset(seed=seed)
    step    = 0
    ep_rewards: list[float] = []

    while step < cfg_dict["total_steps"]:
        for _ in range(cfg.n_steps):
            obs_t = {k: torch.from_numpy(v).float().to(device) for k, v in obs.items()}
            action, log_prob, value, _ = agent.predict(obs_t)
            action_np = action.cpu().numpy()

            next_obs, reward, done, truncated, info = env.step(action_np)
            agent.buffer.add(
                obs=obs,
                action=action_np,
                reward=reward,
                done=done,
                truncated=truncated,
                log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                value=value.cpu().numpy() if value is not None else None,
            )
            obs = next_obs
            step += n_envs

            # Accumulate episode rewards from info (gym vector env convention)
            if isinstance(info, dict) and "episode" in info:
                for ep in info["episode"].get("r", []):
                    ep_rewards.append(float(ep))
            elif isinstance(info, list):
                for i_info in info:
                    if "episode" in i_info:
                        ep_rewards.append(float(i_info["episode"]["r"]))

        last_obs_t = {k: torch.from_numpy(v).float().to(device) for k, v in obs.items()}
        with torch.no_grad():
            _, _, last_value, _ = agent.predict(last_obs_t)
        agent.buffer.compute_returns_and_advantages(
            last_value=last_value.cpu().numpy() if last_value is not None else None
        )

        metrics = agent.update()
        metrics["step"] = step

        if ep_rewards:
            mean_r = sum(ep_rewards[-10:]) / min(len(ep_rewards), 10)
            metrics["train/ep_reward_mean"] = mean_r

        for cb in callbacks:
            cb.on_step_end(step, metrics)

        if step % 50_000 == 0:
            mean_r = (
                sum(ep_rewards[-10:]) / min(len(ep_rewards), 10)
                if ep_rewards else float("nan")
            )
            print(f"  step={step:>10,}  ep_reward_mean={mean_r:+.1f}")

    print(f"\nTraining complete! Final rewards: {ep_rewards[-5:] if ep_rewards else 'N/A'}")
    env.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on classic control envs.")
    parser.add_argument("--env", default="Pendulum-v1",
                        choices=list(ENV_CFGS),
                        help="Gymnasium env ID to train on.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.env, args.device, args.seed)
