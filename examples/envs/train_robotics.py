"""Train SAC on gymnasium-robotics Goal environments.

Supported environments:
  - FetchReach-v4          (obs_flat=16, act=4)  — easiest
  - FetchPush-v4           (obs_flat=31, act=4)
  - FetchPickAndPlace-v4   (obs_flat=31, act=4)
  - FetchSlide-v4          (obs_flat=31, act=4)  — hardest

The flat observation is:
  ``[observation | achieved_goal | desired_goal]``

Usage:
  python examples/envs/train_robotics.py --env FetchReach-v4
  python examples/envs/train_robotics.py --env FetchPush-v4

Requires: gymnasium-robotics
  pip install "srl-rl[robotics]"
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Register robotics envs before anything else
import gymnasium_robotics
gymnasium_robotics.register_robotics_envs()

from srl.algorithms.sac import SAC
from srl.core.config import SACConfig
from srl.envs.goal_env_wrapper import GoalEnvWrapper
from srl.registry.builder import ModelBuilder
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

# ──────────────────────────────────────────────────────────────
# Per-environment hyperparameters
# ──────────────────────────────────────────────────────────────
ENV_CFGS = {
    "FetchReach-v4": dict(
        yaml="configs/envs/fetch_reach_sac.yaml",
        obs_flat=16, action_dim=4,
        total_steps=500_000, buffer_size=1_000_000,
        batch_size=256, gamma=0.98, tau=0.005,
        learning_starts=10_000, gradient_steps=40,
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        target="success_rate > 0.90",
    ),
    "FetchPush-v4": dict(
        yaml="configs/envs/fetch_push_sac.yaml",
        obs_flat=31, action_dim=4,
        total_steps=2_000_000, buffer_size=1_000_000,
        batch_size=256, gamma=0.98, tau=0.005,
        learning_starts=10_000, gradient_steps=40,
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        target="success_rate > 0.80",
    ),
    "FetchPickAndPlace-v4": dict(
        yaml="configs/envs/fetch_pick_and_place_sac.yaml",
        obs_flat=31, action_dim=4,
        total_steps=5_000_000, buffer_size=1_000_000,
        batch_size=256, gamma=0.98, tau=0.005,
        learning_starts=10_000, gradient_steps=40,
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        target="success_rate > 0.70",
    ),
    "FetchSlide-v4": dict(
        yaml="configs/envs/fetch_slide_sac.yaml",
        obs_flat=31, action_dim=4,
        total_steps=5_000_000, buffer_size=1_000_000,
        batch_size=256, gamma=0.98, tau=0.005,
        learning_starts=10_000, gradient_steps=40,
        lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        target="success_rate > 0.60",
    ),
}


def train(env_id: str, device: str, seed: int = 0):
    if env_id not in ENV_CFGS:
        raise ValueError(f"Unknown env: {env_id}. Choose from {list(ENV_CFGS)}")

    cfg_dict = ENV_CFGS[env_id]
    print(f"\n=== SAC on {env_id} (GoalEnv) ===")
    print(f"Flat obs dim: {cfg_dict['obs_flat']}  action_dim: {cfg_dict['action_dim']}")
    print(f"Target: {cfg_dict['target']}")

    model  = ModelBuilder.from_yaml(cfg_dict["yaml"])
    target = ModelBuilder.from_yaml(cfg_dict["yaml"])

    sac_cfg = SACConfig(
        lr_actor        = cfg_dict["lr_actor"],
        lr_critic       = cfg_dict["lr_critic"],
        lr_alpha        = cfg_dict["lr_alpha"],
        buffer_size     = cfg_dict["buffer_size"],
        batch_size      = cfg_dict["batch_size"],
        gamma           = cfg_dict["gamma"],
        tau             = cfg_dict["tau"],
        learning_starts = cfg_dict["learning_starts"],
        train_freq      = 1,
        gradient_steps  = cfg_dict["gradient_steps"],
        action_dim      = cfg_dict["action_dim"],
    )

    env   = GoalEnvWrapper(gym.make(env_id))
    agent = SAC(model, target, config=sac_cfg, device=device)

    tag    = env_id.lower().replace("-", "_")
    logger = Logger(f"runs/sac_{tag}")
    cm     = CheckpointManager(f"checkpoints/sac_{tag}")

    obs, _ = env.reset(seed=seed)
    ep_reward    = 0.0
    ep_successes: list[float] = []

    for step in range(cfg_dict["total_steps"]):
        obs_t = {k: torch.from_numpy(v).float().unsqueeze(0).to(device)
                 for k, v in obs.items()}

        if step < sac_cfg.learning_starts:
            action_np = env.act_space.sample()
        else:
            action, _, _, _ = agent.predict(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, done, truncated, info = env.step(action_np)
        ep_reward += reward

        agent.buffer.add(
            obs=obs, action=action_np,
            reward=np.array([reward], dtype=np.float32),
            done=np.array([done],     dtype=bool),
            truncated=np.array([truncated], dtype=bool),
            next_obs=next_obs,
        )
        obs = next_obs

        if done or truncated:
            success = float(info.get("is_success", 0.0))
            ep_successes.append(success)
            logger.log("train/ep_reward",  ep_reward,         step)
            logger.log("train/is_success", success,           step)
            obs, _ = env.reset()
            ep_reward = 0.0

        if step >= sac_cfg.learning_starts:
            metrics = agent.update()
            if step % 10_000 == 0:
                logger.log_dict(metrics, step)

        if step % 100_000 == 0 and step > 0:
            cm.save(agent.model, step=step)
            recent_sr = (
                sum(ep_successes[-50:]) / min(50, len(ep_successes))
                if ep_successes else float("nan")
            )
            print(f"  [{env_id}] step={step:>8,}  success_rate={recent_sr:.2f}")

    print(f"\nTraining complete on {env_id}.")
    env.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on gymnasium-robotics GoalEnvs.")
    parser.add_argument("--env",    default="FetchReach-v4",
                        choices=list(ENV_CFGS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()
    train(args.env, args.device, args.seed)
