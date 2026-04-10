"""Train PPO on Gymnasium Box2D continuous environments.

Supported environments:
  - BipedalWalker-v3
  - LunarLanderContinuous-v3
  - CarRacing-v3

Usage:
  python examples/envs/train_box2d.py --env BipedalWalker-v3
  python examples/envs/train_box2d.py --env LunarLanderContinuous-v3
  python examples/envs/train_box2d.py --env CarRacing-v3
"""

from __future__ import annotations

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.sync_vector_env import SyncVectorEnv
from srl.registry.builder import ModelBuilder
from srl.utils.callbacks import CheckpointCallback, LogCallback
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger


class CHWObsWrapper(gym.ObservationWrapper):
    """Convert image observations from HWC uint8 to CHW uint8."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(c, h, w),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


ENV_CFGS = {
    "BipedalWalker-v3": dict(
        yaml="configs/envs/bipedal_walker_ppo.yaml",
        total_steps=3_000_000,
        n_envs=8,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target="ep_reward > 250",
        image=False,
    ),
    "LunarLanderContinuous-v3": dict(
        yaml="configs/envs/lunar_lander_continuous_ppo.yaml",
        total_steps=1_500_000,
        n_envs=8,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target="ep_reward > 200",
        image=False,
    ),
    "CarRacing-v3": dict(
        yaml="configs/envs/car_racing_ppo_visual.yaml",
        total_steps=5_000_000,
        n_envs=4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target="ep_reward > 700",
        image=True,
    ),
}


def train(env_id: str, device: str, seed: int = 0):
    if env_id not in ENV_CFGS:
        raise ValueError(f"Unknown env: {env_id}")
    cfg = ENV_CFGS[env_id]

    print(f"\n=== PPO on {env_id} ===")
    print(f"Target: {cfg['target']}")

    model = ModelBuilder.from_yaml(cfg["yaml"])
    ppo_cfg = PPOConfig(
        n_steps=cfg["n_steps"],
        num_envs=cfg["n_envs"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        lr=cfg["lr"],
        entropy_coef=cfg["entropy_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
    )

    def make_env():
        e = gym.make(env_id)
        if cfg["image"]:
            e = CHWObsWrapper(e)
        return GymnasiumWrapper(e)

    env = SyncVectorEnv([make_env for _ in range(cfg["n_envs"])])
    agent = PPO(model, config=ppo_cfg, device=device)

    tag = env_id.lower().replace("-", "_")
    logger = Logger(log_dir=f"runs/ppo_{tag}")
    cm = CheckpointManager(f"checkpoints/ppo_{tag}")
    callbacks = [
        LogCallback(logger, log_interval=ppo_cfg.n_steps * cfg["n_envs"]),
        CheckpointCallback(cm, save_interval=200_000),
    ]

    obs, _ = env.reset(seed=seed)
    step = 0

    while step < cfg["total_steps"]:
        for _ in range(ppo_cfg.n_steps):
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
            step += cfg["n_envs"]

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

        if step % 100_000 == 0:
            print(f"[{env_id}] step={step:,}")

    env.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on Box2D environments")
    parser.add_argument("--env", default="BipedalWalker-v3", choices=list(ENV_CFGS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.env, args.device, args.seed)
