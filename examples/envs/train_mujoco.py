"""Train PPO or SAC on MuJoCo locomotion environments.

Supported environments:
  PPO:
    - HalfCheetah-v5     (obs=17,  act=6)
    - Hopper-v5          (obs=11,  act=3)
    - Walker2d-v5        (obs=17,  act=6)
    - Humanoid-v5        (obs=348, act=17)
    - Swimmer-v5  [SAC also great]  (obs=8,  act=2)

  SAC:
    - HalfCheetah-v5     (obs=17,  act=6)
    - Ant-v5             (obs=105, act=8)
    - Swimmer-v5         (obs=8,   act=2)
    - Pusher-v5          (obs=23,  act=7)
    - Reacher-v5         (obs=10,  act=2)

Usage:
  python examples/envs/train_mujoco.py --env HalfCheetah-v5 --algo ppo
  python examples/envs/train_mujoco.py --env Ant-v5 --algo sac
  python examples/envs/train_mujoco.py --env Humanoid-v5 --algo ppo
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from srl.algorithms.ppo import PPO
from srl.algorithms.sac import SAC
from srl.core.config import PPOConfig, SACConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.sync_vector_env import SyncVectorEnv
from srl.registry.builder import ModelBuilder
from srl.utils.callbacks import CheckpointCallback, LogCallback
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

# ──────────────────────────────────────────────────────────────
# Per-environment hyperparameters
# ──────────────────────────────────────────────────────────────
ENV_CFGS = {
    # (yaml, algo, train_kwargs, target_description)
    ("HalfCheetah-v5", "ppo"): dict(
        yaml="configs/envs/halfcheetah_ppo.yaml",
        total_steps=2_000_000, n_envs=8, n_steps=2048, batch_size=256,
        n_epochs=10, lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, entropy_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        action_dim=6, target="ep_reward > 5000",
    ),
    ("HalfCheetah-v5", "sac"): dict(
        yaml="configs/envs/halfcheetah_sac.yaml",
        total_steps=1_000_000, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        buffer_size=1_000_000, batch_size=256, gamma=0.99, tau=0.005,
        learning_starts=10_000, train_freq=1, gradient_steps=1,
        action_dim=6, target="ep_reward > 8000",
    ),
    ("Ant-v5", "sac"): dict(
        yaml="configs/envs/ant_sac.yaml",
        total_steps=3_000_000, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        buffer_size=1_000_000, batch_size=256, gamma=0.99, tau=0.005,
        learning_starts=10_000, train_freq=1, gradient_steps=1,
        action_dim=8, target="ep_reward > 4000",
    ),
    ("Hopper-v5", "ppo"): dict(
        yaml="configs/envs/hopper_ppo.yaml",
        total_steps=1_000_000, n_envs=4, n_steps=2048, batch_size=64,
        n_epochs=10, lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, entropy_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        action_dim=3, target="ep_reward > 2500",
    ),
    ("Walker2d-v5", "ppo"): dict(
        yaml="configs/envs/walker2d_ppo.yaml",
        total_steps=2_000_000, n_envs=8, n_steps=2048, batch_size=256,
        n_epochs=10, lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, entropy_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        action_dim=6, target="ep_reward > 3000",
    ),
    ("Humanoid-v5", "ppo"): dict(
        yaml="configs/envs/humanoid_ppo.yaml",
        total_steps=10_000_000, n_envs=16, n_steps=2048, batch_size=512,
        n_epochs=10, lr=2.5e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, entropy_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        action_dim=17, target="ep_reward > 5000",
    ),
    ("Swimmer-v5", "sac"): dict(
        yaml="configs/envs/swimmer_sac.yaml",
        total_steps=500_000, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        buffer_size=1_000_000, batch_size=256, gamma=0.9999, tau=0.005,
        learning_starts=5_000, train_freq=1, gradient_steps=1,
        action_dim=2, target="ep_reward > 300",
    ),
    ("Pusher-v5", "sac"): dict(
        yaml="configs/envs/pusher_sac.yaml",
        total_steps=500_000, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        buffer_size=1_000_000, batch_size=256, gamma=0.99, tau=0.005,
        learning_starts=5_000, train_freq=1, gradient_steps=1,
        action_dim=7, target="ep_reward > -50",
    ),
    ("Reacher-v5", "sac"): dict(
        yaml="configs/envs/reacher_sac.yaml",
        total_steps=200_000, lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
        buffer_size=200_000, batch_size=256, gamma=0.99, tau=0.005,
        learning_starts=5_000, train_freq=1, gradient_steps=1,
        action_dim=2, target="ep_reward > -5",
    ),
}


# ──────────────────────────────────────────────────────────────
# PPO training loop
# ──────────────────────────────────────────────────────────────
def train_ppo(env_id, cfg_dict, device, seed):
    n_envs = cfg_dict["n_envs"]
    model  = ModelBuilder.from_yaml(cfg_dict["yaml"])
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
    env = SyncVectorEnv(
        [lambda eid=env_id: GymnasiumWrapper(gym.make(eid)) for _ in range(n_envs)]
    )
    agent     = PPO(model, config=ppo_cfg, device=device)
    tag       = env_id.lower().replace("-", "_").replace(".", "_")
    logger    = Logger(f"runs/ppo_{tag}")
    cm        = CheckpointManager(f"checkpoints/ppo_{tag}")
    callbacks = [
        LogCallback(logger, log_interval=ppo_cfg.n_steps * n_envs),
        CheckpointCallback(cm, save_interval=500_000),
    ]

    obs, _ = env.reset(seed=seed)
    step   = 0
    ep_rewards: list[float] = []

    while step < cfg_dict["total_steps"]:
        for _ in range(ppo_cfg.n_steps):
            obs_t = {k: torch.from_numpy(v).float().to(device) for k, v in obs.items()}
            action, log_prob, value, _ = agent.predict(obs_t)
            next_obs, reward, done, truncated, info = env.step(action.cpu().numpy())
            agent.buffer.add(
                obs=obs, action=action.cpu().numpy(),
                reward=reward, done=done, truncated=truncated,
                log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                value=value.cpu().numpy() if value is not None else None,
            )
            obs = next_obs
            step += n_envs
            if isinstance(info, list):
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
            metrics["train/ep_reward_mean"] = sum(ep_rewards[-10:]) / min(10, len(ep_rewards))
        for cb in callbacks:
            cb.on_step_end(step, metrics)
        if step % 200_000 == 0:
            mean_r = metrics.get("train/ep_reward_mean", float("nan"))
            print(f"  [{env_id}] step={step:>10,}  mean_reward={mean_r:+.1f}")

    env.close()
    logger.close()


# ──────────────────────────────────────────────────────────────
# SAC training loop
# ──────────────────────────────────────────────────────────────
def train_sac(env_id, cfg_dict, device, seed):
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
        train_freq      = cfg_dict["train_freq"],
        gradient_steps  = cfg_dict["gradient_steps"],
        action_dim      = cfg_dict["action_dim"],
    )
    env   = GymnasiumWrapper(gym.make(env_id))
    agent = SAC(model, target, config=sac_cfg, device=device)
    tag   = env_id.lower().replace("-", "_").replace(".", "_")
    logger= Logger(f"runs/sac_{tag}")
    cm    = CheckpointManager(f"checkpoints/sac_{tag}")

    obs, _ = env.reset(seed=seed)
    ep_reward = 0.0

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
            done=np.array([done], dtype=bool),
            truncated=np.array([truncated], dtype=bool),
            next_obs=next_obs,
        )
        obs = next_obs

        if done or truncated:
            logger.log("train/ep_reward", ep_reward, step)
            obs, _ = env.reset()
            ep_reward = 0.0

        if step >= sac_cfg.learning_starts:
            metrics = agent.update()
            if step % 10_000 == 0:
                logger.log_dict(metrics, step)

        if step % 200_000 == 0 and step > 0:
            cm.save(agent.model, step=step)
            print(f"  [{env_id}] step={step:>10,}  ckpt saved")

    env.close()
    logger.close()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
ENVS_PPO = [k[0] for k in ENV_CFGS if k[1] == "ppo"]
ENVS_SAC = [k[0] for k in ENV_CFGS if k[1] == "sac"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO/SAC on MuJoCo environments.")
    parser.add_argument("--env",    required=True,
                        help="Gymnasium MuJoCo env ID (e.g. HalfCheetah-v5)")
    parser.add_argument("--algo",   default="sac", choices=["ppo", "sac"],
                        help="Algorithm: ppo or sac")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",   type=int, default=0)
    args = parser.parse_args()

    key = (args.env, args.algo)
    if key not in ENV_CFGS:
        available = "\n  ".join(str(k) for k in sorted(ENV_CFGS))
        raise SystemExit(f"No config for {key}.\nAvailable:\n  {available}")

    cfg = ENV_CFGS[key]
    print(f"\n=== {args.algo.upper()} on {args.env} ===")
    print(f"Target: {cfg['target']}")

    if args.algo == "ppo":
        train_ppo(args.env, cfg, args.device, args.seed)
    else:
        train_sac(args.env, cfg, args.device, args.seed)

    print("Done.")
