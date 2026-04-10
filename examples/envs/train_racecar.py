"""Train PPO on racecar_gym single-agent tracks.

This script expects racecar_gym to be installed from source. On Python 3.11,
the current upstream release may fail to import due dataclass defaults. If that
happens, use Python 3.10 (or patch racecar_gym locally).

Usage:
  python examples/envs/train_racecar.py --env SingleAgentAustria-v0
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.racecar_wrapper import RacecarWrapper
from srl.registry.builder import ModelBuilder
from srl.utils.callbacks import CheckpointCallback, LogCallback
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

ENV_IDS = [
    "SingleAgentAustria-v0",
    "SingleAgentBerlin-v0",
    "SingleAgentMontreal-v0",
    "SingleAgentTorino-v0",
    "SingleAgentCircle-v0",
    "SingleAgentPlechaty-v0",
]


def _build_runtime_config(base_yaml: str, input_dim: int, action_dim: int) -> str:
    with open(base_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["encoders"][0]["input_dim"] = int(input_dim)
    cfg["actor"]["action_dim"] = int(action_dim)

    fd, temp_path = tempfile.mkstemp(prefix="racecar_cfg_", suffix=".yaml")
    os.close(fd)
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return temp_path


def train(env_id: str, device: str = "cpu", seed: int = 0):
    try:
        import gymnasium as gym
        import racecar_gym  # noqa: F401
    except Exception as e:  # pylint: disable=broad-except
        raise SystemExit(
            "racecar_gym is unavailable in this environment.\n"
            "Install from source and consider Python 3.10 for compatibility.\n"
            f"Original error: {e}"
        ) from e

    raw_env = gym.make(env_id)
    env = RacecarWrapper(raw_env)
    obs, _ = env.reset(seed=seed)

    input_dim = int(obs["state"].shape[0])
    action_dim = int(env.act_space.shape[0])
    runtime_yaml = _build_runtime_config(
        "configs/envs/racecar_austria_ppo.yaml",
        input_dim=input_dim,
        action_dim=action_dim,
    )

    model = ModelBuilder.from_yaml(runtime_yaml)

    cfg = PPOConfig(
        n_steps=1024,
        num_envs=1,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        lr=3e-4,
        entropy_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    agent = PPO(model, config=cfg, device=device)

    tag = env_id.lower().replace("-", "_")
    logger = Logger(log_dir=f"runs/ppo_{tag}")
    cm = CheckpointManager(f"checkpoints/ppo_{tag}")
    callbacks = [
        LogCallback(logger, log_interval=cfg.n_steps),
        CheckpointCallback(cm, save_interval=200_000),
    ]

    total_steps = 2_000_000
    step = 0
    ep_reward = 0.0

    while step < total_steps:
        for _ in range(cfg.n_steps):
            obs_t = {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in obs.items()}
            action, log_prob, value, _ = agent.predict(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            next_obs, reward, done, truncated, info = env.step(action_np)
            ep_reward += reward

            agent.buffer.add(
                obs=obs,
                action=action_np,
                reward=torch.tensor([reward], dtype=torch.float32).numpy(),
                done=torch.tensor([done], dtype=torch.bool).numpy(),
                truncated=torch.tensor([truncated], dtype=torch.bool).numpy(),
                log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                value=value.cpu().numpy() if value is not None else None,
            )

            obs = next_obs
            step += 1

            if done or truncated:
                logger.log("train/ep_reward", ep_reward, step)
                obs, _ = env.reset()
                ep_reward = 0.0

        last_obs_t = {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in obs.items()}
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
    os.remove(runtime_yaml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on racecar_gym tracks")
    parser.add_argument("--env", default="SingleAgentAustria-v0", choices=ENV_IDS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.env, args.device, args.seed)
