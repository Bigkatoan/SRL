"""Train SAC on Ant-v4 using a config file."""

import gymnasium as gym
import numpy as np
import torch

from srl.algorithms.sac import SAC
from srl.core.config import SACConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.registry.builder import ModelBuilder
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

YAML_CONFIG = "configs/sac_state.yaml"
TOTAL_STEPS = 3_000_000
WARMUP_STEPS = 10_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Build model + target ------------------------------------------------
model = ModelBuilder.from_yaml(YAML_CONFIG)
target = ModelBuilder.from_yaml(YAML_CONFIG)

cfg = SACConfig(
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    lr_actor=3e-4,
    lr_critic=3e-4,
    lr_alpha=3e-4,
    action_dim=8,
)
agent = SAC(model, target, config=cfg, device=DEVICE)

# ---- Environment ----------------------------------------------------------
env = GymnasiumWrapper(gym.make("Ant-v4"))

# ---- Logging --------------------------------------------------------------
logger = Logger(log_dir="runs/sac_ant")
cm = CheckpointManager("checkpoints/sac_ant")

# ---- Training loop -------------------------------------------------------
obs, _ = env.reset()
ep_reward = 0.0

for step in range(TOTAL_STEPS):
    obs_t = {k: torch.from_numpy(v).float().unsqueeze(0).to(DEVICE) for k, v in obs.items()}

    if step < WARMUP_STEPS:
        action_np = env.act_space.sample()
    else:
        action, _, _, _ = agent.predict(obs_t)
        action_np = action.squeeze(0).cpu().numpy()

    next_obs, reward, done, truncated, info = env.step(action_np)
    ep_reward += reward

    agent.buffer.add(
        obs=obs,
        action=action_np,
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

    if step >= WARMUP_STEPS:
        metrics = agent.update()
        if step % 5000 == 0:
            logger.log_dict(metrics, step)

    if step % 100_000 == 0 and step > 0:
        cm.save(agent.model, step=step)

print("Training complete.")
env.close()
logger.close()
