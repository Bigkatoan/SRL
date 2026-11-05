"""Train PPO on HalfCheetah-v4 using a config file."""

import gymnasium as gym
import torch

from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.sync_vector_env import SyncVectorEnv
from srl.registry.builder import ModelBuilder
from srl.utils.callbacks import CheckpointCallback, LogCallback
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

# ---- Config ---------------------------------------------------------------
YAML_CONFIG = "configs/ppo_state.yaml"
TOTAL_STEPS = 1_000_000
N_ENVS = 4
N_STEPS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Build model ----------------------------------------------------------
model = ModelBuilder.from_yaml(YAML_CONFIG)


# ---- Vectorised envs -------------------------------------------------------
def make_env():
    return GymnasiumWrapper(gym.make("HalfCheetah-v4"))


env = SyncVectorEnv([make_env for _ in range(N_ENVS)])

# ---- Agent -----------------------------------------------------------------
cfg = PPOConfig(
    n_steps=N_STEPS,
    num_envs=N_ENVS,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    lr=3e-4,
    entropy_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)
agent = PPO(model, config=cfg, device=DEVICE)

# ---- Callbacks -------------------------------------------------------------
logger = Logger(log_dir="runs/ppo_halfcheetah")
cm = CheckpointManager("checkpoints/ppo_halfcheetah")
callbacks = [
    LogCallback(logger, log_interval=cfg.n_steps * N_ENVS),
    CheckpointCallback(cm, save_interval=50_000),
]

# ---- Training loop ---------------------------------------------------------
obs, _ = env.reset()
step = 0

while step < TOTAL_STEPS:
    # Collect N_STEPS * N_ENVS transitions
    for _ in range(N_STEPS):
        obs_t = {k: torch.from_numpy(v).float().to(DEVICE) for k, v in obs.items()}
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
        step += N_ENVS

    # Bootstrap and update
    last_obs_t = {k: torch.from_numpy(v).float().to(DEVICE) for k, v in obs.items()}
    with torch.no_grad():
        _, _, last_value, _ = agent.predict(last_obs_t)
    last_val = last_value.cpu().numpy() if last_value is not None else None
    agent.buffer.compute_returns_and_advantages(last_value=last_val)

    metrics = agent.update()
    metrics["step"] = step

    for cb in callbacks:
        cb.on_step_end(step, metrics)

print("Training complete.")
env.close()
logger.close()
