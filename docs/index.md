# SRL — Simple Reinforcement Learning

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/srl-rl)](https://pypi.org/project/srl-rl/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://bigkatoan.github.io/SRL/)

</div>

**SRL** is a modular, pip-installable reinforcement learning library  
for continuous action-space environments.

---

## Features

- **Algorithms**: PPO, SAC, DDPG, A2C, A3C — all supporting continuous actions
- **Composable networks**: plug-and-play MLP / CNN / GRU encoders via YAML config
- **Multi-modal inputs**: state, pixels, lidar, text — all in one model
- **Vectorised training**: `SyncVectorEnv` and `AsyncVectorEnv` for fast data collection
- **Goal-conditioned RL**: `GoalEnvWrapper` for gymnasium-robotics Fetch tasks
- **Isaac Lab support**: massively-parallel GPU environments out of the box
- **ROS 2 node**: deploy trained agents directly as ROS 2 topics/services
- **CLI**: `srl-train --config ... --env ... --algo ...`

---

## Quick install

```bash
# Core (CPU / GPU)
pip install srl-rl

# Install from GitHub (latest)
pip install git+https://github.com/Bigkatoan/SRL.git

# With MuJoCo
pip install "srl-rl[mujoco]"

# With Box2D
pip install "srl-rl[box2d]"

# With gymnasium-robotics
pip install "srl-rl[robotics]"

# With racecar_gym (Python 3.10 recommended)
pip install "srl-rl[racecar]"

# Everything
pip install "srl-rl[all]"
```

---

## Supported environments

| Suite | Environments | Algorithm |
|---|---|---|
| Gymnasium Classic | Pendulum, MountainCarContinuous | PPO |
| Gymnasium Box2D | BipedalWalker, LunarLanderContinuous, CarRacing | PPO |
| Gymnasium MuJoCo | HalfCheetah, Ant, Hopper, Walker2d, Humanoid, Swimmer, Pusher, Reacher | PPO / SAC |
| Gymnasium Robotics | FetchReach, FetchPush, FetchPickAndPlace, FetchSlide | SAC |
| racecar_gym | SingleAgentAustria, SingleAgentBerlin, SingleAgentMontreal, SingleAgentTorino | PPO |
| Isaac Lab | Cartpole, Ant, Humanoid | PPO |

---

## Five-minute example

```python
import gymnasium as gym
import torch
from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.sync_vector_env import SyncVectorEnv
from srl.registry.builder import ModelBuilder

model = ModelBuilder.from_yaml("configs/envs/pendulum_ppo.yaml")
env   = SyncVectorEnv([lambda: GymnasiumWrapper(gym.make("Pendulum-v1"))] * 4)
agent = PPO(model, PPOConfig(n_steps=512, num_envs=4), device="cuda")

obs, _ = env.reset()
for _ in range(200_000 // 512):
    for _ in range(512):
        obs_t  = {k: torch.from_numpy(v).float().cuda() for k, v in obs.items()}
        action, log_prob, value, _ = agent.predict(obs_t)
        obs, reward, done, trunc, _ = env.step(action.cpu().numpy())
        agent.buffer.add(obs=obs, action=action.cpu().numpy(),
                         reward=reward, done=done, truncated=trunc,
                         log_prob=log_prob.cpu().numpy(),
                         value=value.cpu().numpy())
    agent.buffer.compute_returns_and_advantages()
    agent.update()
```

---

## CLI

```bash
srl-train --config configs/envs/halfcheetah_sac.yaml \
          --env HalfCheetah-v5 \
          --algo sac \
          --steps 1000000 \
          --device cuda
```
