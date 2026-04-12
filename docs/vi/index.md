# SRL — Simple Reinforcement Learning

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://bigkatoan.github.io/SRL/)

**SRL** is a modular reinforcement learning library  
for continuous action-space environments.

> **Install note**: `srl-rl` is not published on PyPI yet. Do not use `pip install srl-rl`.
> Install from GitHub instead.

---

## Features

- **Algorithms**: PPO, SAC, DDPG, TD3, A2C, A3C — all supporting continuous actions
- **YAML-first model system**: declare encoders, heads, flows, and multimodal routing as data
- **Multi-modal inputs**: state, pixels, lidar, text — all in one model
- **Vectorised training**: `SyncVectorEnv` and `AsyncVectorEnv` for fast data collection
- **Goal-conditioned RL**: `GoalEnvWrapper` for gymnasium-robotics Fetch tasks
- **Isaac Lab support**: headless testing and training for PPO, A2C, SAC, DDPG, and TD3
- **ROS 2 Python API**: integrate trained agents into your own ROS 2 code
- **CLI**: `srl-train --config ... --env ... --algo ...`
- **Async off-policy runner**: decoupled collector/trainer threads for Isaac Lab throughput
- **GPU replay buffer**: zero-copy CUDA circular buffer for pixel-based Isaac Lab tasks
- **Stable visual RL**: separate encoder optimizer, `encoder_update_freq`, and 8 encoder loss modes

---

## Quick install

```bash
# From GitHub (not yet on PyPI)
pip install git+https://github.com/Bigkatoan/SRL.git

# Or clone locally
git clone https://github.com/Bigkatoan/SRL.git && cd SRL
pip install -e .

# With MuJoCo
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[mujoco]"

# With Box2D
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[box2d]"

# With gymnasium-robotics
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[robotics]"

# With racecar_gym (Python 3.10 recommended)
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[racecar]"

# Everything
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[all]"
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
| Isaac Lab | Cartpole, Ant, Humanoid | PPO / A2C / SAC / DDPG |

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
          --device cuda \
          --log-interval 5000 \
          --episode-window 25 \
          --plot-metrics train/score_mean,sac/critic_loss
```

Each run writes compact terminal summaries during training and stores `summary.json`, `history.csv`, `metrics.jsonl`, and `training_curves.svg` in the run directory. Use `--no-plots`, `--plot-metrics`, `--log-interval`, `--episode-window`, and `--console-metrics` to customize the logging layout and exported curves.

---

## Read this next

If you want to understand SRL properly, start from the YAML layer rather than the algorithm pages.

- [YAML Core Guide](yaml_core.md) explains the declarative model graph, observation routing, and current system boundary.
- [Configuration Reference](config_reference.md) lists the supported schema fields in detail.
- [Quick Start](quickstart.md) shows how the same YAML file is consumed by the CLI and Python API.
