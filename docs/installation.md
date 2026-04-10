# Installation

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| PyTorch | ≥ 2.0 |
| Gymnasium | ≥ 1.0 |
| CUDA (optional) | ≥ 11.8 |

---

## Core installation

```bash
pip install srl-rl
```

## Install from GitHub (bleeding-edge)

```bash
pip install git+https://github.com/Bigkatoan/SRL.git
```

## Editable install (for development)

```bash
git clone https://github.com/Bigkatoan/SRL.git
cd SRL
pip install -e ".[dev]"
```

---

## Optional extras

```bash
# MuJoCo physics environments
pip install "srl-rl[mujoco]"

# Gymnasium Box2D
pip install "srl-rl[box2d]"

# gymnasium-robotics (Fetch, AntMaze, …)
pip install "srl-rl[robotics]"

# racecar_gym (best effort; currently recommended on Python 3.10)
pip install "srl-rl[racecar]"

# Everything at once
pip install "srl-rl[all]"

# ROS 2 dependencies
pip install "srl-rl[ros2]"

# Development tools (mkdocs, pytest, mypy, …)
pip install "srl-rl[dev]"
```

---

## Isaac Lab

Isaac Lab requires a separate install.  Follow the [official guide](https://isaac-sim.github.io/IsaacLab/),
then install SRL inside the Isaac Lab Python environment:

```bash
# Inside Isaac Lab conda/venv:
pip install srl-rl
```

---

## Verify installation

```python
import srl
print(srl.__version__)

import gymnasium as gym
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
env = GymnasiumWrapper(gym.make("Pendulum-v1"))
obs, _ = env.reset()
print("obs keys:", list(obs))   # ['state']
env.close()
```
