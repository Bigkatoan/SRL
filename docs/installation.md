# Installation

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| PyTorch | ≥ 2.0 |
| Gymnasium | ≥ 1.0 |
| CUDA (optional) | ≥ 11.8 |

---

## Install

> **Note**: `srl-rl` is **not yet on PyPI**. Use GitHub install.

```bash
# From GitHub (recommended)
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
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[mujoco]"

# Gymnasium Box2D
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[box2d]"

# gymnasium-robotics (Fetch, AntMaze, …)
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[robotics]"

# racecar_gym (Python 3.10 recommended)
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[racecar]"

# Everything at once
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[all]"

# Development tools (mkdocs, pytest, mypy, …)
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[dev]"
```

---

## Isaac Lab

Isaac Lab requires a separate install.  Follow the [official guide](https://isaac-sim.github.io/IsaacLab/),
then install SRL inside the Isaac Lab Python environment:

```bash
# Inside Isaac Lab conda/venv:
pip install git+https://github.com/Bigkatoan/SRL.git
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
