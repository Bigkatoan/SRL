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

## M3bot on the current machine

If you are working with the separate `M3bot` task repository on this same machine, the currently verified runtime path is:

```bash
/path/to/isaaclab/venv/bin/python
```

The validated repository layout is:

```text
/path/to/SRL (repo root)
tests/M3bot
/path/to/isaaclab/venv
```

Recommended setup flow:

```bash
cd tests
git clone https://github.com/Bigkatoan/M3bot.git
cd M3bot

python3 tools/validate_source.py

OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python train.py --help
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python play.py --help
```

For a fuller task-specific guide, see [M3bot](environments/m3bot.md).

---

## Verify installation

Verify both the Python package and the console scripts in the same environment where you plan to run training.

```bash
python -m pip show srl-rl
command -v srl-train
command -v srl-benchmark
command -v srl-visualize

srl-train --help
srl-benchmark --help
srl-visualize --help
```

If the console scripts are not present yet, you are usually in one of these situations:

- SRL was not installed into the active environment.
- The environment was installed but not activated in the current shell.
- You are using a different Python than the one that owns the installed package.

You can also use the module fallback from source:

```bash
python -m srl.cli.train --help
python -m srl.cli.benchmark --help
python -m srl.cli.visualize --help
```

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
