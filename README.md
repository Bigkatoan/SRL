# SRL — Simple Reinforcement Learning

SRL is a modular PyTorch reinforcement learning library for continuous-control environments with YAML-defined models and a Python-first workflow.

## What is included

- Algorithms: PPO, SAC, DDPG, A2C, A3C
- Environment adapters: Gymnasium, Box2D, MuJoCo, Fetch robotics, Isaac Lab, racecar_gym
- Config-driven model builder for MLP, CNN, and multi-input policies
- CLI training entrypoint: `srl-train`
- Optional ROS 2 Python API for inference

## Install

`srl-rl` is not published on PyPI yet.

```bash
# recommended
pip install git+https://github.com/Bigkatoan/SRL.git

# or clone locally
git clone https://github.com/Bigkatoan/SRL.git
cd SRL
pip install -e .
```

Optional extras:

```bash
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[mujoco]"
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[box2d]"
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[robotics]"
pip install "git+https://github.com/Bigkatoan/SRL.git#egg=srl-rl[all]"
```

## Quick start

```bash
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --algo ppo \
          --steps 100000 \
          --device cpu
```

## Supported environments

| Suite | Examples | Wrapper |
|---|---|---|
| Gymnasium classic | Pendulum, MountainCarContinuous | `GymnasiumWrapper` |
| Box2D | BipedalWalker, LunarLanderContinuous, CarRacing | `GymnasiumWrapper` |
| MuJoCo | HalfCheetah, Ant, Hopper, Walker2d, Humanoid, Swimmer, Pusher, Reacher | `GymnasiumWrapper` |
| Robotics | FetchReach, FetchPush, FetchPickAndPlace, FetchSlide | `GoalEnvWrapper` |
| Isaac Lab | Isaac-Cartpole, Isaac-Ant, Isaac-Humanoid | `IsaacLabWrapper` |
| racecar_gym | SingleAgentAustria | `RacecarWrapper` |

## Testing

Deep environment and algorithm validation lives in [tests/test_deep_env_algorithms.py](tests/test_deep_env_algorithms.py) and can be run with:

```bash
bash scripts/run_deep_env_tests.sh
```

## Documentation

- Docs home: https://bigkatoan.github.io/SRL
- Installation: https://bigkatoan.github.io/SRL/installation
- Environments: https://bigkatoan.github.io/SRL/environments/
- Config reference: https://bigkatoan.github.io/SRL/config_reference
- ROS 2 Python API: https://bigkatoan.github.io/SRL/ros2

## License

MIT. See [LICENSE](LICENSE).