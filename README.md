# SRL — Simple Reinforcement Learning

[![Tests](https://github.com/Bigkatoan/SRL/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/Bigkatoan/SRL/actions/workflows/test.yml)
[![Docs](https://github.com/Bigkatoan/SRL/actions/workflows/docs.yml/badge.svg?branch=main)](https://bigkatoan.github.io/SRL)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
<!-- Uncomment once srl-rl is published on PyPI:
[![PyPI](https://img.shields.io/pypi/v/srl-rl.svg)](https://pypi.org/project/srl-rl/)
-->

SRL is a modular reinforcement learning library for continuous-control environments with a YAML-first model system.

```python
# From a clone of this repo (configs/ lives at the repo root).
# Equivalent to: srl-train --config configs/envs/pendulum_ppo.yaml --steps 100000
from srl.cli.train import main as srl_train

srl_train([
    "--config", "configs/envs/pendulum_ppo.yaml",  # model graph + hyperparameters
    "--steps", "100000",
    "--device", "auto",                            # cuda if available, else cpu
])
# -> runs/ppo_pendulum_ppo/{summary.json,history.csv,...}
# -> checkpoints/ppo_pendulum_ppo/final_0000100000/   (srl-agent format v1)
```

Run the trained policy (numpy in, env-unit actions out):

```python
import gymnasium as gym
import srl

policy = srl.load_policy("checkpoints/ppo_pendulum_ppo")
env = gym.make("Pendulum-v1")
obs, _ = env.reset(seed=0)
obs, reward, terminated, truncated, _ = env.step(policy(obs))
```

The Python training API (`PPO.from_config(...).learn(...)`), checkpoint format,
TorchScript export and ROS 2 deployment are documented in the
[docs](https://bigkatoan.github.io/SRL).

## What is included

- Algorithms: PPO, SAC, DDPG, TD3, A2C, A3C
- Environment adapters: Gymnasium, Box2D, MuJoCo, Fetch robotics, Isaac Lab, mjlab, racecar_gym
- YAML-driven model graph builder for encoders, heads, flows, and multimodal policies
- CLI tools: `srl-train`, `srl-benchmark`, `srl-visualize`, `srl-policy` (inspect / TorchScript-export a checkpoint)
- Pipeline visualization export for model graphs and training graphs
- One checkpoint format (`srl-agent` v1: safetensors weights + resolved config, pickle-free) for resume, `srl.load_policy` inference and ROS 2 deployment
- Optional ROS 2 Python API for inference
- **v0.2.0** — encoder optimizer fix, async off-policy runner, GPU replay buffer, expanded encoder loss modes (AE/VAE/CURL/BYOL/DrQ/SPR/Barlow Twins)

## Install

`srl-rl` is not published on PyPI yet.

```bash
# recommended -- also gets you configs/, which every example below needs
git clone https://github.com/Bigkatoan/SRL.git
cd SRL
pip install -e .
```

> **Important:** `pip install git+https://github.com/Bigkatoan/SRL.git` (no clone) installs
> the `srl` Python package and CLI entry points, but **not** the repo-root
> `configs/` directory — it isn't part of the installed package. Every
> `--config configs/envs/...` example in this README and the docs assumes
> `configs/` is present, so unless you're supplying your own YAML from
> scratch, clone the repo rather than installing straight from the git URL.

```bash
# package-only install, if you don't need the example configs
pip install git+https://github.com/Bigkatoan/SRL.git
```

Optional extras:

```bash
pip install "srl-rl[mujoco] @ git+https://github.com/Bigkatoan/SRL.git"
pip install "srl-rl[box2d] @ git+https://github.com/Bigkatoan/SRL.git"
pip install "srl-rl[robotics] @ git+https://github.com/Bigkatoan/SRL.git"
pip install "srl-rl[all] @ git+https://github.com/Bigkatoan/SRL.git"
```

After install, verify the CLI entry points in the same environment:

```bash
srl-train --help
srl-benchmark --help
srl-visualize --help
```

If those commands are missing, SRL is not installed into the currently active environment yet. See the installation guide and CLI reference for the fallback `python -m ...` path.

`racecar_gym` is not on PyPI; install it manually before using the `racecar` extra (see the [installation guide](https://bigkatoan.github.io/SRL/source/installation.html)).

## Quick start

The most important concept in SRL is the YAML model graph. Treat the config file as the source of truth for model structure, observation routing, and the currently supported declarative parts of training.

Start here before diving into algorithms or CLI flags:

- YAML core guide: https://bigkatoan.github.io/SRL/source/yaml_core.html
- CLI reference: https://bigkatoan.github.io/SRL/source/cli.html
- Config reference: https://bigkatoan.github.io/SRL/source/config_reference.html
- Quick start: https://bigkatoan.github.io/SRL/source/quickstart.html

```bash
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --algo ppo \
          --steps 100000 \
          --device cpu \
          --log-interval 4096 \
          --episode-window 20 \
          --plot-metrics train/score_mean,ppo/total
```

Training runs now export:

- compact terminal summaries with score, rolling score, episode length, throughput, and algorithm metrics
- TensorBoard scalars under `runs/...`
- `summary.json`, `history.csv`, `metrics.jsonl`, and `training_curves.svg` after training
- optional PNG pipeline graphs for the model and training flow

You can disable plot export with `--no-plots` or choose specific curves with `--plot-metrics`.

Useful CLI examples:

```bash
# export pipeline graphs without training
srl-visualize --config configs/envs/halfcheetah_sac.yaml --output-dir runs/pipelines

# save pipeline graphs before training
srl-train --config configs/envs/halfcheetah_sac.yaml \
          --save-model-pipeline \
          --save-training-pipeline

# resume from a checkpoint created by srl-train
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --algo ppo \
          --steps 200000 \
          --resume checkpoints/ppo_pendulum_ppo/final_0000100000

# compare sync vs async vectorization locally
srl-benchmark --config configs/envs/halfcheetah_sac.yaml \
              --env HalfCheetah-v5 \
              --modes sync,async \
              --n-envs 4
```

## Algorithms and features

| | PPO | A2C | A3C | SAC | DDPG | TD3 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Policy type | on-policy | on-policy | on-policy (async workers) | off-policy | off-policy | off-policy |
| Vectorized envs (`--n-envs`) | ✓ | ✓ | workers | ✓ | ✓ | ✓ |
| Pixel observations + auxiliary encoder losses (AE/CURL/…) | ✓ | – | – | ✓ | – | – |
| Hindsight Experience Replay (`use_her`) | – | – | – | ✓ | – | – |
| n-step returns (`replay_n_step`) | – | – | – | ✓ | ✓ | ✓ |
| Async off-policy runner / GPU replay buffer | – | – | – | ✓ | ✓ | ✓ |
| Automatic entropy / temperature tuning | ✓ (target entropy) | – | – | ✓ | – | – |
| Adaptive KL learning-rate schedule | ✓ | – | – | – | – | – |

All algorithms target continuous (Box) action spaces and are configured from the same YAML model graph. See the [algorithms guide](https://bigkatoan.github.io/SRL/source/algorithms.html) for details and the [limitations page](https://bigkatoan.github.io/SRL/source/limitations.html) for known gaps.

## Supported environments

| Suite | Examples | Wrapper |
|---|---|---|
| Gymnasium classic | Pendulum, MountainCarContinuous | `GymnasiumWrapper` |
| Box2D | BipedalWalker, LunarLanderContinuous, CarRacing | `GymnasiumWrapper` |
| MuJoCo | HalfCheetah, Ant, Hopper, Walker2d, Humanoid, Swimmer, Pusher, Reacher | `GymnasiumWrapper` |
| Robotics | FetchReach, FetchPush, FetchPickAndPlace, FetchSlide | `GoalEnvWrapper` |
| Isaac Lab | Isaac-Cartpole, Isaac-Ant, Isaac-Humanoid | `IsaacLabWrapper` |
| mjlab (`--env mjlab:<task>`) | any task registered via a project's own `mjlab.tasks` entry point | `IsaacLabWrapper` |
| M3bot task repo | Isaac-M3-Reach, Isaac-M3-Lift, Isaac-M3-Push, Isaac-M3-PickPlace | external Isaac Lab task repo |
| racecar_gym | SingleAgentAustria | `RacecarWrapper` |

## Testing

```bash
pip install -e ".[test]"
pytest -m "not slow and not gpu"   # fast unit suite (what CI runs)
pytest -m slow --timeout=3600      # end-to-end learning regressions (tests/test_learning.py)
```

Every shipped YAML config is checked by [tests/test_configs_parse.py](tests/test_configs_parse.py) (builds, runs a forward pass, and matches its env's spaces when the env is installed).

For a full config-matrix benchmark sweep with preserved per-case artifacts, use:

```bash
bash scripts/run_full_matrix_benchmark.sh --python "$(command -v python)" --label quick_check
```

That script iterates every YAML under `configs/envs`, continues through failures, records `passed` / `failed` / `blocked` per case, writes one master log under `matrix_runs/.../matrix.log`, keeps each case's `runs/` and `checkpoints/` outputs separately, and emits `cases.jsonl`, `report.json`, and `summary.md` with target-threshold judging. By default it runs in a convergence-oriented mode with a shorter shared budget and lighter targets from `configs/benchmarks/convergence_targets.yaml`. Use `--budget-mode full --target-file configs/benchmarks/core_targets.yaml` for the stricter beat-env gate:

```bash
bash scripts/run_full_matrix_benchmark.sh --skip-install \
    --budget-mode full \
    --target-file configs/benchmarks/core_targets.yaml \
    --label strict_gate
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, lint and docs commands.

## Documentation

- Docs home: https://bigkatoan.github.io/SRL
- Installation: https://bigkatoan.github.io/SRL/source/installation.html
- YAML core guide: https://bigkatoan.github.io/SRL/source/yaml_core.html
- CLI reference: https://bigkatoan.github.io/SRL/source/cli.html
- Environments: https://bigkatoan.github.io/SRL/source/environments/index.html
- Config reference: https://bigkatoan.github.io/SRL/source/config_reference.html
- Limitations: https://bigkatoan.github.io/SRL/source/limitations.html
- ROS 2 Python API: https://bigkatoan.github.io/SRL/source/ros2.html

## Citation

If you use SRL in your research, please cite it (see [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button exports BibTeX):

```bibtex
@software{tran_srl,
  author  = {Tran, Dang An},
  title   = {{SRL}: Simple Reinforcement Learning},
  url     = {https://github.com/Bigkatoan/SRL},
  license = {MIT}
}
```

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report security issues privately as described in [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
