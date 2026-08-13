# SRL — Simple Reinforcement Learning

SRL is a modular reinforcement learning library for continuous-control environments with a YAML-first model system.

## What is included

- Algorithms: PPO, SAC, DDPG, TD3, A2C, A3C
- Environment adapters: Gymnasium, Box2D, MuJoCo, Fetch robotics, Isaac Lab, mjlab, racecar_gym
- YAML-driven model graph builder for encoders, heads, flows, and multimodal policies
- CLI tools: `srl-train`, `srl-benchmark`, `srl-visualize`
- Pipeline visualization export for model graphs and training graphs
- Checkpoint save and resume support from the training CLI
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

If you are also working with the separate `M3bot` Isaac Lab task repository on this machine, see the SRL docs pages for the verified environment layout and task-specific setup:

- Installation note: https://bigkatoan.github.io/SRL/source/installation.html
- M3bot environment guide: https://bigkatoan.github.io/SRL/source/environments/m3bot.html

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
          --resume checkpoints/ppo_pendulum_ppo/final_0000100000.pt

# compare sync vs async vectorization locally
srl-benchmark --config configs/envs/halfcheetah_sac.yaml \
              --env HalfCheetah-v5 \
              --modes sync,async \
              --n-envs 4
```

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

Deep environment and algorithm validation lives in [tests/test_deep_env_algorithms.py](tests/test_deep_env_algorithms.py) and can be run with:

```bash
bash scripts/run_deep_env_tests.sh
```

That runner now also executes headless Isaac Lab deep tests for PPO, A2C, SAC, DDPG, and TD3 when `tests/IsaacLab` is available.

For a full config-matrix benchmark sweep with preserved per-case artifacts, use:

```bash
bash scripts/run_full_matrix_benchmark.sh --python tests/venv/bin/python
```

That script iterates every YAML under `configs/envs`, continues through failures, records `passed` / `failed` / `blocked` per case, writes one master log under `matrix_runs/.../matrix.log`, keeps each case's `runs/` and `checkpoints/` outputs separately, and emits `cases.jsonl`, `report.json`, and `summary.md` with target-threshold judging. By default it now runs in a convergence-oriented mode with a shorter shared budget and lighter targets from `configs/benchmarks/convergence_targets.yaml` so you can see learning progress without waiting for full solved-policy training. Use `--budget-mode full --target-file configs/benchmarks/core_targets.yaml` when you want the stricter beat-env gate.

Examples:

```bash
# short convergence sweep with live logs
bash run_full_matrix_benchmark.sh --skip-install --python tests/venv/bin/python --label quick_check

# strict full-budget gate using YAML step counts and stricter thresholds
bash run_full_matrix_benchmark.sh --skip-install --python tests/venv/bin/python \
    --budget-mode full \
    --target-file configs/benchmarks/core_targets.yaml \
    --label strict_gate
```

## Documentation

- Docs home: https://bigkatoan.github.io/SRL
- Installation: https://bigkatoan.github.io/SRL/source/installation.html
- YAML core guide: https://bigkatoan.github.io/SRL/source/yaml_core.html
- CLI reference: https://bigkatoan.github.io/SRL/source/cli.html
- Environments: https://bigkatoan.github.io/SRL/source/environments/index.html
- M3bot environment guide: https://bigkatoan.github.io/SRL/source/environments/m3bot.html
- Config reference: https://bigkatoan.github.io/SRL/source/config_reference.html
- Limitations: https://bigkatoan.github.io/SRL/source/limitations.html
- ROS 2 Python API: https://bigkatoan.github.io/SRL/source/ros2.html

## License

MIT. See [LICENSE](LICENSE).