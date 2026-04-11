# CLI Reference

SRL exposes three command-line entry points:

- `srl-train`
- `srl-benchmark`
- `srl-visualize`

These commands are declared in [pyproject.toml](/home/ubuntu/antd/SRL/pyproject.toml) and become available after SRL is installed into the active Python environment.

## Before you run anything

The most common CLI failure is not a bug in the command itself. It is usually one of these two cases:

1. The package is not installed into the current environment, so the console script does not exist yet.
2. You are running from a Python environment that does not have SRL dependencies installed.

Recommended verification steps:

```bash
python -m pip show srl-rl
command -v srl-train
command -v srl-benchmark
command -v srl-visualize

srl-train --help
srl-benchmark --help
srl-visualize --help
```

If the console script is missing but the source tree is present, the module fallback also works:

```bash
python -m srl.cli.train --help
python -m srl.cli.benchmark --help
python -m srl.cli.visualize --help
```

Use console scripts as the canonical interface. Use `python -m ...` mainly for debugging or when working directly from source.

## `srl-train`

`srl-train` is the main training entry point.

```bash
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --algo ppo \
          --steps 100000 \
          --device cpu
```

Important flag groups from [train.py](/home/ubuntu/antd/SRL/srl/cli/train.py):

- Core inputs: `--config`, `--env`, `--algo`, `--steps`, `--n-envs`, `--device`
- Vectorization: `--vec-mode auto|sync|async`
- Logging and artifacts: `--logdir`, `--ckptdir`, `--log-interval`, `--episode-window`, `--console-layout`, `--plot-metrics`, `--no-plots`
- Checkpointing and resume: `--resume`
- Pipeline export: `--save-model-pipeline`, `--save-training-pipeline`, `--export-pipeline-only`
- Evaluation: `--eval-freq`, `--eval-episodes`, `--render`

Example: resume from a checkpoint.

```bash
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --algo ppo \
          --steps 200000 \
          --resume checkpoints/ppo_pendulum_ppo/final_0000100000.pt
```

Example: export pipelines without training.

```bash
srl-train --config configs/envs/halfcheetah_sac.yaml \
          --save-model-pipeline \
          --save-training-pipeline \
          --export-pipeline-only
```

## `srl-benchmark`

`srl-benchmark` runs short benchmark cases across vectorization modes and writes structured results.

```bash
srl-benchmark --config configs/envs/halfcheetah_sac.yaml \
              --env HalfCheetah-v5 \
              --modes sync,async \
              --n-envs 4
```

Important flags from [benchmark.py](/home/ubuntu/antd/SRL/srl/cli/benchmark.py):

- Inputs: `--config`, `--env`, `--algo`
- Budget and scaling: `--steps`, `--n-envs`, `--modes`, `--device`
- Reporting: `--target-file`, `--output`
- Eval during benchmark: `--eval-freq`, `--eval-episodes`

Supported modes:

- `single`
- `sync`
- `async`
- `isaac`

`srl-benchmark` is useful for comparing throughput and basic training behavior, not for replacing long-form experiment tracking.

## `srl-visualize`

`srl-visualize` renders model and training pipeline PNGs from a YAML config without launching a training run.

```bash
srl-visualize --config configs/envs/halfcheetah_sac.yaml \
              --output-dir runs/pipelines
```

Important flags from [visualize.py](/home/ubuntu/antd/SRL/srl/cli/visualize.py):

- Inputs: `--config`, `--env`, `--algo`
- Output control: `--output-dir`, `--model-output`, `--training-output`

This command is the fastest way to inspect whether a YAML graph says what you think it says.

## Environment-specific notes

- Gymnasium, MuJoCo, Box2D, and robotics configs are normally run through `srl-train` after standard package installation.
- Isaac Lab requires activation of the Isaac Lab Python environment first. See [Isaac Lab](isaaclab.md) for the environment-specific workflow.
- ROS 2 deployment is not driven through these CLI tools. See [ROS 2 Python API](ros2.md).

## Common failure modes

### `srl-train: command not found`

The package is not installed into the active shell environment, or the shell has not picked up the environment's `bin` directory.

### `ModuleNotFoundError` for `torch`, `gymnasium`, or similar

You are running from a Python environment that does not contain SRL runtime dependencies.

### Config or algorithm mismatch

`srl-train` validates model-head compatibility against the selected algorithm. For example, SAC expects a squashed Gaussian actor and a twin-Q critic.

### Isaac Lab import failures

The Isaac Lab stack must be installed and activated before running Isaac Lab configs.