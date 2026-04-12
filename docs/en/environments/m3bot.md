# M3bot Environment

`M3bot` is a task suite built on Isaac Lab for a 4-DOF arm + gripper robot with Reach, Lift, Push, and PickPlace tasks in both state-based and vision-augmented variants.

This page documents how `M3bot` was validated on the current machine and how it relates to SRL's broader Isaac Lab workflow.

## Repository and runtime layout

Validated local paths on the current machine (example placeholders):

- M3bot checkout: `tests/M3bot` (or /path/to/tests/M3bot)
- Isaac runtime interpreter: `/path/to/isaaclab/venv/bin/python`

The `M3bot` repository is separate from the `SRL` repository. In practice, the working flow on this machine is:

1. Keep `SRL` as the library/docs repo.
2. Keep `M3bot` as the task/environment repo.
3. Run `M3bot` with the Isaac Lab Python runtime, not a random system Python.

## Installation path for M3bot

Clone the task repo into a separate workspace folder:

```bash
cd tests
git clone https://github.com/Bigkatoan/M3bot.git
cd M3bot
```

If the repository depends on an Isaac Lab submodule, make sure it is actually populated:

```bash
git submodule update --init --recursive
```

On the current machine, the verified runtime is the pre-existing Isaac Lab environment at (example):

```bash
/path/to/isaaclab/venv/bin/python
```

For non-interactive terminal runs, add:

```bash
OMNI_KIT_ACCEPT_EULA=YES
```

## Verified local checks

### Cheap validation

```bash
cd tests/M3bot
python3 tools/validate_source.py

OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python train.py --help
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python play.py --help
```

### Verified smoke training

The following state-task smoke run completed successfully on this machine:

```bash
cd tests/M3bot
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python \
  train.py --task Isaac-M3-Reach-v0 --headless --num_envs 64 --max_iterations 1
```

Observed artifacts:

- `logs/rsl_rl/m3_reach/<timestamp>/model_0.pt`
- `logs/rsl_rl/m3_reach/<timestamp>/params/env.yaml`
- `logs/rsl_rl/m3_reach/<timestamp>/params/agent.yaml`

### Verified checkpoint load and export

The following play/export path also completed successfully:

```bash
cd tests/M3bot
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python \
  play.py --task Isaac-M3-Reach-v0 --headless \
  --checkpoint logs/rsl_rl/m3_reach/<timestamp>/model_0.pt \
  --video --video_length 8
```

Observed artifacts:

- `logs/rsl_rl/m3_reach/<timestamp>/exported/policy.pt`
- `logs/rsl_rl/m3_reach/<timestamp>/exported/policy.onnx`
- `logs/rsl_rl/m3_reach/<timestamp>/videos/play/`

## Current limitations on this machine

### State tasks are verified

The current local runtime is good enough for state-based `M3bot` tasks.

### Vision tasks are not yet verified

The current Isaac Lab runtime on this machine does not expose `RslRlCNNModelCfg` and `RslRlMLPModelCfg` from `isaaclab_rl.rsl_rl`.

That means:

- state-based tasks can run locally
- vision-task PPO configs are currently blocked by an Isaac Lab API mismatch
- vision bring-up needs either a matching newer Isaac runtime or a compatibility migration in the task repo

### Runtime warnings still worth fixing

The validated state-task smoke run still produced warnings that should be resolved before calling the stack production-grade:

- invalid or incomplete rigid-body mass/inertia on the robot end effector
- point-instancer prototype mismatch warnings under `/Visuals/Command/*`
- `obs_groups` fallback warnings from `rsl_rl`

## Why this matters for SRL docs

`M3bot` is a concrete example of how Isaac Lab-based task repos interact with SRL-adjacent tooling and environment expectations.

The important operational lessons are:

- pin the exact Isaac runtime, not just Python version
- keep task repos separate from the core RL library when appropriate
- document verified machine paths explicitly
- separate "state tasks verified" from "vision tasks intended but not yet verified"

## Related docs

- [Installation](../installation.md)
- [Isaac Lab](isaaclab.md)
- [ROS 2](../ros2.md)