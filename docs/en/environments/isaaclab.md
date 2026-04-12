# Isaac Lab Environments

[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) is NVIDIA's GPU-accelerated
robot learning framework built on Isaac Sim.

---

## Requirements

- NVIDIA GPU (RTX 3090 or better recommended)
- Isaac Sim ≥ 5.1
- Isaac Lab ≥ 0.5
- Python 3.10 or 3.11

---

## Supported environments

| Env | obs | act | n_envs | Steps |
|---|---|---|---|---|
| Isaac-Cartpole-v0 | 4 | 1 | 512 | ~500k |
| Isaac-Ant-v0 | 60 | 8 | 4 096 | ~5M |
| Isaac-Humanoid-v0 | 87 | 21 | 4 096 | ~10M |

---

## Training

The recommended package workflow is YAML + `srl-train`, not only the standalone example script.

```bash
# Activate Isaac Lab environment first
source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh
conda activate isaaclab

# Install SRL
pip install git+https://github.com/Bigkatoan/SRL.git

# Verify CLI in the active Isaac Lab environment
srl-train --help

# Train with YAML configs
srl-train --config configs/envs/isaaclab_cartpole_ppo.yaml \
          --env Isaac-Cartpole-v0 \
          --algo ppo \
          --device cuda

srl-train --config configs/envs/isaaclab_ant_ppo.yaml \
          --env Isaac-Ant-v0 \
          --algo ppo \
          --n-envs 4096 \
          --device cuda

srl-train --config configs/envs/isaaclab_humanoid_ppo.yaml \
          --env Isaac-Humanoid-v0 \
          --algo ppo \
          --n-envs 4096 \
          --device cuda
```

The example script path still exists, but the CLI path matches the package's YAML-first workflow more closely.

## Bootstrap order

Isaac Lab is sensitive to import order and active environment selection. The stable order is:

1. Activate the Isaac Sim / Isaac Lab environment
2. Confirm `python -c "import isaaclab"` works in that shell
3. Install SRL into that same environment
4. Run `srl-train` from that same shell

If you install SRL into one Python environment and activate Isaac Lab in another, the CLI may exist but imports will still fail at runtime.

## Env-id normalization

The training CLI normalizes Isaac Lab names to the internal `isaaclab:<task>` form when the environment family is Isaac Lab.

That means these user-facing values are treated as equivalent in the CLI path:

- `Isaac-Cartpole-v0`
- `isaaclab:Isaac-Cartpole-v0`

Use the plain task name in docs and commands unless you are debugging lower-level environment bootstrapping.

## Runtime notes

- Isaac Lab bootstrap must happen from a Python environment where Isaac Lab and Isaac Sim are already activated.
- Isaac Lab's internal vectorization is distinct from SRL's sync/async Gymnasium vectorization modes.
- The current integration assumes the task name, config, and observation routing conventions stay aligned with the provided YAML files.
- Observation groups returned by Isaac Lab are preserved as dict keys by [isaac_lab_wrapper.py](https://github.com/Bigkatoan/SRL/blob/main/srl/envs/isaac_lab_wrapper.py). In practice, encoder names or `input_name` mappings must match those observation-group keys.
- Image observations returned as HWC are transposed to CHW to match SRL CNN encoder expectations.

## M3bot on this machine

The separate `M3bot` task repository can be validated alongside SRL. Replace the paths below with the locations used on your machine.

Verified paths (example placeholders):

- M3bot repo: `tests/M3bot` (or /path/to/tests/M3bot)
- Isaac runtime: `/path/to/isaaclab/venv/bin/python`

Verified commands (adjust paths to your layout):

```bash
cd tests/M3bot
python3 tools/validate_source.py

OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python \
  train.py --task Isaac-M3-Reach-v0 --headless --num_envs 64 --max_iterations 1

OMNI_KIT_ACCEPT_EULA=YES /path/to/isaaclab/venv/bin/python \
  play.py --task Isaac-M3-Reach-v0 --headless \
  --checkpoint logs/rsl_rl/m3_reach/<timestamp>/model_0.pt \
  --video --video_length 8
```

Current local status:

- state tasks: verified runnable
- checkpoint load + JIT/ONNX export: verified runnable
- vision tasks: not yet verified on the current Isaac runtime because the local `isaaclab_rl.rsl_rl` API does not expose the model-config classes expected by the `M3bot` vision PPO configs

For the full task-specific installation and validation notes, see [M3bot](m3bot.md).

---

## IsaacLabWrapper

```python
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

env = IsaacLabWrapper("Isaac-Cartpole-v0", num_envs=512)
obs, _ = env.reset()
# obs = {"state": tensor(512, 4)}   ← GPU tensor converted to numpy

action = env.act_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

---

## PPO config for Isaac Lab

Isaac Lab envs benefit from:

- **Large `n_envs`** (512–4096): GPU parallelism
- **Short `n_steps`** (16–32): Fast inner loops
- **Large `batch_size`** (8k–32k): Efficient GPU utilisation
- **`elu` activations**: Smoother gradients than `tanh`
- **`vf_coef=1.0`**: Equal policy/value loss weighting

```yaml
train:
  total_steps: 5_000_000
  n_envs: 4096
  n_steps: 32
  batch_size: 16384
  n_epochs: 5
  lr: 5e-4
  entropy_coef: 0.005
  vf_coef: 1.0
  max_grad_norm: 1.0
```

Reference configs in this repo:

- [isaaclab_cartpole_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_cartpole_ppo.yaml)
- [isaaclab_ant_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_ant_ppo.yaml)
- [isaaclab_humanoid_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_humanoid_ppo.yaml)

## Troubleshooting

### `ModuleNotFoundError` for Isaac Lab packages

You are not running inside the Isaac Lab Python environment, or SRL was installed into a different interpreter.

### CLI works but training still fails during startup

That usually means the console script exists, but Isaac Sim or Isaac Lab dependencies are not importable in the active shell.

### Wrong observation routing

Inspect the observation-group keys returned by the environment and ensure your YAML encoder names or explicit `input_name` values line up with those keys.

### Large `n_envs` is unstable on your machine

Reduce `n_envs`, batch size, and rollout length together. Isaac Lab configs assume substantial GPU memory and fast device-side stepping.

### M3bot state tasks run but vision tasks fail during config import

That usually means your local Isaac Lab runtime does not expose the newer `isaaclab_rl.rsl_rl` vision model-config API expected by the task repository. In that case, keep state-task validation separate from vision-task readiness and either upgrade the Isaac runtime or migrate the vision PPO config layer.
