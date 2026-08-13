# Isaac Lab Integration

[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) is NVIDIA's GPU-accelerated robot
learning framework. SRL integrates with it natively through `IsaacLabWrapper`.

## Requirements

- NVIDIA GPU (RTX 3090 or better recommended)
- Isaac Sim ≥ 5.1 and Isaac Lab ≥ 0.5
- Python 3.10 or 3.11
- SRL installed into the *same* Isaac Lab Python environment

## Setup

### 1. Install Isaac Lab

Follow the [Isaac Lab official guide](https://isaac-sim.github.io/IsaacLab/).

### 2. Install SRL into the Isaac Lab environment

```bash
# Activate the Isaac Lab environment first
source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh
conda activate isaaclab

# Install SRL
pip install git+https://github.com/Bigkatoan/SRL.git

# Verify
python -c "import isaaclab; import srl; print('OK')"
```

```{admonition} Order matters
:class: warning
SRL has to be installed into the Isaac Lab environment. If it lands in a different
environment the console script may still be on `PATH`, but the `isaaclab` imports will
fail at runtime.

1. Activate the Isaac Lab environment.
2. Confirm `python -c "import isaaclab"` passes.
3. Install SRL into that environment.
4. Run `srl-train` from that same shell.
```

## Observation routing

`IsaacLabWrapper` preserves Isaac Lab's observation *group* names as dict keys —
commonly `policy` and `critic`. There are two ways to route them into encoders:

**Explicit (recommended)** — set `input_name` to the obs group key:

```yaml
encoders:
  - name: policy_enc
    type: mlp
    input_name: policy          # ← must match the Isaac Lab obs group key
    input_dim: 60
    latent_dim: 256
```

A missing `input_name` key raises `KeyError`, and obs keys left unused after explicit
routing emit a warning — both of which are easier to debug than a silent mismatch.

**Implicit** — leave `input_name` unset and let the CLI remap by count: one obs key is
broadcast to every unnamed encoder, and N obs keys are zipped in order onto N encoders.
The shipped Isaac Lab configs rely on this (they declare `actor_state_enc` and
`critic_state_enc` with no `input_name`).

You can inspect an env's group keys with `print(env.observation_space)` after building
the task.

## PPO on Isaac Lab (recommended)

```yaml
# configs/envs/isaaclab_ant_ppo.yaml
env_id:   Isaac-Ant-v0
env_type: isaaclab
algo:     ppo

encoders:
  - name: actor_state_enc
    type: mlp
    input_dim: 60
    latent_dim: 256
    layers:
      - {out_features: 256, activation: elu, norm: none}
      - {out_features: 128, activation: elu, norm: none}
  - name: critic_state_enc
    type: mlp
    input_dim: 60
    latent_dim: 256
    layers:
      - {out_features: 256, activation: elu, norm: none}
      - {out_features: 128, activation: elu, norm: none}

flows:
  - "actor_state_enc -> actor"
  - "critic_state_enc -> critic"

actor:
  name: actor
  type: gaussian
  action_dim: 8

critic:
  name: critic
  type: value

train:
  total_steps:   5_000_000
  n_envs:        4096
  n_steps:       32
  batch_size:    16384
  n_epochs:      5
  lr:            5e-4
  vf_coef:       1.0
  entropy_coef:  0.005
  max_grad_norm: 1.0
  gae_lambda:    0.95
  gamma:         0.99
```

```bash
srl-train --config configs/envs/isaaclab_ant_ppo.yaml --device cuda
```

Isaac Lab batches envs internally, so SRL builds one wrapped env rather than a
`SyncVectorEnv`/`AsyncVectorEnv` stack — `--n-envs` is passed through to
`parse_env_cfg(num_envs=...)`.

The other shipped Isaac Lab configs are
[isaaclab_cartpole_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_cartpole_ppo.yaml)
and
[isaaclab_humanoid_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_humanoid_ppo.yaml).

(sac-async-gpu-buffer)=
## SAC with the async runner and GPU buffer

`AsyncOffPolicyRunner` keeps collection on the main thread — which Isaac Lab's
simulation context requires — and moves gradient updates onto a daemon thread with its
own CUDA stream. Pair it with `GPUReplayBuffer` to keep the sampled batches on device:

```python
from srl.core.config import AsyncRunnerConfig
from srl.runners import AsyncOffPolicyRunner

runner = AsyncOffPolicyRunner(
    agent=sac_agent,
    env=isaac_env,
    total_steps=1_000_000,
    runner_cfg=AsyncRunnerConfig(use_async=True, use_gpu_buffer=True),
    device="cuda:0",
)
runner.run()
```

```{warning}
This is a Python-API path. `use_async` / `use_gpu_buffer` in a YAML `train:` block are
ignored — `srl-train` never constructs an `AsyncRunnerConfig`, and its Isaac Lab path
goes through the synchronous runner with the CPU buffer. `IsaacLabWrapper` also
converts observations to CPU numpy, so an end-to-end copy-free path means feeding the
buffer CUDA tensors yourself.
```

## Tuning tips

- Prefer `elu` activations — smoother gradients than `relu` on these tasks.
- `vf_coef: 1.0` balances policy and value loss well on Isaac Lab.
- Scale `n_envs` before `n_steps` when you have the VRAM.
- Keep `max_grad_norm` around 1.0 at large batch sizes.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: isaaclab` | Wrong environment | Activate the correct Isaac Lab env |
| `ModuleNotFoundError: pxr` | Isaac Sim not set up | Run `source setup_conda_env.sh` first |
| `KeyError: Missing observation key '...'` | `input_name` does not match an obs group | Check `env.observation_space` keys |
| `--visualize` prints a warning and does nothing | Not supported for Isaac Lab | One process hosts a single Isaac Sim render context; see {ref}`the CLI reference <live-viewer-visualize>` |

## See also

- [Replay Buffers](../training/buffers.md)
- [Runners & Training Loop](../training/runners.md)
- [Algorithms](../algorithms.md)
- [Isaac Lab Environments](../environments/isaaclab.md)
- [mjlab Integration](mjlab.md) — the same GPU-batched shape without an Isaac Sim install
