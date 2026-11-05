# Saving, loading & deployment

Every SRL agent -- whether saved with `agent.save(...)` or written by
`srl-train` -- is stored in one directory format, **`srl-agent` format v1**.
The same directory serves three purposes:

| Use | API | Needs |
|---|---|---|
| resume training | `SAC.load(dir, env=...)` / `srl-train --resume dir` | everything (+ replay buffer for an exact off-policy resume) |
| inference | `srl.load_policy(dir)` | `meta.json`, `config.yaml`, `model.safetensors`, `normalizers.json` |
| deployment | TorchScript export, ROS 2 node | the same as inference |

## Format v1 layout

```text
my_agent/
  meta.json           format, format_version, SRL / torch versions, algo,
                      obs_spec, action_spec, normalizers present, counters
  config.yaml         resolved experiment config (explicit dims, every
                      hyperparameter) -- the single source of truth for the
                      model and the algorithm config
  model.safetensors   model weights
  normalizers.json    observation / reward running statistics (when used)
  train_state.pt      optimizers, target network, alpha, RNG, step counters
  replay_buffer/      only with include_replay_buffer=True
    state.pt
```

- **`config.yaml`** is re-validated with {func}`srl.config.load_experiment` on
  load: a typo or a key a newer SRL removed is reported, not ignored. Agents
  built without a config (a hand-built model) store `model.yaml` (the model
  spec) instead, or -- for fully custom `nn.Module`s -- a pickled `model.pt`.
- **`model.safetensors`** is loaded into the model built from `config.yaml`
  with a strict key check; a mismatch raises
  {class}`~srl.core.agent_io.CheckpointFormatError`.
- **`train_state.pt`** and **`replay_buffer/state.pt`** contain only tensors
  and plain Python values and are read with `torch.load(weights_only=True)`:
  loading a v1 checkpoint never unpickles arbitrary objects (the exception is
  the custom-model `model.pt` fallback, which warns).
- **`meta.json`** is written last; a directory without it is an incomplete
  write. Example:

```json
{
  "format": "srl-agent",
  "format_version": 1,
  "srl_version": "0.6.0",
  "torch_version": "2.6.0",
  "algo": "sac",
  "created": "2026-11-05T10:00:00+00:00",
  "num_timesteps": 20000,
  "obs_spec": {
    "obs_key": "state",
    "obs_shape": [3],
    "encoders": {
      "state_enc": {"type": "mlp", "input_name": null, "shape": [3],
                    "dtype": "float32", "recurrent": false}
    }
  },
  "action_spec": {
    "shape": [1], "low": [-2.0], "high": [2.0],
    "rescale": {"low": [-2.0], "high": [2.0], "finite": [true]},
    "clip": null
  },
  "normalizers": ["obs"],
  "model_source": "config.yaml",
  "includes_train_state": true,
  "includes_replay_buffer": false
}
```

`action_spec.rescale` is the affine map from the model's `[-1, 1]` action
domain to env units; `clip` is the range on-policy (Gaussian) actions are
clipped to before rescaling.

## Saving

```python
agent.save("runs/pendulum")                              # weights + config + train state
agent.save("runs/pendulum", include_replay_buffer=True)  # + replay buffer (exact resume)
agent.save("runs/pendulum", include_train_state=False)   # inference-only
```

`srl-train` writes the same directories under `--ckptdir/<run>/`:
`ckpt_<step>/` (periodic, every `train.save_interval` steps, last 5 kept),
`final_<step>/` and, with `--save-best`, `best_<step>/`. Periodic checkpoints
leave the replay buffer out by default -- it can be gigabytes and was
previously rewritten on every save; set `train.checkpoint_replay_buffer: true`
when you need `--resume` to continue with the exact buffer.

## Loading an agent

```python
from srl import SAC, BaseAgent

agent = SAC.load("runs/pendulum", env=gym.make("Pendulum-v1"))   # resume training
agent = BaseAgent.load("checkpoints/sac_pendulum_sac/final_0000020000")  # dispatch on meta.json
agent.load("runs/pendulum")                                       # in place
```

Older checkpoints still load: v0 directories (`agent.pt` + `model.yaml`) and
single-file `ckpt_*.pt` / `final_*.pt` / `best_*.pt` files. They are pickle
files, so they are read with `torch.load(weights_only=False)` after a
{class}`~srl.utils.safe_serialization.LegacyCheckpointWarning` -- only load
files you trust, and re-save them (`agent.save(dir)`) to upgrade.

A directory with an unknown `format`, a `format_version` newer than this SRL
reads, or a missing required file raises
{class}`~srl.core.agent_io.CheckpointFormatError` with the reason.

## Inference: `srl.load_policy`

```python
import gymnasium as gym
import srl

policy = srl.load_policy("checkpoints/sac_pendulum_sac")   # best, else final, else latest
env = gym.make("Pendulum-v1")
obs, _ = env.reset(seed=0)
done = False
while not done:
    obs, reward, terminated, truncated, _ = env.step(policy(obs))
    done = terminated or truncated
```

{func}`srl.load_policy` returns a {class}`srl.policy.Policy`: the model, the
observation normalizer statistics and the action rescale map -- no env,
optimizer, replay buffer or algorithm object. `policy(obs)` /
`policy.predict(obs, deterministic=True)` takes what the env returns (an
array, a batch of arrays, or a dict of arrays) and returns the action in env
units; it equals `agent.predict(obs, deterministic=True)[0]` for the agent
that was saved.

`path` may be an `agent.save()` directory, an `srl-train` checkpoint
directory (`checkpoint="auto" | "best" | "final" | "latest"`), a single
checkpoint inside one, or a legacy `.pt` file. Recurrent policies keep their
hidden state between calls; `policy.reset()` clears it.

### Export

```python
path = policy.export_torchscript("policy.ts")
module = torch.jit.load("policy.ts")         # no SRL needed at runtime
action = module(torch.as_tensor(obs[None], dtype=torch.float32))
```

The traced module contains the whole deterministic policy (normalize ->
actor -> clip -> rescale) and takes one batched `float32` tensor per entry of
`policy.input_keys`. The same from the command line:

```bash
python -m srl.policy info   checkpoints/sac_pendulum_sac
python -m srl.policy export checkpoints/sac_pendulum_sac --format torchscript -o policy.ts
```

ONNX export is not built in; `torch.onnx.export` on the module returned by
`torch.jit.load` (or on `srl.policy._ExportedPolicy(policy)`) works for
feed-forward MLP/CNN policies.

## ROS 2

The ROS 2 node (`srl.ros2.rl_node`) loads a policy with `srl.load_policy`, so
it works with any v1 checkpoint, including configs that used `auto` dims. See
[ROS 2](ros2.md).
