# Replay Buffers

SRL ships two replay buffers: a CPU/numpy one (the default) and a CUDA-resident one
for GPU simulators.

## CPU `ReplayBuffer` (default)

A standard circular buffer that stores transitions as numpy arrays.

```python
from srl.core.replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=1_000_000)
buf.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
batch = buf.sample(256)
```

`obs_shape` and `action_dim` are optional — the buffer allocates lazily from the first
transition you add. This is the right choice for almost every workload.

```{warning}
Pass the transition fields by keyword. `ReplayBuffer.add` and `GPUReplayBuffer.add`
have different positional orders (`(obs, action, reward, next_obs, done)` versus
`(obs, action, reward, done, next_obs)`), so positional calls do not transfer between
the two.
```

`done` must be **true termination only**, never OR'd with `truncated`. SAC, DDPG, and
TD3 bootstrap the target as `(1 - done) * next_q`, so folding time-limit cutoffs into
`done` biases Q-values low on every env with an episode limit. Pass `truncated`
separately.

## `GPUReplayBuffer` (v0.2.0)

`GPUReplayBuffer` is a pre-allocated CUDA circular buffer aimed at GPU-batched
simulators.

### The problem with the CPU buffer

```
CPU ReplayBuffer:
  GPU sim (CUDA) → numpy copy (host) → store → batch.to(device) → GPU
  ↑ host↔device round-trips on both the store and the sample path
```

### The fix

```
GPUReplayBuffer:
  GPU sim (CUDA) → GPUReplayBuffer (CUDA) → batch already on the GPU
  ↑ no host↔device copies
```

`add()` accepts CUDA tensors and writes them through
`Tensor.copy_(src, non_blocking=True)` on a dedicated copy stream; `sample()` returns a
`ReplayBatch` whose tensors already live on `device`, so no `.to(device)` is needed in
the training loop. It also takes numpy arrays via `torch.as_tensor`, and `add()` is
thread-safe, which is what makes it usable from the async runner's collector thread.

```{note}
Getting the end-to-end zero-copy path requires feeding the buffer CUDA tensors
yourself. The CLI's `IsaacLabWrapper` converts observations to CPU numpy before they
reach the buffer, so the copy-free path today is a Python-API one.
```

### Usage

```python
from srl.core.gpu_replay_buffer import GPUReplayBuffer

buf = GPUReplayBuffer(
    capacity=1_000_000,
    device="cuda:0",
)

buf.add(
    obs={"pixels": obs_tensor, "state": state_tensor},
    action=action_tensor,
    reward=reward_tensor,
    done=done_tensor,
    next_obs={"pixels": next_pixels, "state": next_state},
)

batch = buf.sample(256)
# batch.obs["pixels"]: (256, C, H, W) on cuda:0
# batch.actions:       (256, action_dim) on cuda:0
```

### Constructor arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `capacity` | `int` | required | Maximum number of transitions |
| `device` | `str \| torch.device` | `"cuda"` | Device all storage is allocated on |
| `n_step` | `int` | `1` | N-step return horizon (1 = standard 1-step TD) |
| `gamma` | `float` | `0.99` | Discount used for n-step returns |
| `use_fp16` | `bool` | `False` | Store float32 obs/actions as float16 to halve VRAM |
| `num_envs` | `int` | `1` | Number of parallel envs writing to the buffer |

### Checkpointing

`GPUReplayBuffer` serialises to CPU tensors, so a saved buffer restores onto any
device:

```python
state = buf.state_dict()    # CPU tensors, portable
buf.load_state_dict(state)  # restore on any device
```

## Pairing with the async runner

`AsyncOffPolicyRunner` swaps the agent's CPU buffer for a `GPUReplayBuffer` when
`use_gpu_buffer=True`, carrying `capacity`, `n_step`, `gamma`, `use_fp16`, and the env
count over from the old buffer:

```python
from srl.core.config import AsyncRunnerConfig
from srl.runners import AsyncOffPolicyRunner

runner = AsyncOffPolicyRunner(
    agent=sac_agent,
    env=env,
    total_steps=1_000_000,
    runner_cfg=AsyncRunnerConfig(use_async=True, use_gpu_buffer=True),
    device="cuda:0",
)
runner.run()
```

With that configuration:

1. The collector (main thread) writes CUDA tensors straight into the `GPUReplayBuffer`.
2. The trainer (daemon thread) samples batches that are already on the GPU.
3. There are no CPU↔GPU copies in the hot path.

```{note}
`use_async` and `use_gpu_buffer` are Python-API settings. Putting them in a YAML
`train:` block currently has no effect, because `srl-train` never builds an
`AsyncRunnerConfig` from it.
```

## See also

- [GPU Replay Buffer API](../gpu_replay_buffer.md)
- [Runners & Training Loop](runners.md)
- [Isaac Lab Integration](../integrations/isaaclab.md)
