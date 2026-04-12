# GPU Replay Buffer

`GPUReplayBuffer` is a pre-allocated CUDA circular buffer designed for zero-copy
integration with Isaac Lab and other GPU-native simulators.

---

## Motivation

Standard `ReplayBuffer` stores data as numpy arrays on the CPU. When training with a
GPU-based simulator like Isaac Lab:

```
Isaac Lab (CUDA) → numpy copy (host) → ReplayBuffer (host) → batch.to(device) → GPU
```

One host↔device round-trip happens on every `add()` and another on every `sample()`.
`GPUReplayBuffer` eliminates both:

```
Isaac Lab (CUDA) → GPUReplayBuffer (CUDA) → batch already on GPU
```

---

## Quick start

```python
import torch
from srl.core.gpu_replay_buffer import GPUReplayBuffer

buf = GPUReplayBuffer(
    capacity = 1_000_000,
    device   = "cuda:0",   # or "cpu" for debugging
)

# add a transition (tensors already on CUDA — no copy)
buf.add(
    obs      = {"pixels": obs_tensor, "state": state_tensor},
    action   = action_tensor,
    reward   = reward_float,
    done     = done_bool,
    next_obs = {"pixels": next_pixels, "state": next_state},
)

# sample a batch
batch = buf.sample(256)
# batch.obs["pixels"] shape: (256, C, H, W)  — already on cuda:0
# batch.actions shape:       (256, action_dim)
# batch.rewards shape:       (256, 1)
# batch.dones shape:         (256, 1)
```

---

## API

### `GPUReplayBuffer(capacity, device, storage_dtype, n_step, gamma, num_envs)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `capacity` | `int` | required | Maximum number of transitions |
| `device` | `str \| torch.device` | `"cuda"` | Storage device |
| `storage_dtype` | `torch.dtype` | `float32` | Floating-point precision for obs/action |
| `n_step` | `int` | `1` | N-step return lookahead |
| `gamma` | `float` | `0.99` | Discount used for n-step accumulation |
| `num_envs` | `int` | `1` | Number of parallel environments for n-step bookkeeping |

### `.add(obs, action, reward, done, next_obs)`

Add one transition. Accepts any combination of:
- `obs` / `next_obs`: `torch.Tensor` or `dict[str, torch.Tensor]`
- `action`: `torch.Tensor` with shape `(action_dim,)` or `(1, action_dim)`
- `reward`: `float` or scalar tensor
- `done`: `bool` or scalar tensor

Tensors already on `device` are written via a dedicated non-blocking CUDA copy stream.
CPU tensors are moved to device automatically.

### `.sample(batch_size) → ReplayBatch`

Returns a `ReplayBatch` with fields:
- `obs` — same structure as input obs
- `next_obs` — same structure as input next_obs
- `actions` — shape `(B, action_dim)`
- `rewards` — shape `(B, 1)`
- `dones` — shape `(B, 1)`

All tensors live on `device`.

### `len(buf)`

Current number of stored transitions (up to `capacity`).

### `.state_dict()` / `.load_state_dict(state)`

Serialises buffer contents to CPU tensors for portable checkpointing.

---

## Dict observations

When `obs` is a `dict`, each key is stored in a separate pre-allocated tensor:

```python
buf = GPUReplayBuffer(capacity=100_000, device="cuda")
buf.add(
    obs      = {"rgb": rgb_tensor, "proprio": proprio_tensor},
    action   = action_tensor,
    reward   = 0.5,
    done     = False,
    next_obs = {"rgb": next_rgb, "proprio": next_proprio},
)
batch = buf.sample(256)
print(batch.obs["rgb"].shape)       # (256, 3, 64, 64)
print(batch.obs["proprio"].shape)   # (256, 24)
```

---

## N-step returns

```python
buf = GPUReplayBuffer(
    capacity  = 500_000,
    device    = "cuda",
    n_step    = 3,
    gamma     = 0.99,
    num_envs  = 8,
)
```

N-step accumulation is handled internally in CPU rolling buffers. Only completed
n-step transitions are written to the GPU storage.

---

## Thread safety

`GPUReplayBuffer` is thread-safe. All write operations hold a `threading.Lock`.
`AsyncOffPolicyRunner` relies on this when the collector (main thread) and trainer
(daemon thread) share the same buffer instance.

---

## CPU fallback

Passing `device="cpu"` works without any CUDA dependency. All tensors are stored on
host RAM. This mode is useful for unit testing and non-GPU machines:

```python
buf = GPUReplayBuffer(capacity=10_000, device="cpu")
```

---

## Comparison with `ReplayBuffer`

| Feature | `ReplayBuffer` | `GPUReplayBuffer` |
|---|---|---|
| Storage | NumPy (host) | Pre-allocated CUDA tensors |
| `add()` copy | Always (numpy) | Zero-copy if already on device |
| `sample()` copy | `.to(device)` call | None (already on device) |
| Dict obs | Yes | Yes |
| N-step | Yes | Yes |
| PER (prioritised) | Planned | Not yet |
| Checkpointing | NumPy `.npy` | CPU serialisation via `state_dict` |

---

## See also

- [async_runner.md](async_runner.md) — pair with the async runner for maximum Isaac Lab throughput
- [config_reference.md](config_reference.md#asyncrunnerconfig-v020) — `use_gpu_buffer` flag
