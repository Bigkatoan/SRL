# Replay Buffers

SRL có 2 loại replay buffer: CPU-based (default) và GPU-based (cho GPU simulators).

## CPU ReplayBuffer (default)

Standard circular buffer lưu transitions dưới dạng numpy arrays.

```python
from srl.core.replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=1_000_000)
buf.add(obs, action, reward, done, next_obs)
batch = buf.sample(256)
```

Dùng cho hầu hết mọi trường hợp.

## GPU Replay Buffer (v0.2.0)

`GPUReplayBuffer` là pre-allocated CUDA circular buffer, thiết kế cho zero-copy với Isaac Lab.

### Vấn đề với CPU buffer

```
CPU ReplayBuffer:
  Isaac Lab (CUDA) → numpy copy (host) → store → batch.to(device) → GPU
  ↑ 2 host↔device round-trips mỗi step
```

### Giải pháp

```
GPUReplayBuffer:
  Isaac Lab (CUDA) → GPUReplayBuffer (CUDA) → batch already on GPU
  ↑ 0 host↔device copies
```

### Sử dụng

```python
from srl.core.gpu_replay_buffer import GPUReplayBuffer

buf = GPUReplayBuffer(
    capacity=1_000_000,
    device="cuda:0",
)

buf.add(
    obs={"pixels": obs_tensor, "state": state_tensor},
    action=action_tensor,
    reward=reward_float,
    done=done_bool,
    next_obs={"pixels": next_pixels, "state": next_state},
)

batch = buf.sample(256)
# batch.obs["pixels"]: (256, C, H, W) on cuda:0
# batch.actions:       (256, action_dim) on cuda:0
```

### Constructor arguments

| Argument | Type | Default | Mô tả |
|---|---|---|---|
| `capacity` | int | required | Max transitions |
| `device` | str | `"cuda"` | Storage device |
| `storage_dtype` | torch.dtype | float32 | Precision |
| `n_step` | int | 1 | N-step return lookahead |
| `gamma` | float | 0.99 | Discount cho n-step |
| `num_envs` | int | 1 | Số parallel envs |

### Checkpointing

GPUReplayBuffer serialize sang CPU tensors khi lưu:

```python
state = buf.state_dict()    # CPU tensors, portable
buf.load_state_dict(state)  # restore on any device
```

## Kết hợp với Async Runner

```yaml
train:
  use_async: true
  use_gpu_buffer: true
```

Với cấu hình này:

1. Collector (main thread) viết CUDA tensors trực tiếp vào GPUReplayBuffer
2. Trainer (daemon thread) sample batch đã ở trên GPU
3. Không có CPU↔GPU copies trong hot path

## Xem thêm

- [GPU Replay Buffer API](../gpu_replay_buffer.md)
- [Runners & Training Loop](runners.md)
- [Isaac Lab Integration](../integrations/isaaclab.md)
