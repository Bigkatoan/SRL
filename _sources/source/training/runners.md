# Runners & Training Loop

Runner kiểm soát vòng lặp training: thu thập transitions từ môi trường, lưu vào buffer, và trigger gradient updates.

## Synchronous Runner (default)

Tất cả diễn ra trên cùng một thread theo thứ tự:

```
collect → store → sample → update → repeat
```

Đây là default, đủ dùng cho hầu hết trường hợp.

```bash
# Sync runner — không cần config thêm
srl-train --config configs/envs/halfcheetah_sac.yaml
```

## Async Off-Policy Runner (v0.2.0)

`AsyncOffPolicyRunner` tách collection và training vào 2 threads:

```
Main thread (collector)            Daemon thread (trainer)
────────────────────────           ─────────────────────────
env.step()                    →    trainer_thread.step()
buf.add(transition)           ←    signals via threading.Condition
```

**Lợi ích:**

- Collector không bị block bởi gradient computation
- Đặc biệt hữu ích với Isaac Lab (CUDA context phải ở main thread)
- Trainer có CUDA stream riêng, không tranh tài nguyên với simulator

**Kích hoạt:**

```yaml
train:
  use_async: true
  use_gpu_buffer: true   # khuyến nghị kết hợp GPUReplayBuffer
  prefill_steps: 1000
```

```python
from srl.core.config import AsyncRunnerConfig
from srl.runners import AsyncOffPolicyRunner

runner_cfg = AsyncRunnerConfig(
    use_async=True,
    use_gpu_buffer=True,
    prefill_steps=1000,
    queue_maxsize=4,
)
```

## `AsyncRunnerConfig` fields

| Field | Type | Default | Mô tả |
|---|---|---|---|
| `use_async` | bool | False | Bật collector/trainer thread split |
| `use_gpu_buffer` | bool | False | Dùng GPUReplayBuffer |
| `prefill_steps` | int | 1000 | Random-action steps trước gradient update đầu |
| `queue_maxsize` | int | 4 | Max transitions queued giữa threads |

## On-Policy Runner (PPO/A2C)

```yaml
train:
  n_steps: 2048
  n_envs: 8
  n_epochs: 10
  batch_size: 256
```

## Checkpointing

Runner tự động lưu checkpoint theo `eval_freq`:

```
runs/{algo}_{config_stem}/
  checkpoints/
    ckpt_{step:010d}.pt
    final_{step:010d}.pt
  metrics.jsonl
  summary.json
  training_curves.png
```

Resume:

```bash
srl-train --config my.yaml --resume runs/sac_half/checkpoints/final_0001000000.pt
```

## Xem thêm

- [Replay Buffers](buffers.md)
- [Async Runner API Reference](../async_runner.md)
- [Isaac Lab Integration](../integrations/isaaclab.md)
