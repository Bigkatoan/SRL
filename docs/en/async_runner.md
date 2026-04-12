# Async Off-Policy Runner

`AsyncOffPolicyRunner` decouples data collection from gradient updates for SAC, DDPG,
and TD3. This is especially useful with Isaac Lab, where the simulator and the GPU
replay buffer live in the same CUDA context and you want the trainer to consume
transitions as fast as they arrive without blocking the simulator.

---

## Architecture

```
Main thread (collector)          Daemon thread (trainer)
────────────────────────         ─────────────────────────
env.step()                       →  trainer_thread.step()
buf.add(transition)              ←  signals via threading.Condition
```

- **Collector** — always runs on the **main thread**. Isaac Lab's USD/PhysX simulation
  context must not be touched from a background thread.
- **Trainer** — runs on a **daemon thread** with its own `torch.cuda.Stream` so that
  memory copies and gradient updates do not block the collector.
- **Synchronisation** — a `threading.Condition` lets the trainer wait for enough
  transitions before the first gradient step, and signals back when the model is updated.

---

## Quick start

```python
from srl.core.config import SACConfig, AsyncRunnerConfig
from srl.algorithms.sac import SAC
from srl.core.replay_buffer import ReplayBuffer
from srl.runners import AsyncOffPolicyRunner

algo = SAC(SACConfig(action_dim=6))
buf  = ReplayBuffer(capacity=1_000_000)

runner_cfg = AsyncRunnerConfig(
    use_async      = True,
    use_gpu_buffer = True,   # swap ReplayBuffer → GPUReplayBuffer automatically
    prefill_steps  = 1000,
)

runner = AsyncOffPolicyRunner(
    algo       = algo,
    env        = env,          # any gym-compatible env
    buffer     = buf,
    runner_cfg = runner_cfg,
    total_steps= 500_000,
    batch_size = 256,
)
runner.run()
```

---

## Sync fallback

Setting `use_async=False` (the default) makes the runner use the standard synchronous
training loop. This is useful for debugging or for environments where the async
threading overhead outweighs its benefits (e.g., very fast CPU simulators).

```python
runner_cfg = AsyncRunnerConfig(use_async=False, use_gpu_buffer=True)
# → GPU buffer used; collection and training still on one thread
```

---

## `AsyncRunnerConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `use_async` | `bool` | `False` | Enable collector/trainer thread split |
| `use_gpu_buffer` | `bool` | `False` | Replace CPU buffer with `GPUReplayBuffer` |
| `prefill_steps` | `int` | `1000` | Random-action steps before first gradient update |
| `queue_maxsize` | `int` | `4` | Max transitions queued between threads |

---

## Isaac Lab integration

Isaac Lab environments expose a CUDA tensor API directly. Pair
`use_gpu_buffer=True` with an Isaac Lab env to avoid any host↔device copy in the
collect → store → sample path:

```
Isaac Lab env (CUDA tensors)
  ↓  no copy
GPUReplayBuffer (pre-allocated CUDA tensors)
  ↓  no copy
SAC critic/actor forward (same CUDA device)
```

See [gpu_replay_buffer.md](gpu_replay_buffer.md) for the buffer API.

---

## Checkpointing

`AsyncOffPolicyRunner` calls `algo.save_checkpoint()` at the same intervals as the
synchronous runner. The `GPUReplayBuffer` serialises to CPU tensors automatically when
`state_dict()` is called, so checkpoint files remain portable.

---

## See also

- [gpu_replay_buffer.md](gpu_replay_buffer.md) — GPU circular buffer
- [algorithms.md](algorithms.md) — encoder optimizer and `encoder_update_freq`
- [config_reference.md](config_reference.md#asyncrunnerconfig-v020) — full field reference
