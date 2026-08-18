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
import copy

from srl.algorithms.sac import SAC
from srl.core.config import AsyncRunnerConfig, SACConfig
from srl.registry.builder import ModelBuilder
from srl.runners import AsyncOffPolicyRunner

model = ModelBuilder.from_yaml("configs/envs/halfcheetah_sac.yaml")
agent = SAC(model, copy.deepcopy(model), SACConfig(action_dim=6), device="cuda:0")

runner_cfg = AsyncRunnerConfig(
    use_async      = True,
    use_gpu_buffer = True,   # swap agent.buffer → GPUReplayBuffer automatically
)

runner = AsyncOffPolicyRunner(
    agent          = agent,      # the agent owns its own replay buffer
    env            = env,        # any gym-compatible env
    total_steps    = 500_000,
    runner_cfg     = runner_cfg,
    device         = "cuda:0",
    random_steps   = 1000,       # random-action warmup before updates start
    update_after   = 1000,
    update_every   = 1,
    gradient_steps = 1,
)
runner.run()
```

The runner takes the agent, not a separate buffer: it reads and (for
`use_gpu_buffer=True`) replaces `agent.buffer`, and the batch size comes from the
agent's own config.

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
| `prefill_steps` | `int` | `0` | Declared but currently unread — use the runner's `random_steps` argument |
| `queue_maxsize` | `int` | `2` | Declared but currently unread |

The runner's own constructor arguments control the loop: `random_steps` (random-action
warmup), `update_after` (minimum buffer fill), `update_every` (env steps per trigger),
and `gradient_steps` (updates per trigger).

## `srl-train` / YAML usage

Set `use_async`/`use_gpu_buffer` directly under a SAC/DDPG/TD3 config's `train:` block --
`srl-train` builds an `AsyncRunnerConfig` from those two keys and switches to
`AsyncOffPolicyRunner` automatically, no Python-API code needed:

```yaml
algo: sac
train:
  n_envs: 512
  update_every: 512
  gradient_steps: 16
  use_async: true
  use_gpu_buffer: true
```

`random_steps`/`update_after`/`update_every`/`gradient_steps` are read from the
algorithm config the same way the sync path already reads them (`start_steps`,
`update_after`, `update_every`, `gradient_steps` in the YAML `train:` block) --
nothing async-specific to configure beyond the two boolean flags themselves.

```{note}
Real-world reference: [JAVIS](https://github.com/Bigkatoan/JAVIS)'s
`configs/srl/javis_mjlab_sac.yaml` runs this path against a GPU-batched mjlab task on
an RTX 3090 -- both flags together measured a ~9.5x wall-clock speedup over the sync
path once `batch_size`/`gradient_steps` were also restructured toward fewer, larger
updates (see that file's own comments for the exact numbers and the reasoning).
```

---

## Isaac Lab integration

A GPU-batched simulator can hand CUDA tensors straight to the buffer. Pair
`use_gpu_buffer=True` with such an env to keep the collect → store → sample path free
of host↔device copies:

```
GPU sim (CUDA tensors)
  ↓  no copy
GPUReplayBuffer (pre-allocated CUDA tensors)
  ↓  no copy
SAC critic/actor forward (same CUDA device)
```

```{warning}
`IsaacLabWrapper` currently converts observations, rewards, and done flags to CPU numpy
before returning them, so going through that wrapper reintroduces a host round-trip.
Feed the runner CUDA tensors directly if the copy-free path matters.
```

See [gpu_replay_buffer.md](gpu_replay_buffer.md) for the buffer API.

---

## Checkpointing

The runner itself does not save checkpoints — drive them from the `log_fn` callback, or
call `agent.checkpoint_payload()` / `CheckpointManager.save(...)` from your own loop.
The `GPUReplayBuffer` serialises to CPU tensors automatically when `state_dict()` is
called, so checkpoint files remain portable.

---

## See also

- [gpu_replay_buffer.md](gpu_replay_buffer.md) — GPU circular buffer
- [algorithms.md](algorithms.md) — encoder optimizer and `encoder_update_freq`
- {ref}`config_reference.md <asyncrunnerconfig>` — full field reference
