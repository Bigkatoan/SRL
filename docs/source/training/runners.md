# Runners & Training Loop

The runner drives the training loop: it collects transitions from the environment,
stores them in a buffer, and triggers gradient updates.

## Synchronous runner (default)

Everything happens on one thread, in order:

```
collect → store → sample → update → repeat
```

This is what `srl-train` uses, and it is enough for almost every workload.

```bash
# Sync runner — nothing extra to configure
srl-train --config configs/envs/halfcheetah_sac.yaml
```

## On-policy runner (PPO, A2C)

On-policy algorithms collect a fixed-size rollout, compute returns and advantages, then
run several optimisation epochs over that rollout before discarding it. The relevant
`train:` fields are:

```yaml
train:
  n_steps: 2048      # steps collected per env per rollout
  n_envs: 8
  n_epochs: 10
  batch_size: 256
```

A3C is different again: it spawns `n_workers` CPU worker processes that collect and
compute gradients asynchronously, and it does not go through the vectorised runner at
all.

## Async off-policy runner (v0.2.0)

`AsyncOffPolicyRunner` splits collection and training across two threads:

```
Main thread (collector)            Daemon thread (trainer)
────────────────────────           ─────────────────────────
env.step()                    →    trainer thread's agent.update()
buf.add(transition)           ←    signals via threading.Condition
```

**Why:**

- The collector is not blocked while gradients are computed.
- It suits Isaac Lab in particular, where the simulation context has to stay on the
  main thread.
- The trainer gets its own CUDA stream, so its compute overlaps with environment
  stepping instead of contending with it.

**Enabling it:**

```python
from srl.core.config import AsyncRunnerConfig
from srl.runners import AsyncOffPolicyRunner

runner = AsyncOffPolicyRunner(
    agent=sac_agent,          # SAC / DDPG / TD3
    env=env,
    total_steps=1_000_000,
    runner_cfg=AsyncRunnerConfig(use_async=True, use_gpu_buffer=True),
    device="cuda:0",
    random_steps=1000,        # random-action warmup before updates start
    update_after=1000,
    update_every=1,
    gradient_steps=1,
)
runner.run()
```

With `use_async=False` (the default) the runner falls back to the standard synchronous
loop; with `use_gpu_buffer=True` alone it swaps in the GPU buffer but keeps collection
and training on one thread.

```{warning}
The async runner is a Python-API feature. `use_async` / `use_gpu_buffer` in a YAML
`train:` block are currently ignored: `srl-train` maps the `train:` block onto the
algorithm config dataclasses only, and never constructs an `AsyncRunnerConfig` from it.
```

See [Async Off-Policy Runner](../async_runner.md) for the full field reference and the
Isaac Lab notes.

## Checkpointing

`srl-train` attaches a `CheckpointCallback` that saves every 100,000 steps, plus a
`final_*` save when training ends. Checkpoints and run artifacts go to two separate
top-level directories — `--ckptdir` (default `checkpoints/`) and `--logdir` (default
`runs/`) — sharing the same `{algo}_{config_stem}` run name:

```
checkpoints/{algo}_{config_stem}/
  ckpt_{step:010d}.pt
  final_{step:010d}.pt

runs/{algo}_{config_stem}/
  metrics.jsonl
  history.csv
  summary.json
  training_curves.png
```

Resume by pointing at the exact checkpoint file:

```bash
srl-train --config my.yaml --resume checkpoints/sac_my/final_0001000000.pt
```

## See also

- [Replay Buffers](buffers.md)
- [Async Off-Policy Runner](../async_runner.md)
- [Checkpointing](../checkpointing.md)
- [Isaac Lab Integration](../integrations/isaaclab.md)
