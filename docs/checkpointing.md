# Checkpointing

SRL checkpointing is intentionally simple: training saves model state, optional optimizer state, the current step, and any metrics attached to that save.

The source of truth is [srl/utils/checkpoint.py](/home/ubuntu/antd/SRL/srl/utils/checkpoint.py).

## What gets saved

`CheckpointManager.save(...)` writes a payload with these fields:

- `model_state`: `state_dict()` for plain `torch.nn.Module` models, or the result of `checkpoint_payload()` for richer agent objects
- `step`: integer training step attached to the save
- `metrics`: free-form metrics dict attached by the caller
- `optimizer_state`: included only when an optimizer is passed

If `safetensors` is available and the payload is just model weights plus metadata, SRL writes:

- `*.safetensors` for tensor weights
- `*.meta.pt` for `step` and `metrics`

Otherwise it falls back to a single `*.pt` file via `torch.save`.

## Default file naming

Checkpoint files are named:

```text
{tag}_{step:010d}.pt
```

Examples:

- `ckpt_0000005000.pt`
- `final_0000100000.pt`

When `safetensors` is used, the weight file keeps the same stem:

- `ckpt_0000005000.safetensors`
- `ckpt_0000005000.meta.pt`

## Retention policy

`CheckpointManager` keeps only the most recent `max_keep` saves that it recorded in the current process.

That policy is local and simple by design:

- old checkpoint files are deleted once the in-memory save list exceeds `max_keep`
- companion `*.meta.pt` files are also deleted when present
- `latest()` returns the newest known checkpoint, or scans the checkpoint directory if the manager instance has no local save history yet

## Training CLI behavior

`srl-train` exposes checkpoint-related control through:

- `--ckptdir`: directory where checkpoints are written
- `--resume`: resume from an existing checkpoint file

Typical training layout looks like this:

```text
runs/
  ppo_pendulum_ppo/
    checkpoints/
      ckpt_0000005000.pt
      final_0000100000.pt
    metrics.jsonl
    history.csv
    summary.json
    training_curves.png
```

The exact run directory name is derived by the training CLI from the algorithm and config stem.

## Resume semantics

Resume is step-based, not experiment-discovery-based. You point `--resume` at the exact checkpoint file you want to restore.

Example:

```bash
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --algo ppo \
          --steps 200000 \
          --resume runs/ppo_pendulum_ppo/checkpoints/final_0000100000.pt
```

What resume restores depends on the saved payload:

- model weights are always expected
- optimizer state is restored only if it was saved and the receiving code passes an optimizer into `load(...)`
- step and metrics are returned in the payload for the caller to interpret

## Compatibility expectations

SRL does not yet define a long-term checkpoint compatibility policy across major schema or architecture changes.

Treat checkpoints as compatible only when all of these stay aligned:

- the YAML model graph
- encoder/head shapes
- algorithm-specific checkpoint payload format
- the SRL version you trained with

If you change model topology, routing, or algorithm internals, expect old checkpoints to become load-incompatible.

## Minimal Python usage

```python
import torch

from srl.registry.builder import ModelBuilder
from srl.utils.checkpoint import CheckpointManager

model = ModelBuilder.from_yaml("configs/envs/pendulum_ppo.yaml")
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

manager = CheckpointManager("runs/demo/checkpoints", max_keep=3)
path = manager.save(model, optimizer=optimizer, step=5000, metrics={"score": 123.4})

payload = manager.load(model, path, optimizer=optimizer, device="cpu")
print(payload["step"], payload["metrics"])
```

## Recommended practice

- Keep checkpoint directories per run, not shared across unrelated experiments.
- Use `final_*` saves for explicit promotion points and `ckpt_*` for rolling periodic saves.
- Treat `summary.json` and checkpoint files together as one artifact set when comparing experiments.
- If you need long-lived checkpoints, pin the exact YAML config alongside the saved weights.