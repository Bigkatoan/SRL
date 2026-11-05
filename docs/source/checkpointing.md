# Checkpointing

SRL checkpointing is intentionally simple: training saves model state, optional optimizer state, the current step, and any metrics attached to that save.

The source of truth is [srl/utils/checkpoint.py](https://github.com/Bigkatoan/SRL/blob/main/srl/utils/checkpoint.py).

## What gets saved

For an agent (every `BaseAgent`), `CheckpointManager.save(agent, step=..., metrics=..., tag=...)`
writes an `srl-agent` format v1 **directory** `{tag}_{step:010d}/` -- the same
format as `agent.save(dir)`, see [Saving, loading & deployment](deployment.md):
weights in `model.safetensors`, the resolved `config.yaml`, normalizer
statistics, and `train_state.pt` with the optimizers, `step`, `metrics` and the
`python`/`numpy`/`torch` (+ CUDA) RNG state. The replay buffer is left out
unless the manager was built with `include_replay_buffer=True`
(`srl-train`: `train.checkpoint_replay_buffer: true`).

For a bare `torch.nn.Module`:

- without an optimizer: `{tag}_{step:010d}.safetensors` (weights) + `.meta.pt`
  (`step`, `metrics`, `rng_state`)
- with an optimizer: one `{tag}_{step:010d}.pt` (`model_state`,
  `optimizer_state`, `step`, `metrics`, `rng_state`)

Every file SRL writes loads with `torch.load(weights_only=True)`. Checkpoints
from older SRL versions (pickled `*.pt`) still load, with
`torch.load(weights_only=False)` after a `LegacyCheckpointWarning`.

## Default naming

```text
checkpoints/<run>/
  ckpt_0000005000/        periodic (last 5 kept)
  final_0000100000/
  best_0000080000/        with --save-best
```

## Retention policy

`CheckpointManager` keeps only the most recent `max_keep` saves that it recorded in the current process.

That policy is local and simple by design:

- old checkpoints (directories or files) are deleted once the in-memory save list exceeds `max_keep`
- companion `*.meta.pt` files are also deleted when present
- `latest()` returns the newest known checkpoint, or scans the checkpoint directory if the manager instance has no local save history yet

## Training CLI behavior

`srl-train` exposes checkpoint-related control through:

- `--ckptdir`: directory where checkpoints are written (default `checkpoints/`)
- `--logdir`: directory where TensorBoard/metrics artifacts are written (default `runs/`)
- `--resume`: resume from an existing checkpoint (directory or legacy file)

Checkpoints and run artifacts live in two separate top-level directories,
not nested inside one another — `--ckptdir`/`<run_name>/` is a *sibling* of
`--logdir`/`<run_name>/`:

```text
checkpoints/
  ppo_pendulum_ppo/
    final_0000100000/

runs/
  ppo_pendulum_ppo/
    metrics.jsonl
    history.csv
    summary.json
    training_curves.svg
    training_curves.png
```

(Both `.svg` and `.png` are written unless `--no-plots` is passed.)

The exact run directory name (`ppo_pendulum_ppo` above) is derived by the
training CLI from the algorithm and config stem, and is the same under both
`--ckptdir` and `--logdir`.

## Resume semantics

Resume is step-based, not experiment-discovery-based. You point `--resume` at the exact checkpoint you want to restore.

Example:

```bash
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --algo ppo \
          --steps 200000 \
          --resume checkpoints/ppo_pendulum_ppo/final_0000100000
```

What resume restores depends on the saved payload:

- model weights are always expected
- an off-policy replay buffer only when the checkpoint has one
  (`train.checkpoint_replay_buffer: true`); otherwise it refills from the
  resumed policy
- optimizer state is restored only if it was saved and the receiving code passes an optimizer into `load(...)`
- `python`/`numpy`/`torch` RNG state is restored automatically when present, so a
  resumed run continues with the same exploration-noise/minibatch-shuffle sequence an
  uninterrupted run would have had — not just the same model/optimizer/step.
  Checkpoints saved before this was tracked simply have nothing to restore here.
- step and metrics are returned in the payload for the caller to interpret

## Compatibility expectations

Agent checkpoints carry `format: srl-agent` and a `format_version` in
`meta.json`; this SRL reads versions 0 and 1 and rejects newer ones with a
clear error. The stored `config.yaml` is re-validated on load, and the weights
are loaded with a strict key check against the model it describes, so a
checkpoint whose config no longer validates, or whose weights do not fit the
model, fails loudly instead of loading partially.
