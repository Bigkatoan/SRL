# Encoders (Setup & Best Practices)

This page describes the role of `encoder` blocks in the SRL model pipeline, the
built-in encoder types, the config fields that matter, YAML examples, tuning
advice, and a quick smoke test.

## 1. Overview

An encoder is the module that turns raw observations (images, vectors, text, ...)
into latent vectors consumed by the actor/critic heads. It sits at the boundary
between the environment and the rest of the model graph — choosing and
configuring it correctly is usually the single biggest factor in performance.

## 2. Which encoder type to use

- `mlp` — vector / low-dimensional state input. Lightweight, fast.
- `cnn` — image (vision) input. Avoid very deep stacks for realtime or Isaac Lab use.
- `lstm` — sequences, latency, or stacked frames with a time dependency.
- `text` — token or embedding input for language.

## 3. Key config fields (schema)

- `name` — the encoder's id, referenced from `flows`
- `type` — `mlp`, `cnn`, `lstm`, `text`, or a registered custom key
- `input_name` — the observation dict key this encoder reads (always declare this explicitly)
- `input_dim` / `input_shape` — input size
- `latent_dim` — output latent size
- `layers` — layer structure (builder-driven)
- `aux_type`, `aux_latent_dim` — set these to attach an auxiliary head (`autoencoder`, `contrastive`, `byol`)
- `use_momentum`, `momentum_tau` — enable a momentum (EMA) target encoder for contrastive/BYOL
- `recurrent`, `lstm_hidden`, `frame_stack`

Full schema reference: [srl/registry/config_schema.py](https://github.com/Bigkatoan/SRL/blob/main/srl/registry/config_schema.py)

```{note}
`aux_type` (encoder-level, one of `autoencoder`/`contrastive`/`byol`) and
`aux_loss_type` (train-config level, one of `none`/`ae`/`vae`/`curl`/`byol`/`drq`/`spr`/`barlow`)
are two different fields with overlapping names — see
[Limitations](limitations.md) for how they relate and the current state of
wiring this up through `srl-train`.
```

## 4. Example YAML configs

MLP (state) example:

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 24
    latent_dim: 128
    layers:
      - {out_features: 256, activation: relu}
      - {out_features: 128, activation: relu}
```

CNN (vision) example with an auxiliary reconstruction head:

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 96, 96]
    latent_dim: 256
    aux_type: autoencoder
    aux_latent_dim: 128
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

## 5. Encoder optimizer & update policy

Relevant training-config fields when an encoder is involved:

- A separate `encoder_optimizer` avoids double-updating encoder params when both
  the actor and critic backward passes touch them.
- `encoder_update_freq` — update the encoder every N critic steps; `2` is a
  common choice for vision.
- `encoder_optimize_with_critic` — if true, the critic loss also updates the
  encoder (instead of only the auxiliary loss).

Example (`VisualSACConfig`-style train block):

```yaml
train:
  total_steps: 1_000_000
  encoder_update_freq: 2
  encoder_optimize_with_critic: true
  encoder_lr: 3e-4
  lr_actor: 3e-4
  lr_critic: 3e-4
```

```{note}
As of this writing, `srl-train` does not yet read this block into
`VisualPPOConfig`/`VisualSACConfig` — see [Limitations](limitations.md).
```

## 6. Auxiliary representation learning

Supported auxiliary modes: `autoencoder`, `vae`, `contrastive`, `byol`, `drq`, `spr`, `barlow`.

- Set `use_momentum: true` when configuring BYOL or contrastive-momentum encoders.
- The projection head's `aux_latent_dim` is usually smaller than the main latent dim.
- Augmentations and batch size have a large effect on contrastive quality.

## 7. Wiring into actor/critic (`flows`)

Declare `flows` explicitly so the actor/critic receive the right encoder:

```yaml
flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "image_enc -> critic"
  - "state_enc -> critic"
```

If the actor and critic need different representations, use separate encoders
per branch.

## 8. Quick testing & debugging

- On a shape error, check `input_shape` / `input_dim` and the `input_name` mapping first.
- Inspect the built model graph with `--save-model-pipeline` (writes a pipeline
  diagram instead of training) to confirm encoder output sizes match what
  downstream heads expect.
- Run a smoke test:

```bash
# module fallback, running from source
python -m srl.cli.train --config configs/envs/halfcheetah_sac.yaml --device cpu --seed 0 --no-plots
```

## 9. Best practices

- For vision / Isaac workloads: use a momentum encoder for contrastive/BYOL, and mixed precision when a GPU is available.
- Avoid very deep networks for realtime use; prefer larger batches and stronger augmentation for contrastive learning.
- Use a separate optimizer to tune the encoder's learning rate independently when fine-tuning.
- Validate the pipeline with a small end-to-end smoke test before scaling up.

## 10. More examples

See [Examples: encoder examples](examples/encoder_examples.md) for further
configuration examples and smoke-test steps.
