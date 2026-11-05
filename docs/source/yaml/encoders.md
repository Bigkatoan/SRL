# Encoders

An encoder turns an observation (image, vector, text) into the latent vector consumed
by the actor and critic heads.

## Which encoder should I use?

| Type | Use for | Example environments |
|---|---|---|
| `mlp` | Vector state (joint angles, velocities) | HalfCheetah, Pendulum, Isaac Lab |
| `cnn` | Pixel images (RGB, depth) | CarRacing, Isaac Lab visual tasks |
| `lstm` | Time series, stacked frames | POMDPs, partially observable envs |
| `text` | Language/instruction embeddings | Language-conditioned tasks |

## Configuration fields

These map one-to-one onto `EncoderConfig` in
[srl/registry/config_schema.py](https://github.com/Bigkatoan/SRL/blob/main/srl/registry/config_schema.py).

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | ✓ | — | Unique node id, referenced in `flows` |
| `type` | ✓ | — | `mlp`, `cnn`, `lstm`, `text`, or a registry key |
| `input_name` | recommended | `null` | Observation dict key this encoder reads |
| `input_dim` | mlp/lstm | `auto` | Input vector dimension; omit or write `auto` to take it from the env |
| `input_shape` | cnn | `auto` | `[C, H, W]`; omit or write `auto` to take it from the env |
| `latent_dim` | | `128` | Output latent width |
| `layers` | | `[]` | Layer definitions |
| `aux_type` | | `null` | `autoencoder`, `contrastive`, or `byol` |
| `aux_latent_dim` | | `64` | Projection-head width for `contrastive`/`byol` |
| `use_momentum` | | `false` | Wrap the encoder in a momentum/EMA copy |
| `momentum_tau` | | `0.99` | EMA coefficient for the momentum encoder |
| `recurrent` | | `false` | Wrap a non-LSTM encoder in an LSTM |
| `lstm_hidden` | | `256` | Hidden size used when wrapping with LSTM |
| `extra` | | `{}` | Keyword arguments for custom encoder types |

Unknown keys **are an error** (with a "did you mean" suggestion — see
[strict validation](../config_reference.md#strict-validation)). Keys specific to one
encoder type are accepted when that type declares them: `text` accepts `vocab_size`,
`embed_dim`, `channels` and `max_len`; a custom type registered with
`register_encoder("name", extra_keys=(...))` accepts its own. Anything else for a
custom type goes under an explicit `extra:` mapping. Both end up in `cfg.extra`. See
[Extending SRL](../extending.md).

Explicit `input_dim`/`input_shape` values are cross-checked against the environment
when the model is built with env spaces, so a stale number fails with a clear message
instead of a shape error in the first forward pass.

```{warning}
`aux_type` accepts only `autoencoder`, `contrastive`, and `byol` (anything else is a
validation error). The builder attaches a `ConvDecoderHead` for `autoencoder` (CNN
encoders only, since it needs `input_shape` — `autoencoder` on a flat encoder is
rejected) and a `ProjectionHead` for `contrastive`/`byol`.
```

There is no `frame_stack` key (it was never applied and has been removed); stack
frames in an environment wrapper and size `input_shape`/`input_dim` accordingly.

## MLP example (state-based)

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}
      - {out_features: 256, activation: relu}
```

## CNN example (vision)

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    layers:
      - {out_channels: 32, kernel: 8, stride: 4, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 4, stride: 2, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 3, stride: 1, padding: 0, activation: relu}
```

## CNN with auxiliary reconstruction

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 96, 96]
    latent_dim: 256
    aux_type: autoencoder
    aux_latent_dim: 128
    layers:
      - {out_channels: 32, kernel: 8, stride: 4, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 4, stride: 2, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 3, stride: 1, padding: 0, activation: relu}
```

`aux_type: autoencoder` attaches a `ConvDecoderHead` to the model graph. Whether the
reconstruction term actually enters the training loss depends on the algorithm — see
[Auxiliary Representation Learning](auxiliary.md).

## Multi-modal example (image + state)

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_name: front_camera
    input_shape: [3, 84, 84]
    latent_dim: 128

  - name: state_enc
    type: mlp
    input_name: joint_states
    input_dim: 18
    latent_dim: 64
```

```{tip}
Always set `input_name` when you have more than one encoder, so routing is explicit
instead of relying on the fallback heuristics.
```

## Encoder optimizer (v0.2.0)

SAC, DDPG, and TD3 give the encoder its own optimizer so it is not updated twice per
gradient step:

| Optimizer | Parameters | When it steps |
|---|---|---|
| `critic_optimizer` | Critic head only | Every gradient step |
| `actor_optimizer` | Actor head only | Every gradient step |
| `encoder_optimizer` | All encoder parameters | Every `encoder_update_freq` critic steps |

`encoder_update_freq` is a field on `SACConfig`, `DDPGConfig`, and `TD3Config`, so it
can be set from the `train:` block:

```yaml
train:
  encoder_update_freq: 2   # step the encoder every 2 critic updates
```

The encoder optimizer's learning rate comes from `encoder_lr`, which only exists on
`VisualSACConfig`/`VisualPPOConfig` (not on `DDPGConfig`/`TD3Config`, which have no
Visual variant). `srl-train` builds one of the Visual configs automatically when an
encoder declares a recognised `aux_type`, or when the `train:` block itself sets a
Visual-only field such as `encoder_lr` — see [Auxiliary Representation
Learning](auxiliary.md). When `encoder_lr` is absent, the encoder optimizer falls back
to `lr_critic`.

## See also

- [Auxiliary Representation Learning](auxiliary.md)
- [Heads & Flows](heads_flows.md)
- [Training Block Reference](training_block.md)
- [Encoders guide](../encoders.md)
