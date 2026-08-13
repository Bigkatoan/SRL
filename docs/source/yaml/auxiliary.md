# Auxiliary Representation Learning

Auxiliary losses help a visual encoder learn better representations by training it
against an objective that runs alongside the main RL loss.

There are two halves to this, and they are configured in different places:

| Half | Where | What it does |
|---|---|---|
| The auxiliary **module** | `encoders[].aux_type` in YAML | Adds a decoder or projection head to the model graph |
| The auxiliary **objective** | `aux_loss_type` on `VisualSACConfig` | Selects which loss SAC computes with that module |

```{warning}
Only the first half is declarative today. `srl-train` always builds a plain `SACConfig`
from the `train:` block, and `aux_loss_type`, `aux_weight`, `encoder_lr`, and
`encoder_optimize_with_critic` exist only on `VisualSACConfig`. Putting them in a
`train:` block has no effect — build the config in Python instead.
```

## The encoder side (YAML)

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    aux_type: contrastive     # autoencoder | contrastive | byol
    aux_latent_dim: 128
    use_momentum: true
    momentum_tau: 0.99
    layers:
      - {out_channels: 32, kernel: 8, stride: 4, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 4, stride: 2, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 3, stride: 1, padding: 0, activation: relu}
```

`aux_type` is what actually adds a module to the graph:

| `aux_type` | Module added | Notes |
|---|---|---|
| `autoencoder` | `ConvDecoderHead` | CNN encoders only — needs `input_shape` to know the reconstruction target |
| `contrastive` | `ProjectionHead` | Width comes from `aux_latent_dim` |
| `byol` | `ProjectionHead` | Same head; pair it with `use_momentum: true` |

Any other value is ignored and no auxiliary module is built.

`use_momentum: true` wraps the encoder in a `MomentumEncoder` with EMA rate
`momentum_tau`. CURL and BYOL both need it — they read the target view through the
momentum copy, and return no loss at all if it is missing.

## The algorithm side (Python)

```python
from srl.core.config import VisualSACConfig

cfg = VisualSACConfig(
    action_dim                   = 6,
    aux_loss_type                = "curl",   # none|ae|vae|curl|byol|drq|spr|barlow
    aux_weight                   = 0.1,
    augmentation_mode            = "curl",   # drq | curl | aggressive
    momentum_tau                 = 0.99,
    encoder_lr                   = 1e-4,
    encoder_update_freq          = 2,
    encoder_optimize_with_critic = True,
)
```

The auxiliary loss is only computed when the config is a `VisualSACConfig`,
`aux_loss_type != "none"`, `aux_weight > 0`, and the observation dict contains at least
one 4-D (pixel) tensor.

### Auxiliary loss types

| `aux_loss_type` | Method | Requires | Good for |
|---|---|---|---|
| `none` | No auxiliary loss | — | State-based SAC/TD3 |
| `ae` | Autoencoder (MSE reconstruction) | `ConvDecoderHead` (`aux_type: autoencoder`) | Vision baseline |
| `vae` | Variational AE (MSE + KL) | `VAEHead` + `ConvDecoderHead` | Generative representations |
| `curl` | CURL InfoNCE contrastive | `ProjectionHead` + momentum encoder | Default for vision SAC |
| `byol` | BYOL self-prediction | `ProjectionHead` + momentum encoder | Stable, no negatives needed |
| `drq` | DrQ augmented Q-consistency | Nothing extra — augments the critic input | Data-augmented RL |
| `spr` | SPR latent forward prediction | `LatentTransitionModel` | Model-based auxiliary |
| `barlow` | Barlow Twins redundancy reduction | `ProjectionHead` | Decorrelated features |

`VAEHead` and `LatentTransitionModel` exist in
[srl/networks/heads/aux_head.py](https://github.com/Bigkatoan/SRL/blob/main/srl/networks/heads/aux_head.py)
but `ModelBuilder` does not instantiate them from YAML, so `vae` and `spr` need those
modules attached in Python. If the required module is missing, `_compute_aux_loss`
returns `None` and training silently continues with the plain RL loss.

### Augmentation modes

`augmentation_mode` controls the view pipeline used by `curl`, `byol`, and `barlow`
(`drq` always uses random crop):

| Mode | Pipeline |
|---|---|
| `drq` | Random crop |
| `curl` | Random crop + colour jitter |
| `aggressive` | Random crop + colour jitter + translate + cutout |

## Per-type notes

### `ae` — Autoencoder

Reconstructs the observation from the latent. Simple, and a good baseline.

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    aux_type: autoencoder
    layers:
      - {out_channels: 32, kernel: 8, stride: 4, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 4, stride: 2, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 3, stride: 1, padding: 0, activation: relu}
```

```python
cfg = VisualSACConfig(action_dim=6, aux_loss_type="ae")
```

### `curl` — Contrastive Unsupervised Representations for RL

Learns representations by contrasting two augmented views of the same observation.

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    aux_type: contrastive
    aux_latent_dim: 64
    use_momentum: true
    momentum_tau: 0.99
    layers:
      - {out_channels: 32, kernel: 8, stride: 4, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 4, stride: 2, padding: 0, activation: relu}
      - {out_channels: 64, kernel: 3, stride: 1, padding: 0, activation: relu}
```

```python
cfg = VisualSACConfig(action_dim=6, aux_loss_type="curl", batch_size=128)
```

### `byol` — Bootstrap Your Own Latent

No negative samples required, and more stable than contrastive on many tasks.

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    aux_type: byol
    aux_latent_dim: 128
    use_momentum: true
    momentum_tau: 0.995
```

```python
cfg = VisualSACConfig(action_dim=6, aux_loss_type="byol")
```

### `drq` — Data-regularized Q

Applies random augmentation and enforces Q-value consistency across views. It needs no
auxiliary module at all.

```python
cfg = VisualSACConfig(action_dim=6, aux_loss_type="drq", encoder_update_freq=2)
```

## Best practices

1. **Batch size** — contrastive objectives want at least 128; 256 works best.
2. **`momentum_tau`** — 0.99 for contrastive, 0.995 for BYOL.
3. **`aux_latent_dim`** — usually smaller than `latent_dim` (e.g. 64 against 256).
4. **Learning rate** — set `encoder_lr` below `lr_critic` when an auxiliary loss is
   active (1e-4 against 3e-4). Without `encoder_lr`, the encoder optimizer inherits
   `lr_critic`.
5. **Start with `ae`** if you are unsure — it is simple and dependable.

## See also

- [Encoders](encoders.md)
- [Algorithms](../algorithms.md)
- [Configuration Reference](../config_reference.md)
