# Heads & Flows

## Actor heads

An actor head takes the latent vector produced by the encoders and outputs an action
distribution.

| Type | Description | Use with |
|---|---|---|
| `squashed_gaussian` | Gaussian → tanh squash → bounded actions | SAC, vision tasks |
| `gaussian` | Gaussian, no squashing | PPO, A2C, A3C |
| `deterministic` | Emits the mean directly | DDPG, TD3 |

```yaml
actor:
  name: actor
  type: squashed_gaussian
  action_dim: 6
  log_std_min: -5.0
  log_std_max: 2.0
  layers:
    - {out_features: 256, activation: relu, norm: layer_norm}
    - {out_features: 256, activation: relu}
```

## Critic heads

| Type | Description | Use with |
|---|---|---|
| `twin_q` | Two Q-networks, min taken to reduce overestimation | SAC, TD3 |
| `q` | A single Q-network (alias: `q_function`) | DDPG |
| `value` | State value `V(s)` | PPO, A2C, A3C |

```yaml
critic:
  name: critic
  type: twin_q
  action_dim: 6
  layers:
    - {out_features: 256, activation: relu, norm: layer_norm}
    - {out_features: 256, activation: relu}
```

```{note}
The builder derives each head's input size from the flow graph — there is no
`input_dim` to hard-code. `action_dim` is required for actor heads and for `q`/`twin_q`
critics; `srl-train` also checks head types against the selected algorithm before
building anything.
```

## Flows — the routing graph

`flows` declares the data path from encoders to heads as a list of directed edges.

```yaml
flows:
  - "encoder_name -> actor"
  - "encoder_name -> critic"
```

### Important properties

- **Automatic concatenation** — when several encoders feed the same head, their latents
  are concatenated and the head's input dimension is the sum of their `latent_dim`s.
- **Topological ordering** — execution order is resolved automatically; cycles raise a
  `ValueError`.
- **Asymmetric branches** — the actor and critic can consume different encoder sets.
- **Declared names only** — both sides of every edge must name a declared encoder,
  `actor`, or `critic`; anything else is an error at build time.

```{note}
Encoder-to-encoder edges parse, and they are followed when working out which encoders
feed a head. They are not a runtime pipeline, though: at execution time every encoder
reads its own observation key, and latents are only concatenated on the way into a
head.
```

### Example: symmetric

```yaml
flows:
  - "state_enc -> actor"
  - "state_enc -> critic"
```

### Example: asymmetric multi-modal

```yaml
# Actor sees image + state; critic sees state only
flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "state_enc -> critic"
```

### Example: full multi-modal, symmetric

```yaml
flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "image_enc -> critic"
  - "state_enc -> critic"
```

## Layer specification

### Dict style (recommended)

```yaml
layers:
  - {out_features: 256, activation: relu, norm: layer_norm}
  - {out_features: 256, activation: relu, norm: none}
```

MLP layers also accept a bare integer as shorthand for `out_features`:

```yaml
layers: [256, 256]
```

### List style (CNN)

CNN shorthand lists are positional, in this order:

```yaml
# [out_channels, kernel, padding, activation, pooling]
layers:
  - [32, 8, 4, relu]
  - [64, 4, 2, relu]
  - [64, 3, 1, relu]
```

:::{warning}
The third element is **padding**, not stride. `stride` has no shorthand slot and
defaults to `1`, so any layer that needs a stride must use the dict form:

```yaml
layers:
  - {out_channels: 32, kernel: 8, stride: 4, padding: 0, activation: relu}
  - {out_channels: 64, kernel: 4, stride: 2, padding: 0, activation: relu}
```

Note the key is `kernel`, not `kernel_size`, and group-norm groups are set with
`norm_groups`.
:::

Omitted trailing entries fall back to their defaults (`kernel: 3`, `stride: 1`,
`padding: "same"`, `pooling: "none"`).

### Activation options

`relu`, `leaky_relu`, `tanh`, `sigmoid`, `gelu`, `silu`, `elu`, `mish`, `hardswish`,
`none` (alias `identity`)

### Norm options

`batch_norm`, `layer_norm`, `group_norm`, `instance_norm`, `rms_norm`, `none`

## See also

- [Encoders](encoders.md)
- [Configuration Reference](../config_reference.md) — the full `LayerConfig` field list
