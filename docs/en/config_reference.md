# Configuration Reference

SRL uses YAML files to define the model graph and the currently supported declarative parts of training.

This page is the schema-level companion to the [YAML Core Guide](yaml_core.md).

Use this page when you need exact field names and supported options. Use the YAML core guide when you need the bigger picture of how the system fits together.

---

## File structure

```yaml
# Optional — used by CLI to select algorithm
algo: ppo | sac | ddpg | td3 | a2c | a3c

# Optional — used by CLI environment setup
env_id: <str>
env_type: flat | goal | isaaclab | racecar

# List of encoder modules
encoders:
  - name: <str>           # unique key; referenced in flows
    type: mlp | cnn | lstm | text
    input_name: <str>     # optional explicit obs key mapping
    input_dim: <int>      # required for mlp/lstm
    input_shape: [C, H, W]# required for cnn
    latent_dim: <int>     # output dimension
    aux_type: <str>       # optional: autoencoder | contrastive | byol
    aux_latent_dim: <int> # optional projection width
    use_momentum: <bool>
    momentum_tau: <float>
    frame_stack: <int>
    recurrent: <bool>
    lstm_hidden: <int>
    layers:
      - {out_features: <int>, activation: <str>, norm: <str>}

# Data-flow graph (encoder → head)
flows:
  - "<encoder_name> -> actor"
  - "<encoder_name> -> critic"

# Actor head
actor:
  name: actor
  type: gaussian | squashed_gaussian | deterministic
  action_dim: <int>
  log_std_init: <float>   # gaussian / squashed_gaussian
  log_std_min: <float>    # squashed_gaussian
  log_std_max: <float>    # squashed_gaussian

# Critic head
critic:
  name: critic
  type: value | twin_q
  action_dim: <int>       # required for twin_q

# Loss configuration
losses:
  - name: <loss_name>
    weight: <float>
    schedule: constant | linear_decay | cosine

# Optional — consumed by CLI training
train:
  total_steps: <int>
  n_envs: <int>
  ... algorithm-specific fields ...
```

---

## YAML-first rules

- `encoders` declare the feature extractors that enter the graph.
- `flows` declare how encoder outputs feed into downstream nodes.
- `actor` and `critic` are regular graph nodes built from `HeadConfig`.
- `losses` only cover built-in terms currently supported by the runtime.
- `train` is read by the CLI and mapped to algorithm config dataclasses.

## Encoder types

| Type | `input_dim` | Notes |
|---|---|---|
| `mlp` | obs dimension | Fully-connected layers |
| `cnn` | `[C, H, W]` | Convolutional encoder for pixels |
| `lstm` | obs dimension | Recurrent encoder built from an MLP + LSTM |
| `text` | vocabulary size | Embedding + LSTM |

## Encoder fields

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Unique node name used in `flows` |
| `type` | `str` | Encoder family or registered custom encoder key |
| `input_name` | `str \| null` | Explicit observation key for this encoder |
| `input_dim` | `int \| null` | Required for vector encoders |
| `input_shape` | `list[int] \| null` | Required for CNN encoders |
| `latent_dim` | `int` | Encoder output width for non-recurrent paths |
| `layers` | `list` | Layer definitions passed to the encoder builder |
| `aux_type` | `str \| null` | Auxiliary head family attached to this encoder |
| `aux_latent_dim` | `int` | Projection size for contrastive or BYOL heads |
| `use_momentum` | `bool` | Wrap encoder with momentum/EMA behavior |
| `momentum_tau` | `float` | EMA coefficient |
| `frame_stack` | `int` | Declared stacked-frame factor |
| `recurrent` | `bool` | Wrap non-LSTM encoders with LSTM |
| `lstm_hidden` | `int` | Hidden size for recurrent wrapping |

### `input_name`

`input_name` is the preferred way to map observation keys to encoders in multimodal systems.

Example:

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_name: joint_states
    input_dim: 24
    latent_dim: 128

  - name: image_enc
    type: cnn
    input_name: front_camera
    input_shape: [3, 84, 84]
    latent_dim: 256
```

When `input_name` is set:

- the runtime routes by that key first
- missing keys raise `KeyError`
- extra keys that remain unused generate a warning

## Layer entries

Layer lists accept either shorthand or explicit dictionaries depending on the encoder/head builder.

Common dictionary fields from `LayerConfig`:

| Field | Meaning |
|---|---|
| `out_features` | MLP output width |
| `out_channels` | CNN output channels |
| `kernel_size` | CNN kernel size |
| `stride` | CNN stride |
| `padding` | CNN padding |
| `activation` | Activation function name |
| `norm` | Normalization type |
| `dropout` | Dropout probability |
| `dropout_type` | Standard, variational, or spatial |
| `pooling` | Pooling type |
| `pooling_kernel` | Pooling kernel size |
| `residual` | Residual block flag |
| `depthwise` | Depthwise convolution flag |
| `norm_order` | `pre` or `post` normalization |
| `weight_init` | Weight init strategy |

## Flow graph

`flows` is a list of strings in the form `"src -> dst"`.

```yaml
flows:
  - "visual_enc -> actor"
  - "state_enc -> actor"
  - "visual_enc -> critic"
  - "state_enc -> critic"
```

Rules:

- sources and destinations must be declared nodes
- cycles are not allowed
- multiple upstream inputs are concatenated automatically
- actor and critic can consume different subsets of encoders

## Head fields

Common `HeadConfig` fields:

| Field | Meaning |
|---|---|
| `name` | Node name, typically `actor` or `critic` |
| `type` | Built-in head type or registry key |
| `action_dim` | Required for actor heads and Q-style critics |
| `layers` | Optional head-specific MLP layers |
| `log_std_init` | Initial log standard deviation |
| `log_std_min` | Minimum log std clamp |
| `log_std_max` | Maximum log std clamp |
| `state_dependent_std` | Whether std is state-conditioned |

---

## Actor head types

| Type | Distribution | Use with |
|---|---|---|
| `gaussian` | Normal — unbounded | PPO, A2C, A3C |
| `squashed_gaussian` | Tanh-Normal — bounded [−1, 1] | SAC |
| `deterministic` | No distribution | DDPG |

---

## Critic head types

| Type | Output | Use with |
|---|---|---|
| `value` | `V(s)` scalar | PPO, A2C, A3C |
| `twin_q` | `[Q1(s,a), Q2(s,a)]` | SAC, DDPG |

---

## Loss names

| Name | Description | Algorithm |
|---|---|---|
| `policy` | PPO surrogate loss | PPO |
| `value` | Value-function MSE | PPO, A2C |
| `entropy` | Entropy bonus | PPO, A2C |
| `sac_q` | Twin-Q Bellman loss | SAC |
| `sac_policy` | Actor log probability | SAC |
| `sac_temperature` | Alpha auto-tuning | SAC |
| `ddpg_critic` | Bellman MSE | DDPG |
| `ddpg_actor` | Deterministic policy | DDPG |
| `td3_critic` | Bellman MSE (twin) | TD3 |
| `td3_actor` | Deterministic policy | TD3 |
| `aux_reconstruction` | Autoencoder MSE reconstruction | SAC (ae / vae) |
| `aux_curl` | InfoNCE contrastive loss | SAC (curl) |
| `aux_byol` | BYOL self-prediction loss | SAC (byol) |
| `aux_vae` | VAE MSE + KL divergence | SAC (vae) |
| `aux_drq` | DrQ Q-value augmentation consistency | SAC (drq) |
| `aux_spr` | SPR latent forward-prediction | SAC (spr) |
| `aux_barlow` | Barlow Twins redundancy reduction | SAC (barlow) |

Loss entries also support a `schedule` field with the current built-in options `constant`, `linear_decay`, and `cosine`.

## Train section

The `train:` block is not parsed by `ModelBuilder`, but it is part of the practical YAML workflow because the CLI consumes it and maps it into algorithm config dataclasses.

Common fields seen in repo configs:

- `total_steps`
- `n_envs`
- `n_steps`
- `n_epochs`
- `batch_size`
- `lr`, `lr_actor`, `lr_critic`, `lr_alpha`
- `gamma`, `tau`, `gae_lambda`
- `clip_range`
- `buffer_size`
- `start_steps`, `update_after`, `update_every`, `gradient_steps`

See [halfcheetah_sac.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/halfcheetah_sac.yaml), [car_racing_ppo_visual.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/car_racing_ppo_visual.yaml), and [isaaclab_cartpole_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_cartpole_ppo.yaml) for real examples.

---

## Example: Multi-modal (state + lidar)

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 12
    latent_dim: 128
    layers:
      - {out_features: 128, activation: relu, norm: layer_norm}

  - name: lidar_enc
    type: mlp
    input_dim: 1080
    latent_dim: 128
    layers:
      - {out_features: 256, activation: relu, norm: none}
      - {out_features: 128, activation: relu, norm: none}

flows:
  - "state_enc -> actor"
  - "state_enc -> critic"
  - "lidar_enc -> actor"
  - "lidar_enc -> critic"
```

---

## Encoder optimizer fields (v0.2.0)

These fields appear on `SACConfig`, `DDPGConfig`, `TD3Config`, and `VisualSACConfig`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `encoder_update_freq` | `int` | `1` (state), `2` (visual) | Step the encoder optimizer every N critic updates |

`VisualSACConfig`-only fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `encoder_optimize_with_critic` | `bool` | `True` | Allow critic gradients to flow into the encoder |
| `aux_loss_type` | `str` | `"curl"` | Auxiliary encoder loss: `none \| ae \| vae \| curl \| byol \| drq \| spr \| barlow` |
| `lr_encoder` | `float` | `1e-4` | Learning rate for the dedicated encoder optimizer |

---

## AsyncRunnerConfig (v0.2.0)

Controls `AsyncOffPolicyRunner`. Pass an instance under `runner_cfg` to the CLI or
construct it directly.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `use_async` | `bool` | `False` | Split collection and training into separate threads |
| `use_gpu_buffer` | `bool` | `False` | Replace the CPU replay buffer with `GPUReplayBuffer` |
| `prefill_steps` | `int` | `1000` | Random-action prefill before gradient updates start |
| `queue_maxsize` | `int` | `4` | Max transitions queued between collector and trainer |

```python
from srl.core.config import AsyncRunnerConfig

runner_cfg = AsyncRunnerConfig(
    use_async      = True,
    use_gpu_buffer = True,
    prefill_steps  = 5000,
)
```

See [async_runner.md](async_runner.md) for the full runner API.
