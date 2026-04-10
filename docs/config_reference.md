# Configuration Reference

SRL uses YAML files to define model architecture.  
The Python `ModelBuilder.from_yaml(path)` parses the file and constructs all networks.

---

## File structure

```yaml
# Optional — used by CLI to select algorithm
algo: ppo | sac | ddpg | a2c | a3c

# List of encoder modules
encoders:
  - name: <str>           # unique key; referenced in flows
    type: mlp | cnn | gru | text
    input_dim: <int>      # required for mlp/gru
    latent_dim: <int>     # output dimension
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
```

---

## Encoder types

| Type | `input_dim` | Notes |
|---|---|---|
| `mlp` | obs dimension | Fully-connected layers |
| `cnn` | `[C, H, W]` | Convolutional encoder for pixels |
| `gru` | obs dimension | Recurrent; wraps an MLP |
| `text` | vocabulary size | Embedding + LSTM |

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
