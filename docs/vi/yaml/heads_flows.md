# Heads & Flows

## Actor Heads

Actor head nhận latent vector từ encoder và output action distribution.

| Type | Mô tả | Dùng cho |
|---|---|---|
| `squashed_gaussian` | Gaussian → tanh squash → bounded actions | SAC, vision tasks |
| `gaussian` | Gaussian không squash | PPO, unbounded action spaces |
| `deterministic` | Output mean trực tiếp | DDPG, TD3 |

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

## Critic Heads

| Type | Mô tả | Dùng cho |
|---|---|---|
| `twin_q` | Hai Q-networks (min để giảm overestimation) | SAC, TD3 |
| `q` | Một Q-network | DDPG |
| `value` | State value V(s) | PPO, A2C, A3C |

```yaml
critic:
  name: critic
  type: twin_q
  action_dim: 6
  layers:
    - {out_features: 256, activation: relu, norm: layer_norm}
    - {out_features: 256, activation: relu}
```

!!! note
    Builder tự tính kích thước input của head từ flow graph — không cần hard-code `input_dim`.

## Flows — Routing Graph

`flows` định nghĩa data path từ encoder đến heads dưới dạng directed edges.

```yaml
flows:
  - "encoder_name -> actor"
  - "encoder_name -> critic"
```

### Tính chất quan trọng

- **Concatenation tự động**: nhiều encoder input vào cùng một head được concatenate
- **Topological ordering**: thứ tự thực thi được resolve tự động
- **Asymmetric branches**: actor và critic có thể dùng encoder set khác nhau
- **Encoder chaining**: encoder-to-encoder connections được hỗ trợ

### Ví dụ: Symmetric

```yaml
flows:
  - "state_enc -> actor"
  - "state_enc -> critic"
```

### Ví dụ: Asymmetric multi-modal

```yaml
# Actor thấy image + state; critic chỉ thấy state
flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "state_enc -> critic"
```

### Ví dụ: Full multi-modal symmetric

```yaml
flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "image_enc -> critic"
  - "state_enc -> critic"
```

## Layer specification

### Dict style (MLP — khuyến nghị)

```yaml
layers:
  - {out_features: 256, activation: relu, norm: layer_norm}
  - {out_features: 256, activation: relu, norm: none}
```

### List style (CNN)

```yaml
# [out_channels, kernel_size, stride, activation]
layers:
  - [32, 8, 4, relu]
  - [64, 4, 2, relu]
  - [64, 3, 1, relu]
```

### Activation options

`relu`, `elu`, `tanh`, `gelu`, `silu`, `none`

### Norm options

`layer_norm`, `batch_norm`, `none`
