# Encoders

Encoder chuyển đổi observation (ảnh, vector, text) thành latent vector dùng bởi actor và critic.

## Chọn encoder nào?

| Loại | Trường hợp dùng | Ví dụ môi trường |
|---|---|---|
| `mlp` | Vector state (joint angles, velocities) | HalfCheetah, Pendulum, Isaac Lab |
| `cnn` | Ảnh pixel (RGB, depth) | CarRacing, Isaac Lab visual |
| `lstm` | Chuỗi thời gian, stacked frames | POMDPs, partially observable envs |
| `text` | Embedding ngôn ngữ/câu lệnh | Language-conditioned tasks |

## Các trường cấu hình

| Trường | Bắt buộc | Mô tả |
|---|---|---|
| `name` | ✓ | ID duy nhất, dùng trong `flows` |
| `type` | ✓ | `mlp`, `cnn`, `lstm`, `text`, hoặc registry key |
| `input_name` | khuyến nghị | Key trong observation dict |
| `input_dim` | mlp/lstm | Số chiều input vector |
| `input_shape` | cnn | `[C, H, W]` |
| `latent_dim` | ✓ | Kích thước output latent |
| `layers` | ✓ | Cấu trúc layers |
| `aux_type` | | `autoencoder`, `contrastive`, `byol`, `vae`, `drq`, `spr`, `barlow` |
| `aux_latent_dim` | | Kích thước projection head |
| `use_momentum` | | `true` cho BYOL/contrastive momentum |
| `momentum_tau` | | EMA coefficient (0.99 thường dùng) |
| `recurrent` | | Wrap encoder bằng LSTM |
| `lstm_hidden` | | Kích thước hidden khi dùng LSTM |
| `frame_stack` | | Số frames được stack |

## Ví dụ MLP (state-based)

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

## Ví dụ CNN (vision)

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

## Ví dụ CNN với auxiliary reconstruction

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

## Ví dụ đa phương thức (image + state)

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

!!! tip
    Luôn khai báo `input_name` khi có nhiều encoder để tránh ambiguous routing.

## Encoder optimizer (v0.2.0)

Encoder có optimizer riêng biệt để tránh double-update:

```yaml
train:
  encoder_update_freq: 2           # cập nhật encoder mỗi 2 critic steps
  encoder_optimize_with_critic: true
  lr_encoder: 3e-4
```

| Optimizer | Tham số | Khi nào step |
|---|---|---|
| `critic_optimizer` | Q-head params | Mỗi gradient step |
| `actor_optimizer` | Actor head params | Mỗi gradient step |
| `encoder_optimizer` | Tất cả encoder params | Mỗi `encoder_update_freq` critic steps |

## Xem thêm

- [Auxiliary Representation Learning](auxiliary.md)
- [Heads & Flows](heads_flows.md)
- [Training Block Reference](training_block.md)
