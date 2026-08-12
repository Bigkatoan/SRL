# Auxiliary Representation Learning

Auxiliary losses giúp encoder học biểu diễn tốt hơn cho vision tasks bằng cách thêm objective song song với RL loss chính.

## Cấu hình

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    aux_type: curl
    aux_latent_dim: 128
    use_momentum: true
    momentum_tau: 0.99

train:
  aux_loss_type: curl       # none|ae|vae|curl|byol|drq|spr|barlow
  encoder_update_freq: 2
  encoder_optimize_with_critic: true
```

## Các auxiliary loss types

| Type | Phương pháp | Modules cần | Tốt cho |
|---|---|---|---|
| `none` | Không có auxiliary | — | State-based SAC/TD3 |
| `ae` | Autoencoder (MSE reconstruction) | `ConvDecoderHead` | Baseline vision |
| `vae` | Variational AE (MSE + KL) | `VAEHead`, `ConvDecoderHead` | Generative representation |
| `curl` | CURL InfoNCE contrastive | `ProjectionHead` | Default vision SAC |
| `byol` | BYOL self-prediction | Momentum encoder, `ProjectionHead` | Stable contrastive |
| `drq` | DrQ Q-value augmentation consistency | Augmentation pipeline | Data-augmented RL |
| `spr` | SPR latent forward prediction | `LatentTransitionModel` | Model-based auxiliary |
| `barlow` | Barlow Twins redundancy reduction | `ProjectionHead` | Decorrelated features |

## Chi tiết từng loại

### `ae` — Autoencoder

Tái tạo lại observation từ latent. Đơn giản, baseline tốt.

```yaml
encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    aux_type: autoencoder
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]

train:
  aux_loss_type: ae
```

### `curl` — Contrastive Unsupervised Representations for RL

Học representation bằng cách so sánh augmented views của cùng một observation.

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
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]

train:
  aux_loss_type: curl
  batch_size: 128
```

### `byol` — Bootstrap Your Own Latent

Không cần negative samples. Ổn định hơn contrastive trong nhiều tasks.

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

train:
  aux_loss_type: byol
```

### `drq` — Data-regularized Q

Áp dụng random augmentation và enforce Q-value consistency.

```yaml
train:
  aux_loss_type: drq
  encoder_update_freq: 2
```

## Best practices

1. **Batch size**: contrastive losses cần batch >= 128; 256 tốt nhất.
2. **Momentum tau**: 0.99 cho contrastive; 0.995 cho BYOL.
3. **`aux_latent_dim`**: thường nhỏ hơn `latent_dim` (ví dụ 64 vs 256).
4. **Learning rate**: `lr_encoder` thường thấp hơn `lr_critic` khi dùng auxiliary (1e-4 vs 3e-4).
5. **Bắt đầu với `ae`** nếu không chắc — đơn giản và đáng tin cậy.

## Xem thêm

- [Encoders](encoders.md)
- [Algorithms](../training/algorithms.md)
