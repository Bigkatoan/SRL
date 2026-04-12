# Ví dụ: Cấu hình Encoders cho SAC / TD3 / Actor-Critic

Trang này chứa các ví dụ YAML và các bước kiểm thử nhanh để bạn có thể thử encoder trong các cấu hình thực tế.

## 1) SAC — Vision (CNN) + encoder riêng

Mục tiêu: sử dụng `cnn` encoder, tách `encoder_optimizer`, và cập nhật encoder mỗi `encoder_update_freq` critic updates.

```yaml
# configs/examples/sac_image_encoder.yaml (snippet)
env_id: CarRacing-v3
algo: sac

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

actor:
  name: actor
  type: squashed_gaussian
  action_dim: 3

critic:
  name: critic
  type: twin_q
  action_dim: 3

train:
  total_steps: 200000
  batch_size: 256
  encoder_update_freq: 2
  encoder_optimize_with_critic: true
  lr_encoder: 3e-4
  lr_actor: 3e-4
  lr_critic: 3e-4
  buffer_size: 100000
  start_steps: 1000
  update_after: 1000
  update_every: 50
```

Kiểm thử nhanh (smoke):

```bash
python -m srl.cli.train --config configs/examples/sac_image_encoder.yaml --device cpu --seed 0 --no-plots
```

Nếu bạn chạy trên GPU trong venv đã cấu hình:

```bash
tests/venv/bin/python -m srl.cli.train --config configs/examples/sac_image_encoder.yaml --device cuda --seed 0
```

---

## 2) TD3 — MLP (state-based) nhanh

Ví dụ cho môi trường vector-only (HalfCheetah). TD3 thường dùng deterministic actor.

```yaml
# configs/examples/td3_state_encoder.yaml (snippet)
env_id: HalfCheetah-v5
algo: td3

encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 128
    layers:
      - {out_features: 256, activation: relu}
      - {out_features: 128, activation: relu}

actor:
  name: actor
  type: deterministic
  action_dim: 6

critic:
  name: critic
  type: q
  action_dim: 6

train:
  total_steps: 300000
  batch_size: 256
  lr_encoder: 1e-3
  lr_actor: 1e-3
  lr_critic: 1e-3
```

Smoketest:

```bash
python -m srl.cli.train --config configs/examples/td3_state_encoder.yaml --device cpu --no-plots
```

---

## 3) Actor-Critic đa phương thức (image + state)

Khi actor/critic cần thông tin khác nhau, tách encoder cho từng branch.

```yaml
# configs/examples/multi_modal_ac.yaml
env_id: SomeEnv-v0
algo: sac

encoders:
  - name: image_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 128
  - name: state_enc
    type: mlp
    input_dim: 10
    latent_dim: 64

flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "image_enc -> critic"
  - "state_enc -> critic"

actor:
  name: actor
  type: squashed_gaussian
  action_dim: 4

critic:
  name: critic
  type: twin_q
  action_dim: 4

train:
  total_steps: 500000
  encoder_update_freq: 2
  lr_encoder: 3e-4
  lr_actor: 3e-4
  lr_critic: 3e-4
```

---

## Tips & test checklist

- Luôn khai báo `input_name` khi observation dict có nhiều key (tránh mapping heuristics).
- Kiểm tra `ModelBuilder.from_yaml(...).summary()` để xác nhận kích thước latent input cho heads.
- Nếu dùng contrastive/BYOL: bật `use_momentum: true` và tăng batch size/augmentations.
- Với vision trên Isaac Lab: ưu tiên `encoder_update_freq=2`, mixed precision và batch lớn khi có GPU.

## Nơi lưu ví dụ

Lưu các file config ví dụ trong `configs/examples/` để dễ dùng lại.

---

Xem thêm hướng dẫn tổng quan: [Encoders guide](../encoders.md)
