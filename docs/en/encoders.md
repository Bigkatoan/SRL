# Hướng dẫn Encoders (Thiết lập & Best Practices)

Tóm tắt ngắn: trang này mô tả vai trò của `encoder` trong pipeline SRL, các kiểu encoder tích hợp, các trường cấu hình quan trọng, ví dụ YAML, lời khuyên tối ưu hóa và bước kiểm thử đơn giản.

## 1. Tổng quan

Encoder là module chuyển đổi observation (ảnh, vector, text, ...) thành latent vectors dùng cho actor/critic.
Encoder ở lớp giới hạn giữa môi trường và các head chính — chọn và cấu hình encoder đúng là bước quyết định cho hiệu năng.

## 2. Khi nào dùng loại encoder nào

- mlp: dữ liệu vector/low-dim (state). Nhẹ, nhanh.
- cnn: ảnh (vision). Tránh quá sâu nếu dùng realtime/Isaac Lab.
- lstm: chuỗi/độ trễ/stacked frames với phụ thuộc thời gian.
- text: input kí tự hoặc embedding cho ngôn ngữ.

## 3. Các trường cấu hình quan trọng (schema)

- `name` — id của encoder trong `flows`
- `type` — `mlp`, `cnn`, `lstm`, `text`, hoặc key registry
- `input_name` — tên key trong observation dict (khuyến nghị luôn khai báo)
- `input_dim` / `input_shape` — kích thước input
- `latent_dim` — kích thước output latent
- `layers` — cấu trúc lớp (builder-driven)
- `aux_type`, `aux_latent_dim` — nếu dùng auxiliary heads (ae, vae, byol, contrastive...)
- `use_momentum`, `momentum_tau` — bật momentum encoder cho contrastive/BYOL
- `recurrent`, `lstm_hidden`, `frame_stack`

Tham khảo schema chi tiết: [srl/registry/config_schema.py](https://github.com/Bigkatoan/SRL/blob/main/srl/registry/config_schema.py)

## 4. Ví dụ cấu hình YAML

Ví dụ MLP (state):

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

Ví dụ CNN (vision) với auxiliary reconstruction:

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

## 5. Optimizer & update policy cho encoder

Đề xuất cấu hình train liên quan tới encoder:

- Tách `encoder_optimizer` riêng (tránh cập nhật kép khi actor/critic backward đều chứa encoder params).
- `encoder_update_freq` — cập nhật encoder mỗi N bước critic; thường `2` cho vision.
- `encoder_optimize_with_critic` — nếu true, critic loss sẽ cập nhật encoder (thay vì actor).

Ví dụ (train block):

```yaml
train:
  total_steps: 1_000_000
  encoder_update_freq: 2
  encoder_optimize_with_critic: true
  lr_encoder: 3e-4
  lr_actor: 3e-4
  lr_critic: 3e-4
```

## 6. Auxiliary representation learning

Các kiểu phụ trợ: `autoencoder`, `vae`, `contrastive`, `byol`, `drq`, `spr`, `barlow`.

- Dùng `use_momentum=true` khi cấu hình BYOL hoặc contrastive momentum.
- Projection head thường có `aux_latent_dim` nhỏ hơn latent chính.
- Lưu ý: augmentations và batch size có ảnh hưởng lớn tới chất lượng contrastive.

## 7. Kết nối với actor/critic (`flows`)

- Định nghĩa rõ `flows` để actor/critic nhận đúng encoder:

```yaml
flows:
  - "image_enc -> actor"
  - "state_enc -> actor"
  - "image_enc -> critic"
  - "state_enc -> critic"
```

- Khuyến nghị: nếu actor và critic cần biểu diễn khác nhau, tách encoder cho từng branch.

## 8. Kiểm thử & debug nhanh

- Nếu gặp lỗi shape, kiểm tra `input_shape` / `input_dim` và `input_name` mapping.
- Khởi tạo model bằng `ModelBuilder.from_yaml(...)` và in `builder.summary()` để xem kích thước latent.
- Chạy smoke test:

```bash
# module fallback từ source
python -m srl.cli.train --config configs/envs/halfcheetah_sac.yaml --device cpu --seed 0 --no-plots
```

## 9. Best practices (tham khảo NVIDIA & công nghiệp)

- Với vision/Isaac: tận dụng momentum encoder cho contrastive/BYOL; dùng mixed-precision nếu có GPU.
- Tránh mạng quá sâu cho realtime; ưu tiên batch lớn + augment cho contrastive.
- Tách optimizer để điều chỉnh lr riêng cho encoder khi cần fine-tune.
- Kiểm tra pipeline bằng small end-to-end smoke test trước khi scale.

Tham khảo tài liệu NVIDIA Isaac/Isaac Sim khi làm tích hợp môi trường GPU: https://docs.omniverse.nvidia.com/isaac/ (tài liệu tham khảo cấu trúc và best-practices về môi trường GPU).

## 10. Liên kết nhanh



Nếu bạn đồng ý, tôi sẽ:

## Ví dụ và hướng dẫn thêm

Xem các ví dụ cấu hình và các bước kiểm thử nhanh tại: [Examples: encoder examples](examples/encoder_examples.md)

Nếu bạn muốn tôi triển khai thêm, tôi có thể:
- mở rộng các cấu hình ví dụ thành file YAML đầy đủ trong `configs/examples/`
- thêm hướng dẫn kiểm thử CI/smoke tests
- tổ chức lại nav theo mẫu NVIDIA (Architecture → Encoders → Deployment)
Bạn muốn tôi triển khai tiếp phần nào trước?