# Isaac Lab Integration

Isaac Lab là GPU-accelerated robot learning framework của NVIDIA. SRL tích hợp natively thông qua `IsaacLabWrapper`.

## Yêu cầu

- NVIDIA GPU (RTX 3090 hoặc tốt hơn)
- Isaac Sim ≥ 5.1 + Isaac Lab ≥ 0.5
- Python 3.10 hoặc 3.11
- SRL cài vào đúng Isaac Lab Python environment

## Setup

### 1. Cài Isaac Lab

Làm theo [Isaac Lab official guide](https://isaac-sim.github.io/IsaacLab/).

### 2. Cài SRL vào Isaac Lab environment

```bash
# Kích hoạt Isaac Lab environment trước
source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh
conda activate isaaclab

# Cài SRL
pip install git+https://github.com/Bigkatoan/SRL.git

# Kiểm tra
python -c "import isaaclab; import srl; print('OK')"
```

!!! warning "Quan trọng"
    SRL phải được cài vào đúng Isaac Lab environment. Nếu cài vào environment khác, imports sẽ fail tại runtime dù CLI tìm thấy.

**Thứ tự quan trọng:**

1. Kích hoạt Isaac Lab environment
2. Xác nhận `python -c "import isaaclab"` pass
3. Cài SRL vào environment đó
4. Chạy `srl-train` từ shell đó

## Observation keys

`input_name` trong encoder config phải khớp với Isaac Lab observation group key:

```yaml
encoders:
  - name: policy_enc
    type: mlp
    input_name: policy          # ← phải khớp Isaac Lab obs group key
    input_dim: 60
    latent_dim: 256
```

Kiểm tra keys: `python -c "from isaaclab.envs import ...; print(env.observation_space)"`

## PPO Isaac Lab (khuyến nghị)

```yaml
# configs/envs/isaaclab_ant_ppo.yaml
env_id:   Isaac-Ant-v0
env_type: flat
algo:     ppo

encoders:
  - name: state_enc
    type: mlp
    input_name: policy
    input_dim: 60
    latent_dim: 256
    layers:
      - {out_features: 256, activation: elu}
      - {out_features: 128, activation: elu}

flows:
  - "state_enc -> actor"
  - "state_enc -> critic"

actor:
  name: actor
  type: gaussian
  action_dim: 8

critic:
  name: critic
  type: value

train:
  total_steps:   5_000_000
  n_envs:        4096
  n_steps:       32
  batch_size:    16384
  n_epochs:      5
  lr:            5e-4
  vf_coef:       1.0
  entropy_coef:  0.005
  max_grad_norm: 1.0
  gae_lambda:    0.95
  gamma:         0.99
```

```bash
srl-train --config configs/envs/isaaclab_ant_ppo.yaml --device cuda
```

## SAC + Async + GPU Buffer

Tận dụng zero-copy GPU buffer với async runner:

```yaml
train:
  use_async:      true
  use_gpu_buffer: true
  encoder_update_freq: 2
```

```bash
srl-train --config configs/envs/isaaclab_visual_sac.yaml --device cuda
```

## GPU memory tips

- Dùng `elu` activations (gradients smoother hơn `relu`)
- `vf_coef=1.0` cân bằng policy và value loss trong Isaac Lab
- Tăng `n_envs` thay vì `n_steps` khi có đủ VRAM
- `use_gpu_buffer=True` để tránh CPU↔GPU copies

## Troubleshooting

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `ModuleNotFoundError: isaaclab` | Sai environment | Kích hoạt đúng Isaac Lab env |
| `ModuleNotFoundError: pxr` | Isaac Sim chưa setup | Chạy `source setup_conda_env.sh` trước |
| Observation routing errors | `input_name` không khớp | Kiểm tra `env.observation_space` keys |
| Vision config import fails | Isaac Lab version cũ | Nâng cấp hoặc dùng state-based tasks |

## Xem thêm

- [GPU Replay Buffer](../training/buffers.md)
- [Runners & Training Loop](../training/runners.md)
- [Algorithms — PPO](../training/algorithms.md)
- [Environments — Isaac Lab](../environments/isaaclab.md)
