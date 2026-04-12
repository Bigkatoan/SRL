# Algorithms

SRL hỗ trợ 6 RL algorithms cho continuous action spaces.

## Tổng quan

| Algorithm | Type | Ưu điểm | Dùng khi |
|---|---|---|---|
| **PPO** | On-policy | Ổn định, sample-efficient | Locomotion, Isaac Lab |
| **SAC** | Off-policy | Hiệu suất cao, auto-entropy | Manipulation, high-DoF |
| **DDPG** | Off-policy | Đơn giản, deterministic | Low-DoF tasks |
| **TD3** | Off-policy | Giảm Q overestimation | Continuous control |
| **A2C** | On-policy | Nhẹ, nhanh | Nhiều parallel envs |
| **A3C** | On-policy async | Multi-CPU | CPU-heavy simulations |

---

## PPO

```python
from srl.core.config import PPOConfig

cfg = PPOConfig(
    lr=3e-4, n_steps=2048, num_envs=8,
    batch_size=256, n_epochs=10,
    gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, entropy_coef=0.0,
    vf_coef=0.5, max_grad_norm=0.5,
)
```

Isaac Lab — tăng `n_envs`, giảm `n_steps`:

```yaml
train:
  total_steps: 5_000_000
  n_envs: 4096
  n_steps: 32
  batch_size: 16384
  n_epochs: 5
  lr: 5e-4
  vf_coef: 1.0
  entropy_coef: 0.005
  max_grad_norm: 1.0
```

---

## SAC

SAC maximize return + entropy. Tự điều chỉnh temperature alpha.

```python
from srl.core.config import SACConfig

cfg = SACConfig(
    lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
    buffer_size=1_000_000, batch_size=256,
    gamma=0.99, tau=0.005, action_dim=6,
    learning_starts=10_000,
    encoder_update_freq=1,
    auto_entropy_tuning=True,
)
```

### Visual SAC — 3-optimizer design (v0.2.0)

| Optimizer | Parameters | Khi nào step |
|---|---|---|
| `critic_optimizer` | Q-head params | Mỗi gradient step |
| `actor_optimizer` | Actor head params | Mỗi gradient step |
| `encoder_optimizer` | Tất cả encoder params | Mỗi `encoder_update_freq` critic steps |

```yaml
train:
  encoder_update_freq: 2
  encoder_optimize_with_critic: true
  lr_encoder: 1e-4
  aux_loss_type: curl
```

---

## TD3

```python
from srl.core.config import TD3Config

cfg = TD3Config(
    lr_actor=3e-4, lr_critic=3e-4,
    buffer_size=1_000_000, batch_size=256,
    gamma=0.99, tau=0.005, action_dim=6,
    policy_noise=0.2, noise_clip=0.5, policy_delay=2,
)
```

TD3 dùng separate `encoder_optimizer` như SAC. Encoder step mỗi `encoder_update_freq` critic steps, độc lập với `policy_delay`.

---

## DDPG

```python
from srl.core.config import DDPGConfig

cfg = DDPGConfig(
    lr_actor=1e-4, lr_critic=1e-3,
    buffer_size=1_000_000, batch_size=256,
    gamma=0.99, tau=0.005, action_dim=6,
    action_noise="gaussian", noise_sigma=0.1,
)
```

---

## Checkpointing

Tất cả algorithms tự động lưu checkpoint qua `CheckpointManager`:

```
runs/sac_halfcheetah/
  checkpoints/
    ckpt_0000005000.pt
    final_0001000000.pt
  metrics.jsonl
  summary.json
```

Resume:

```bash
srl-train --config configs/envs/halfcheetah_sac.yaml \
          --resume runs/sac_halfcheetah/checkpoints/final_0001000000.pt
```

## Xem thêm

- [Runners & Training Loop](runners.md)
- [Auxiliary Losses](../yaml/auxiliary.md)
- [Config Reference](../config_reference.md)
