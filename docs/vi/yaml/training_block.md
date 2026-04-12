# Training Block Reference

Block `train:` trong file YAML chứa tất cả hyperparameters được CLI đọc và chuyển thành algorithm config dataclasses.

## Các trường chung (tất cả algorithms)

| Trường | Kiểu | Default | Mô tả |
|---|---|---|---|
| `total_steps` | int | 1_000_000 | Tổng số environment steps |
| `batch_size` | int | 256 | Batch size cho gradient update |
| `gamma` | float | 0.99 | Discount factor |
| `device` | str | `auto` | `cpu`, `cuda`, `auto` |
| `seed` | int | 0 | Random seed |
| `n_envs` | int | 1 | Số parallel environments |
| `log_interval` | int | 1000 | Log metrics mỗi N steps |
| `eval_freq` | int | 10000 | Eval mỗi N steps |
| `eval_episodes` | int | 10 | Số episodes mỗi eval |

## Off-policy (SAC, DDPG, TD3)

| Trường | Kiểu | Default | Mô tả |
|---|---|---|---|
| `lr` | float | 3e-4 | Learning rate chung |
| `lr_actor` | float | 3e-4 | Learning rate actor |
| `lr_critic` | float | 3e-4 | Learning rate critic |
| `lr_encoder` | float | 3e-4 | Learning rate encoder optimizer |
| `lr_alpha` | float | 3e-4 | Learning rate entropy coef (SAC) |
| `tau` | float | 0.005 | Soft target update coefficient |
| `buffer_size` | int | 1_000_000 | Replay buffer capacity |
| `start_steps` | int | 10000 | Random steps trước khi train |
| `update_after` | int | 1000 | Bắt đầu update sau N steps |
| `update_every` | int | 1 | Update mỗi N env steps |
| `gradient_steps` | int | 1 | Số gradient updates mỗi lần update |
| `encoder_update_freq` | int | 1 | Cập nhật encoder mỗi N critic steps |
| `encoder_optimize_with_critic` | bool | true | Encoder học qua critic backward |
| `aux_loss_type` | str | `none` | `none\|ae\|vae\|curl\|byol\|drq\|spr\|barlow` |

## On-policy (PPO, A2C)

| Trường | Kiểu | Default | Mô tả |
|---|---|---|---|
| `lr` | float | 3e-4 | Learning rate |
| `n_steps` | int | 2048 | Steps thu thập mỗi rollout |
| `n_epochs` | int | 10 | PPO epochs mỗi rollout |
| `clip_range` | float | 0.2 | PPO clip parameter |
| `gae_lambda` | float | 0.95 | GAE lambda |
| `entropy_coef` | float | 0.0 | Entropy regularization |
| `vf_coef` | float | 0.5 | Value loss coefficient |
| `max_grad_norm` | float | 0.5 | Gradient clipping |

## Async runner (v0.2.0)

| Trường | Kiểu | Default | Mô tả |
|---|---|---|---|
| `use_async` | bool | false | Bật collector/trainer thread split |
| `use_gpu_buffer` | bool | false | Dùng GPUReplayBuffer thay CPU buffer |
| `prefill_steps` | int | 1000 | Random steps trước gradient update đầu tiên |

## Ví dụ đầy đủ — SAC vision task

```yaml
train:
  total_steps:                  500_000
  batch_size:                   128
  gamma:                        0.99
  n_envs:                       1
  lr_actor:                     3e-4
  lr_critic:                    3e-4
  lr_encoder:                   1e-4
  lr_alpha:                     3e-4
  tau:                          0.005
  buffer_size:                  100_000
  start_steps:                  1000
  update_after:                 1000
  update_every:                 50
  gradient_steps:               1
  encoder_update_freq:          2
  encoder_optimize_with_critic: true
  aux_loss_type:                curl
  eval_freq:                    10000
  eval_episodes:                10
```

## Ví dụ đầy đủ — PPO Isaac Lab

```yaml
train:
  total_steps:   5_000_000
  n_envs:        4096
  n_steps:       32
  batch_size:    16384
  n_epochs:      5
  lr:            5e-4
  entropy_coef:  0.005
  vf_coef:       1.0
  max_grad_norm: 1.0
  gae_lambda:    0.95
  gamma:         0.99
  eval_freq:     50000
  eval_episodes: 10
```
