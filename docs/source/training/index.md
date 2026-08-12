# Training System

SRL's training system bao gồm 3 lớp: **Algorithms** (RL logic), **Runners** (training loop), và **Buffers** (lưu trữ transitions).

## Kiến trúc tổng quan

```
Environment
     ↓ observations
  Runner (training loop)
     ↓ collects transitions
  Buffer (replay / rollout storage)
     ↓ samples batches
  Algorithm (gradient updates)
     ↓ updates model
  Actor/Critic heads
```

## Chọn cấu hình phù hợp

| Tình huống | Algorithm | Runner | Buffer |
|---|---|---|---|
| State continuous control | SAC / TD3 | Sync off-policy | CPU ReplayBuffer |
| Vision (GPU sim) | SAC + CURL | Async off-policy | GPUReplayBuffer |
| On-policy locomotion | PPO | On-policy rollout | RolloutBuffer |
| Isaac Lab large-scale | PPO / SAC | Async + GPU buffer | GPUReplayBuffer |
| Debugging / quick test | Bất kỳ | Sync | CPU |

## Các trang trong phần này

- [Algorithms](algorithms.md) — PPO, SAC, DDPG, TD3, A2C, A3C config và hyperparameters
- [Runners & Training Loop](runners.md) — synchronous/asynchronous runners
- [Replay Buffers](buffers.md) — CPU ReplayBuffer và GPU ReplayBuffer

## Quick start

```bash
# SAC chuẩn
srl-train --config configs/envs/halfcheetah_sac.yaml --device cuda

# PPO Isaac Lab
srl-train --config configs/envs/isaaclab_ant_ppo.yaml --device cuda --n-envs 4096
```
