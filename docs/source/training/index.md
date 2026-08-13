# Training System

SRL's training system has three layers: **algorithms** (the RL logic), **runners**
(the training loop), and **buffers** (transition storage).

## Architecture overview

```
Environment
     ↓ observations
  Runner (training loop)
     ↓ collects transitions
  Buffer (replay / rollout storage)
     ↓ samples batches
  Algorithm (gradient updates)
     ↓ updates model
  Actor/critic heads
```

## Choosing a configuration

| Situation | Algorithm | Runner | Buffer |
|---|---|---|---|
| State-based continuous control | SAC / TD3 | Sync off-policy | CPU `ReplayBuffer` |
| Vision on a GPU simulator | SAC + CURL | Async off-policy | `GPUReplayBuffer` |
| On-policy locomotion | PPO | On-policy rollout | `RolloutBuffer` |
| Isaac Lab at scale | PPO / SAC | Async + GPU buffer | `GPUReplayBuffer` |
| Debugging / quick test | Any | Sync | CPU |

```{note}
The async runner and the GPU replay buffer are currently reachable from the Python
API only — see [Runners & Training Loop](runners.md).
```

## Pages in this section

- [Algorithms](../algorithms.md) — PPO, SAC, DDPG, TD3, A2C, and A3C configuration and hyperparameters
- [Runners & Training Loop](runners.md) — synchronous and asynchronous runners
- [Replay Buffers](buffers.md) — the CPU `ReplayBuffer` and the GPU replay buffer

## Quick start

```bash
# Standard SAC
srl-train --config configs/envs/halfcheetah_sac.yaml --device cuda

# PPO on Isaac Lab
srl-train --config configs/envs/isaaclab_ant_ppo.yaml --device cuda --n-envs 4096
```
