# Algorithms

SRL implements five RL algorithms for **continuous action spaces**.

---

## Overview

| Algorithm | Type | Strengths | Best for |
|---|---|---|---|
| **PPO** | On-policy | Sample-efficient, stable | Locomotion, Isaac Lab |
| **SAC** | Off-policy | High performance, auto-entropy | High-DoF manipulation |
| **DDPG** | Off-policy | Simple, deterministic | Low-DoF tasks |
| **A2C** | On-policy | Low memory, fast updates | Parallel envs |
| **A3C** | On-policy, parallel | Asynchronous workers | Multi-CPU setups |

---

## PPO — Proximal Policy Optimization

PPO clips the surrogate objective to prevent large policy updates.

### Config

```python
from srl.core.config import PPOConfig

cfg = PPOConfig(
    lr           = 3e-4,
    n_steps      = 2048,   # steps per env per rollout
    num_envs     = 8,
    batch_size   = 256,
    n_epochs     = 10,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    clip_range   = 0.2,
    entropy_coef = 0.0,
    vf_coef      = 0.5,
    max_grad_norm= 0.5,
)
```

### Recommended environments

- `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Humanoid-v5`
- Isaac Lab: `Isaac-Ant-v0`, `Isaac-Humanoid-v0`

---

## SAC — Soft Actor-Critic

SAC maximizes return + policy entropy.  
Twin-Q critics + automatic temperature tuning.

### Config

```python
from srl.core.config import SACConfig

cfg = SACConfig(
    lr_actor        = 3e-4,
    lr_critic       = 3e-4,
    lr_alpha        = 3e-4,
    buffer_size     = 1_000_000,
    batch_size      = 256,
    gamma           = 0.99,
    tau             = 0.005,
    action_dim      = 6,      # required for target entropy
    learning_starts = 10_000,
    gradient_steps  = 1,
    auto_entropy_tuning = True,
)
```

### Recommended environments

- `HalfCheetah-v5`, `Ant-v5`, `Swimmer-v5`, `Pusher-v5`, `Reacher-v5`
- `FetchReach-v4`, `FetchPush-v4`, `FetchPickAndPlace-v4`, `FetchSlide-v4`

---

## DDPG — Deep Deterministic Policy Gradient

Deterministic off-policy actor-critic.  Simpler than SAC but more sensitive to
hyperparameters.

```python
from srl.core.config import DDPGConfig

cfg = DDPGConfig(
    lr_actor     = 1e-4,
    lr_critic    = 1e-3,
    buffer_size  = 1_000_000,
    batch_size   = 256,
    gamma        = 0.99,
    tau          = 0.005,
    action_dim   = 6,
    action_noise = "gaussian",
    noise_sigma  = 0.1,
)
```

---

## A2C — Advantage Actor-Critic

Synchronous on-policy algorithm. Lower memory than PPO.

```python
from srl.core.config import A2CConfig

cfg = A2CConfig(
    lr            = 7e-4,
    n_steps       = 5,
    gamma         = 0.99,
    entropy_coef  = 0.01,
    vf_coef       = 0.25,
    max_grad_norm = 0.5,
)
```

---

## A3C — Asynchronous Advantage Actor-Critic

Runs `n_workers` parallel CPU workers, each collecting experience and computing
gradients asynchronously.

```python
from srl.core.config import A3CConfig

cfg = A3CConfig(
    lr          = 1e-4,
    n_workers   = 4,
    n_steps     = 20,
    gamma       = 0.99,
    gae_lambda  = 1.0,
)
```
