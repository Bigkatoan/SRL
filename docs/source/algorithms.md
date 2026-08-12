# Algorithms

SRL implements six RL algorithms for **continuous action spaces**.

Algorithm configuration in SRL sits on top of the YAML model graph. The YAML file declares the model structure, routing, and currently supported loss terms; the algorithm layer consumes that graph and applies PPO, SAC, DDPG, TD3, A2C, or A3C-specific optimization logic on top.

Read the [YAML Core Guide](yaml_core.md) first if you want the architectural picture. This page focuses on the algorithm-side hyperparameters and runtime behavior.

---

## Overview

| Algorithm | Type | Strengths | Best for |
|---|---|---|---|
| **PPO** | On-policy | Sample-efficient, stable | Locomotion, Isaac Lab |
| **SAC** | Off-policy | High performance, auto-entropy | High-DoF manipulation |
| **DDPG** | Off-policy | Simple, deterministic | Low-DoF tasks |
| **TD3** | Off-policy | Reduced overestimation bias | Continuous control |
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
    encoder_update_freq = 1,  # encoder optimizer steps every N critic updates
)
```

### Encoder optimizer (v0.2.0)

SAC uses **three optimizers** since v0.2.0:

| Optimizer | Parameters | When it steps |
|---|---|---|
| `critic_optimizer` | Twin-Q head only | Every gradient step |
| `actor_optimizer` | Actor head only | Every gradient step |
| `encoder_optimizer` | All CNN/MLP encoders | Every `encoder_update_freq` critic steps |

The encoder is **never** updated by the actor backward pass, preventing the double
learning-rate that caused distribution shift in earlier versions.

### Visual SAC

Use `VisualSACConfig` for pixel-based tasks. It sets sensible defaults for CNN encoders:

```python
from srl.core.config import VisualSACConfig

cfg = VisualSACConfig(
    encoder_update_freq        = 2,      # step encoder every 2 critic updates
    encoder_optimize_with_critic = True, # False → encoder only trains via aux loss
    aux_loss_type              = "curl", # none | ae | vae | curl | byol | drq | spr | barlow
    lr_encoder                 = 1e-4,
)
```

**`aux_loss_type` options:**

| Value | Method | Notes |
|---|---|---|
| `none` | No auxiliary loss | Encoder trained by critic gradient only |
| `ae` | Autoencoder (MSE reconstruction) | Requires `ConvDecoderHead` |
| `vae` | Variational AE (MSE + KL) | Requires `VAEHead` + `ConvDecoderHead` |
| `curl` | CURL InfoNCE contrastive | Default; requires `ProjectionHead` |
| `byol` | BYOL self-prediction | Uses momentum encoder + `ProjectionHead` |
| `drq` | DrQ Q-value augmentation consistency | Applies random crop/colour augmentation |
| `spr` | SPR latent forward prediction | Requires `LatentTransitionModel` |
| `barlow` | Barlow Twins redundancy reduction | Requires `ProjectionHead` |

### Recommended environments

- `HalfCheetah-v5`, `Ant-v5`, `Swimmer-v5`, `Pusher-v5`, `Reacher-v5`
- `FetchReach-v4`, `FetchPush-v4`, `FetchPickAndPlace-v4`, `FetchSlide-v4`
- Pixel tasks: CarRacing-v2, DMControl suite, Isaac Lab visual tasks

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
    encoder_update_freq = 1,
)
```

---

## TD3 — Twin Delayed Deep Deterministic Policy Gradient

TD3 reduces Q-value overestimation with twin critics and delayed policy updates.

```python
from srl.core.config import TD3Config

cfg = TD3Config(
    lr_actor          = 3e-4,
    lr_critic         = 3e-4,
    buffer_size       = 1_000_000,
    batch_size        = 256,
    gamma             = 0.99,
    tau               = 0.005,
    action_dim        = 6,
    policy_noise      = 0.2,
    noise_clip        = 0.5,
    policy_delay      = 2,
    encoder_update_freq = 1,
)
```

Like SAC and DDPG, TD3 uses a separate `encoder_optimizer` since v0.2.0. The encoder
steps every `encoder_update_freq` critic updates, independent of `policy_delay`.

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

---

## Async off-policy runner (v0.2.0)

`AsyncOffPolicyRunner` decouples data collection from gradient updates for SAC, DDPG,
and TD3. The collector runs on the **main thread** (required for Isaac Lab CUDA-context
safety); the trainer runs on a **daemon thread** with its own CUDA stream.

```python
from srl.core.config import AsyncRunnerConfig
from srl.runners import AsyncOffPolicyRunner

runner_cfg = AsyncRunnerConfig(
    use_async      = True,    # enable async collector/trainer split
    use_gpu_buffer = True,    # swap replay buffer for GPUReplayBuffer
    prefill_steps  = 1000,    # random steps before first gradient update
    queue_maxsize  = 4,       # max transitions queued between threads
)
```

When `use_async=False` (default), the runner falls through to the standard synchronous
training loop. When `use_gpu_buffer=True` only, the GPU buffer is used but collection
and training remain on the same thread.

See [async_runner.md](async_runner.md) and [gpu_replay_buffer.md](gpu_replay_buffer.md)
for detailed usage.
