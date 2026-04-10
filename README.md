# SRL — Simple Reinforcement Learning

**SRL** is a modular, PyTorch-based reinforcement learning library for continuous control. Train agents in Gymnasium, Isaac Lab, or custom environments using YAML configs.

## Features

- **7 algorithms**: PPO, SAC, DDPG, A2C, A3C, TD3, DQN
- **YAML-driven**: Define network architectures without Python code
- **Multi-modal**: Support state, pixels, lidar, text inputs
- **Environments**: Gymnasium, Isaac Lab, racecar_gym
- **CLI**: `srl-train --config ... --env ... --steps ...`
- **Goal-conditioned RL**: Fetch robotics via `GoalEnvWrapper`

## Quick Install

```bash
pip install srl-rl

# With MuJoCo
pip install "srl-rl[mujoco]"

# With all environments
pip install "srl-rl[all]"

# Or from GitHub for latest
pip install git+https://github.com/Bigkatoan/SRL.git
```

## Quick Start

```bash
# Train PPO on Pendulum for 100k steps
srl-train --config configs/envs/pendulum_ppo.yaml \
          --env Pendulum-v1 \
          --steps 100000 \
          --device cuda

# Or train SAC on Fetch  
srl-train --config configs/envs/fetch_reach_sac.yaml \
          --env FetchReach-v4 \
          --steps 500000
```

## YAML Config Example

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 3
    latent_dim: 64
    layers:
      - {out_features: 64, activation: relu}
      - {out_features: 64, activation: relu}

actor:
  name: actor
  type: gaussian
  action_dim: 1

critic:
  name: critic
  type: value

losses:
  - {name: policy, weight: 1.0}
  - {name: value, weight: 0.5}

train:
  total_steps: 100_000
  lr: 3e-4
  batch_size: 128
```

## Supported Environments

| Suite | Envs | Wrapper |
|-------|------|---------|
| **Gymnasium** | Pendulum, HalfCheetah, Hopper, etc. | `GymnasiumWrapper` |
| **Box2D** | BipedalWalker, LunarLander, CarRacing | `GymnasiumWrapper` |
| **Robotics** | FetchReach, FetchPush, FetchPickAndPlace, FetchSlide | `GoalEnvWrapper` |
| **Isaac Lab** | Isaac-Cartpole, Isaac-Ant, Isaac-Humanoid | `IsaacLabWrapper` |
| **racecar_gym** | SingleAgentAustria, SingleAgentBerlin | `RacecarWrapper` |

## Python API Example

```python
import gymnasium as gym
import torch
from srl.registry.builder import ModelBuilder
from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper

# Build model from YAML
model = ModelBuilder.from_yaml("configs/envs/pendulum_ppo.yaml")

# Create environment
env = GymnasiumWrapper(gym.make("Pendulum-v1"))

# Create agent
cfg = PPOConfig(lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10)
agent = PPO(model, config=cfg, device="cuda")

# Training loop
obs, _ = env.reset()
for step in range(10000):
    obs_t = {k: torch.from_numpy(v).float().cuda() for k, v in obs.items()}
    action, log_prob, value, _ = agent.predict(obs_t)
    next_obs, reward, done, trunc, _ = env.step(action.cpu().numpy())
    agent.buffer.add(obs=obs, action=action.cpu().numpy(),
                     reward=reward, done=done, log_prob=log_prob.cpu().numpy(),
                     value=value.cpu().numpy())
    if done or trunc:
        obs, _ = env.reset()
    else:
        obs = next_obs
```

## Documentation

- 📖 [Full Docs](https://bigkatoan.github.io/SRL)
- 🚀 [Installation Guide](https://bigkatoan.github.io/SRL/installation)
- 💡 [Quickstart](https://bigkatoan.github.io/SRL/quickstart)
- 🏗️ [Environments](https://bigkatoan.github.io/SRL/environments/)
- ⚙️ [Config Reference](https://bigkatoan.github.io/SRL/config_reference)
- 🔌 [API Reference](https://bigkatoan.github.io/SRL/api/)

## Testing

Run environment smoke tests:

```bash
pytest tests/test_configs_envs.py -v
pytest tests/test_cli_smoke.py -v
```

## License

MIT — See [LICENSE](LICENSE)
# SRL — Simple Reinforcement Learning Library

A lightweight, config-driven reinforcement learning library for continuous action spaces. Train from scratch with PyTorch, deploy on Gymnasium environments, Isaac Lab, or ROS2 robots — all from a single YAML file.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [YAML Config System](#yaml-config-system)
  - [Encoders](#encoders)
  - [Flow DSL](#flow-dsl)
  - [Actor Heads](#actor-heads)
  - [Critic Heads](#critic-heads)
  - [Loss Registry](#loss-registry)
  - [Full Config Examples](#full-config-examples)
- [Algorithms](#algorithms)
  - [PPO](#ppo)
  - [SAC](#sac)
  - [DDPG](#ddpg)
  - [A2C](#a2c)
  - [A3C](#a3c)
- [Multi-Environment Training](#multi-environment-training)
- [Vision Inputs](#vision-inputs)
- [Recurrent Policies (LSTM)](#recurrent-policies-lstm)
- [Frame Stacking](#frame-stacking)
- [Text / Language Inputs](#text--language-inputs)
- [Multi-Modal Inputs](#multi-modal-inputs)
- [Self-Supervised Representation Learning](#self-supervised-representation-learning)
- [DL Techniques in YAML](#dl-techniques-in-yaml)
- [Checkpointing & Loading](#checkpointing--loading)
- [Logging](#logging)
- [Callbacks](#callbacks)
- [ROS2 Deployment](#ros2-deployment)
- [Isaac Lab Integration](#isaac-lab-integration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)

---

## Features

- **Config-driven model builder** — define full network architectures in YAML; no Python code changes needed
- **DAG flow DSL** — route encoder outputs to heads with `"A -> B"` strings; support branching and multi-input
- **Algorithms** — PPO, SAC, DDPG, A2C, A3C (all continuous action spaces)
- **Encoder types** — MLP, CNN, LSTM (recurrent wrapper), CharCNN text encoder, frame stacking, momentum encoder
- **Self-supervised aux losses** — Contrastive (CURL/InfoNCE), AutoEncoder, BYOL — trained from scratch, no pretrained weights
- **All DL techniques in YAML** — BatchNorm, LayerNorm, RMSNorm, Dropout, DropPath, spectral norm, residual connections, weight init, pre/post norm order
- **Multi-environment training** — `SyncVectorEnv` and `AsyncVectorEnv` wrappers
- **Gymnasium compatible** — standard `gym.make()` environments
- **Isaac Lab compatible** — thin wrapper for `ManagerBasedRLEnv`
- **ROS2 deployment** — subscribe to sensor topics, publish actions at fixed Hz
- **Checkpointing** — `safetensors`-first with `.pt` fallback, automatic rotation
- **TensorBoard logging** — built-in `Logger` wrapper

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended for training)
- Gymnasium 0.29+

```bash
git clone <repo-url> && cd SRL
python -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

Install optional extras individually:

```bash
pip install safetensors        # recommended for checkpointing
pip install tensorboard        # for TensorBoard logging
pip install opencv-python      # for visual augmentations (CURL/DRQ)
```

For ROS2 inference, source your ROS2 workspace before launching.

---

## Quick Start

### 1. Define a config (or use one from `configs/`)

```yaml
# configs/ppo_state.yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
      - {out_features: 256, activation: tanh}
      - {out_features: 256, activation: tanh}

flows:
  - "state_enc -> actor"
  - "state_enc -> critic"

actor:
  name: actor
  type: gaussian
  action_dim: 6

critic:
  name: critic
  type: value
  layers: []
```

### 2. Build the model and train with PPO

```python
import gymnasium as gym
import torch
from srl.registry.builder import ModelBuilder
from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.envs.sync_vector_env import SyncVectorEnv

# Build model from YAML
model = ModelBuilder.from_yaml("configs/ppo_state.yaml")

# Create vectorised envs
env = SyncVectorEnv([lambda: GymnasiumWrapper(gym.make("HalfCheetah-v4"))] * 4)

# Configure and create agent
cfg = PPOConfig(n_steps=2048, num_envs=4, batch_size=256, n_epochs=10)
agent = PPO(model, config=cfg, device="cuda")

# Training loop
obs, _ = env.reset()
for _ in range(500):  # 500 rollouts
    for _ in range(cfg.n_steps):
        obs_t = {k: torch.from_numpy(v).float().cuda() for k, v in obs.items()}
        action, log_prob, value, _ = agent.predict(obs_t)
        next_obs, reward, done, trunc, _ = env.step(action.cpu().numpy())
        agent.buffer.add(obs=obs, action=action.cpu().numpy(),
                         reward=reward, done=done, log_prob=log_prob.cpu().numpy(),
                         value=value.cpu().numpy())
        obs = next_obs
    last_obs_t = {k: torch.from_numpy(v).float().cuda() for k, v in obs.items()}
    _, _, last_val, _ = agent.predict(last_obs_t)
    agent.buffer.compute_returns_and_advantages(last_value=last_val.cpu().numpy())
    metrics = agent.update()
    print(metrics)
```

See `examples/train_ppo_halfcheetah.py` and `examples/train_sac_ant.py` for complete training scripts.

---

## Architecture Overview

```
YAML Config
    │
    ▼
ModelBuilder.from_yaml()
    │
    ├── EncoderConfig  ──► Encoder (MLP / CNN / LSTM / Text / …)
    │                          │
    │                     FlowGraph (DAG)
    │                          │
    ├── HeadConfig     ──► Actor Head  ─── GaussianActorHead
    │                      │              SquashedGaussianActorHead
    │                      │              DeterministicActorHead
    │                      │
    │                  ──► Critic Head ─── ValueHead
    │                                      QFunctionHead
    │                                      TwinQHead
    │
    └── AgentModel (nn.Module)
            │
            ▼
        Algorithm (PPO / SAC / DDPG / A2C / A3C)
```

The `AgentModel` is a pure `nn.Module` — algorithms hold a reference to it and call `.forward()`. This separation means you can swap algorithms without changing your network definition.

---

## YAML Config System

Everything is declared in a single YAML file. Use `ModelBuilder.from_yaml(path)` or `ModelBuilder.from_dict(cfg_dict)`.

### Encoders

Each encoder has a unique `name` used in the flow DSL.

#### MLP Encoder

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 27          # observation dimension
    latent_dim: 256        # output embedding dimension
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}
      - {out_features: 256, activation: relu, norm: layer_norm}
```

#### CNN Encoder

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]   # [C, H, W]
    latent_dim: 256
    layers:
      - [32, 8, 4, relu]   # [out_channels, kernel_size, stride, activation]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

#### LSTM Encoder

Wraps any base encoder (MLP or CNN) with a single-layer LSTM. Hidden states are tracked automatically across rollout steps.

```yaml
encoders:
  - name: state_enc
    type: lstm
    input_dim: 27
    latent_dim: 128
    lstm_hidden: 256
    layers:
      - {out_features: 128, activation: relu}
```

#### Text Encoder (CharCNN)

Encodes variable-length text strings character-by-character with a small CNN. Useful for language-conditioned tasks.

```yaml
encoders:
  - name: lang_enc
    type: text
    latent_dim: 64
```

Input: `torch.LongTensor` of shape `(B, seq_len)` — character indices 0–127.

### Flow DSL

Flows define how encoder outputs are routed to heads. The syntax is `"source -> target"`. Use multiple flow lines to merge multiple encoders into one head.

```yaml
flows:
  - "visual_enc -> actor"   # visual → actor
  - "state_enc  -> actor"   # state + visual are concatenated → actor
  - "visual_enc -> critic"
  - "state_enc  -> critic"
```

- Latents from multiple sources are **concatenated** on the feature dimension.
- The graph is a DAG; cycles are detected and raise an error at build time.
- If no flows are defined, all encoder outputs are concatenated and sent to both heads.

### Actor Heads

#### `gaussian` — Diagonal Gaussian (PPO / A2C)

Outputs a mean and a learnable log-std per dimension. Samples from `N(μ, σ²)`.

```yaml
actor:
  name: actor
  type: gaussian
  action_dim: 6
  layers: []            # optional extra MLP layers before output
  log_std_init: -1.0    # initial log standard deviation
```

#### `squashed_gaussian` — Squashed Gaussian (SAC)

Same as Gaussian but output is squashed through `tanh` to `[-1, 1]`. Includes the log-determinant correction for the change-of-variables in log-prob.

```yaml
actor:
  name: actor
  type: squashed_gaussian
  action_dim: 8
  layers: [{out_features: 256, activation: relu}]
  log_std_min: -5.0
  log_std_max: 2.0
```

#### `deterministic` — Deterministic (DDPG)

Outputs `tanh(μ(s))` directly. Used with OU noise added at inference time.

```yaml
actor:
  name: actor
  type: deterministic
  action_dim: 4
  layers: [{out_features: 256, activation: relu}]
```

### Critic Heads

#### `value` — State-value V(s) (PPO / A2C)

```yaml
critic:
  name: critic
  type: value
  layers: []
```

#### `twin_q` — Twin Q-networks (SAC)

Two independent Q(s, a) networks; returns the minimum to reduce overestimation.

```yaml
critic:
  name: critic
  type: twin_q
  action_dim: 8
  layers: [{out_features: 256, activation: relu}]
```

#### `q_function` — Single Q(s, a) (DDPG)

```yaml
critic:
  name: critic
  type: q_function
  action_dim: 4
  layers: [{out_features: 256, activation: relu}]
```

### Loss Registry

Losses are declared as a list. The `LossComposer` weights and schedules them at runtime.

```yaml
losses:
  - name: policy
    weight: 1.0
  - name: value
    weight: 0.5
  - name: entropy
    weight: 0.01
  - name: reconstruction
    weight: 0.1
    schedule: cosine      # "none" | "linear" | "cosine"
    total_steps: 2000000
```

Available loss names: `policy`, `value`, `entropy`, `sac_q`, `sac_policy`, `sac_temperature`, `contrastive`, `reconstruction`, `byol`.

### Full Config Examples

| File | Algorithm | Encoder | Notes |
|------|-----------|---------|-------|
| `configs/ppo_state.yaml` | PPO | MLP | HalfCheetah-v4 |
| `configs/ppo_visual.yaml` | PPO | CNN + AE | Visual env with autoencoder aux loss |
| `configs/sac_state.yaml` | SAC | MLP + LayerNorm | Ant-v4 |
| `configs/sac_multimodal.yaml` | SAC | CNN + MLP + Text | Visual + state + language |

---

## Algorithms

All algorithms implement a common interface:

```python
action, log_prob, value, hidden = agent.predict(obs_dict, hidden=None, deterministic=False)
metrics = agent.update()   # returns dict of loss scalars
agent.save("checkpoint.pt")
agent.load("checkpoint.pt")
```

### PPO

On-policy, supports `SyncVectorEnv` / `AsyncVectorEnv`.

```python
from srl.algorithms.ppo import PPO
from srl.core.config import PPOConfig

cfg = PPOConfig(
    lr=3e-4,
    n_steps=2048,
    num_envs=4,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    vf_coef=0.5,
    entropy_coef=0.0,
    max_grad_norm=0.5,
)
agent = PPO(model, config=cfg, device="cuda")
```

**Key config fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `lr` | `3e-4` | Adam learning rate |
| `n_steps` | `2048` | Steps per env per rollout |
| `num_envs` | `1` | Number of parallel environments |
| `batch_size` | `64` | Mini-batch size |
| `n_epochs` | `10` | Gradient epochs per rollout |
| `clip_range` | `0.2` | PPO clipping parameter ε |
| `vf_coef` | `0.5` | Value loss coefficient |
| `entropy_coef` | `0.0` | Entropy bonus coefficient |
| `gae_lambda` | `0.95` | GAE λ |
| `max_grad_norm` | `0.5` | Gradient clipping |

### SAC

Off-policy with automatic entropy tuning. Requires a `SquashedGaussian` actor and `TwinQ` critic.

```python
from srl.algorithms.sac import SAC
from srl.core.config import SACConfig
import copy

model = ModelBuilder.from_yaml("configs/sac_state.yaml")
target = copy.deepcopy(model)

cfg = SACConfig(
    lr_actor=3e-4,
    lr_critic=3e-4,
    lr_alpha=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    action_dim=8,          # for automatic target entropy: H* = -action_dim
)
agent = SAC(model, target, config=cfg, device="cuda")
```

**Key config fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `lr_actor` | `3e-4` | Actor learning rate |
| `lr_critic` | `3e-4` | Critic learning rate |
| `lr_alpha` | `3e-4` | Temperature learning rate |
| `buffer_size` | `1_000_000` | Replay buffer capacity |
| `tau` | `0.005` | Soft target update coefficient |
| `action_dim` | `0` | Action dimension (for auto entropy target) |
| `init_alpha` | `0.2` | Initial temperature |

### DDPG

Off-policy deterministic with Ornstein-Uhlenbeck exploration noise.

```python
from srl.algorithms.ddpg import DDPG
from srl.core.config import DDPGConfig
import copy

model = ModelBuilder.from_yaml("configs/ddpg.yaml")
target = copy.deepcopy(model)

cfg = DDPGConfig(lr_actor=1e-4, lr_critic=1e-3, action_dim=4)
agent = DDPG(model, target, config=cfg, device="cuda")
```

### A2C

Synchronous on-policy; simpler and faster than PPO for single-env tasks.

```python
from srl.algorithms.a2c import A2C
from srl.core.config import A2CConfig

cfg = A2CConfig(lr=7e-4, n_steps=5, vf_coef=0.25, entropy_coef=0.01)
agent = A2C(model, config=cfg, device="cpu")
```

### A3C

Asynchronous multi-process on-policy. Workers collect trajectories in parallel and push gradients to a shared model.

```python
from srl.algorithms.a3c import A3C
from srl.core.config import A3CConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
import gymnasium as gym

model = ModelBuilder.from_yaml("configs/ppo_state.yaml")
model.share_memory()

agent = A3C(model, config=A3CConfig(lr=1e-4, n_workers=8), device="cpu")
agent.train(
    total_timesteps=5_000_000,
    env_fn=lambda: GymnasiumWrapper(gym.make("HalfCheetah-v4")),
)
```

> **Note:** A3C uses `multiprocessing` with the `spawn` start method. Ensure all code is inside `if __name__ == "__main__":` guards.

---

## Multi-Environment Training

```python
from srl.envs.sync_vector_env import SyncVectorEnv
from srl.envs.async_vector_env import AsyncVectorEnv
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
import gymnasium as gym

# Synchronous (same process, zero overhead)
env = SyncVectorEnv([lambda: GymnasiumWrapper(gym.make("HalfCheetah-v4"))] * 4)

# Asynchronous (subprocess per env, scales to 16+ envs)
env = AsyncVectorEnv([lambda: GymnasiumWrapper(gym.make("HalfCheetah-v4"))] * 8)

obs, _ = env.reset()            # obs: dict[str, np.ndarray(n_envs, ...)]
obs, rew, done, trunc, info = env.step(actions)   # actions: np.ndarray(n_envs, action_dim)
```

Both wrappers return observations as `dict[str, np.ndarray]` keyed by encoder name.

---

## Vision Inputs

SRL trains visual encoders from scratch. No pretrained weights are used.

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

### Visual Augmentations

Enable data augmentation for sample efficiency (DRQ / CURL style):

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    augmentation: random_shift   # "random_shift" | "color_jitter" | "random_crop"
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

---

## Recurrent Policies (LSTM)

Wrap any encoder with LSTM for partially observable environments.

```yaml
encoders:
  - name: state_enc
    type: lstm
    input_dim: 27
    latent_dim: 128
    lstm_hidden: 256
    layers:
      - {out_features: 128, activation: relu}
```

Hidden states are tracked in the `hidden` dict returned by `agent.predict()`:

```python
hidden = None
for step in range(episode_length):
    action, log_prob, value, hidden = agent.predict(obs_t, hidden=hidden)
    # Pass `hidden` back on the next step
```

When an episode ends, reset the hidden state for that environment index.

---

## Frame Stacking

Stack the last N frames as additional input channels:

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]
    frame_stack: 4              # input becomes [12, 84, 84]
    latent_dim: 256
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]
```

The `FrameStackWrapper` is applied automatically at build time when `frame_stack > 1`.

---

## Text / Language Inputs

Use the `CharCNN` text encoder for language-conditioned policies:

```yaml
encoders:
  - name: lang_enc
    type: text
    latent_dim: 64
```

Pass text as a character-index tensor:

```python
import torch
text = "pick up the red cube"
chars = torch.tensor([ord(c) for c in text], dtype=torch.long).unsqueeze(0)  # (1, seq)
obs = {"lang_enc": chars, "state_enc": state_tensor}
action, *_ = agent.predict(obs)
```

---

## Multi-Modal Inputs

Combine any number of encoders and route them through the flow graph:

```yaml
# configs/sac_multimodal.yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 64, 64]
    latent_dim: 128
  - name: state_enc
    type: mlp
    input_dim: 18
    latent_dim: 64
  - name: lang_enc
    type: text
    latent_dim: 64

flows:
  - "visual_enc -> actor"
  - "state_enc  -> actor"
  - "lang_enc   -> actor"
  - "visual_enc -> critic"
  - "state_enc  -> critic"

actor:
  name: actor
  type: squashed_gaussian
  action_dim: 6
  layers: [{out_features: 256, activation: relu}]

critic:
  name: critic
  type: twin_q
  action_dim: 6
```

At runtime pass a dict with all encoder inputs:

```python
obs = {
    "visual_enc": image_tensor,   # (B, 3, 64, 64) uint8 or float32
    "state_enc":  state_tensor,   # (B, 18)
    "lang_enc":   text_tensor,    # (B, seq_len) long
}
action, *_ = agent.predict(obs)
```

---

## Self-Supervised Representation Learning

Auxiliary losses improve sample efficiency for vision-based tasks. They are trained from scratch — no pretrained backbone is used.

### Contrastive Learning (CURL / InfoNCE)

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 128
    aux_type: contrastive
    aux_latent_dim: 64
    use_momentum: true
    momentum_tau: 0.99
    layers:
      - [32, 8, 4, relu]
      - [64, 4, 2, relu]
      - [64, 3, 1, relu]

losses:
  - name: contrastive
    weight: 0.5
```

### AutoEncoder

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    aux_type: autoencoder

losses:
  - name: reconstruction
    weight: 0.1
    schedule: cosine
    total_steps: 2000000
```

### BYOL

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 128
    aux_type: byol
    use_momentum: true
    momentum_tau: 0.99

losses:
  - name: byol
    weight: 0.5
```

---

## DL Techniques in YAML

All standard deep learning techniques are configurable per-layer inside the `layers` list.

### Layer Config Fields

| Key | Values | Description |
|-----|--------|-------------|
| `out_features` | int | MLP hidden size |
| `out_channels` | int | CNN output channels |
| `kernel_size` | int | CNN kernel size |
| `stride` | int | CNN stride |
| `activation` | `relu` `tanh` `elu` `gelu` `silu` `mish` `leaky_relu` `none` | Activation function |
| `norm` | `batch_norm` `layer_norm` `group_norm` `rms_norm` `spectral_norm` `none` | Normalisation |
| `dropout` | float 0–1 | Dropout probability |
| `dropout_type` | `standard` `alpha` `droppath` | Dropout variant |
| `residual` | bool | Add residual skip connection |
| `depthwise` | bool | Depthwise separable convolution (CNN only) |
| `norm_order` | `pre` `post` | Apply norm before or after activation |
| `weight_init` | `orthogonal` `xavier_uniform` `xavier_normal` `kaiming_uniform` `kaiming_normal` `none` | Weight initialisation scheme |
| `pooling` | `max` `avg` `adaptive_avg` `none` | Pooling (CNN only) |

### Example: PPO with LayerNorm + Orthogonal Init

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 27
    latent_dim: 256
    layers:
      - out_features: 256
        activation: tanh
        norm: layer_norm
        norm_order: post
        weight_init: orthogonal
        dropout: 0.0
      - out_features: 256
        activation: tanh
        norm: layer_norm
        weight_init: orthogonal
```

### Example: CNN with BatchNorm + Residual

```yaml
encoders:
  - name: visual_enc
    type: cnn
    input_shape: [3, 84, 84]
    latent_dim: 256
    layers:
      - out_channels: 32
        kernel_size: 8
        stride: 4
        activation: relu
        norm: batch_norm
        norm_order: pre
      - out_channels: 64
        kernel_size: 4
        stride: 2
        activation: relu
        norm: batch_norm
        residual: true
      - out_channels: 64
        kernel_size: 3
        stride: 1
        activation: relu
        norm: batch_norm
```

---

## Checkpointing & Loading

```python
from srl.utils.checkpoint import CheckpointManager

cm = CheckpointManager(save_dir="checkpoints/my_run", max_keep=5)

# Save (uses safetensors when available, falls back to .pt)
cm.save(agent.model, optimizer=agent.optimizer, step=100_000, metrics={"reward": 1234.5})

# Load
cm.load(agent.model, optimizer=agent.optimizer, path="checkpoints/my_run/ckpt_0100000000.safetensors")
```

Algorithms also expose `save()` / `load()` directly:

```python
agent.save("my_checkpoint.pt")
agent.load("my_checkpoint.pt")
```

---

## Logging

```python
from srl.utils.logger import Logger

logger = Logger(log_dir="runs/ppo_halfcheetah", verbose=True)

logger.log("train/reward", 123.4, step=1000)
logger.log_dict({"loss/policy": 0.1, "loss/value": 0.5}, step=1000)

logger.close()
```

View TensorBoard logs:

```bash
tensorboard --logdir runs/
```

---

## Callbacks

```python
from srl.utils.callbacks import LogCallback, CheckpointCallback, EarlyStopping
from srl.utils.checkpoint import CheckpointManager
from srl.utils.logger import Logger

logger = Logger("runs/my_run")
cm = CheckpointManager("checkpoints/my_run")

callbacks = [
    LogCallback(logger, log_interval=2048),
    CheckpointCallback(cm, save_interval=50_000),
    EarlyStopping(monitor="eval/mean_reward", patience=20, mode="max"),
]

# In your training loop:
for cb in callbacks:
    cb.on_step_end(step, metrics)

early_stop = next(c for c in callbacks if isinstance(c, EarlyStopping))
if early_stop.should_stop:
    break
```

---

## ROS2 Deployment

SRL provides a ROS2 inference node that subscribes to sensor topics, runs the policy, and publishes actions at a configurable frequency.

### Launch

```bash
# Source ROS2 first
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run srl rl_inference_node --ros-args -p config_path:=configs/sac_multimodal.yaml
```

Or use the launch file:

```bash
ros2 launch srl rl_agent.launch.py \
    config_path:=configs/sac_multimodal.yaml \
    checkpoint_path:=checkpoints/sac_ant/ckpt_best.safetensors
```

### Programmatic usage

```python
from srl.ros2.rl_node import RLInferenceNode
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
import rclpy

rclpy.init()

model = ModelBuilder.from_yaml("configs/sac_multimodal.yaml")
model.load_state_dict(torch.load("checkpoint.pt")["model"])

node = RLInferenceNode(
    model=model,
    obs_topics={
        "visual_enc": "/camera/image_raw",
        "state_enc":  "/robot/joint_states",
    },
    action_topic="/robot/cmd_vel",
    action_msg_type=Float32MultiArray,
    obs_msg_type=Image,
    hz=20.0,
    device="cuda",
)

from rclpy.executors import MultiThreadedExecutor
executor = MultiThreadedExecutor()
executor.add_node(node)
executor.spin()
```

### Topic format

- **Observation topics**: one topic per encoder, keyed by encoder name. Messages are converted to `torch.Tensor` automatically.
- **Action topic**: `Float32MultiArray` — `data` field holds the flat action vector.

---

## Isaac Lab Integration

```python
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

# Inside an IsaacLab task script:
from omni.isaac.lab.envs import ManagerBasedRLEnv

env = ManagerBasedRLEnv(cfg=my_task_cfg)
wrapped = IsaacLabWrapper(env)

obs, _ = wrapped.reset()   # obs: dict[str, np.ndarray]
obs, rew, done, trunc, info = wrapped.step(actions)
```

The wrapper converts Isaac Lab's GPU tensors to CPU numpy arrays, which the SRL collector expects.

---

## API Reference

### `ModelBuilder`

```python
from srl.registry.builder import ModelBuilder

model = ModelBuilder.from_yaml("path/to/config.yaml")
model = ModelBuilder.from_dict({"encoders": [...], "flows": [...], "actor": {...}, "critic": {...}})
```

### `AgentModel.forward()`

```python
result = model(obs_dict, hidden_states=None, action=None)
# result keys:
#   "latents"    — dict[str, Tensor]: encoder outputs
#   "actor_out"  — dict: {"action", "log_prob", "mean"} (head-dependent)
#   "value"      — Tensor or tuple[Tensor, Tensor] (twin_q)
#   "new_hidden" — dict: updated hidden states for LSTM encoders
```

### `RolloutBuffer`

```python
from srl.core.rollout_buffer import RolloutBuffer

buf = RolloutBuffer(capacity=2048, num_envs=4, gamma=0.99, lam=0.95)
buf.add(obs, action, reward, done, log_prob, value)
buf.compute_returns_and_advantages(last_value=last_val)

for mini_batch in buf.get_batches(batch_size=256):
    # mini_batch.obs, .actions, .log_probs, .advantages, .returns
    pass

buf.reset()
```

### `ReplayBuffer`

```python
from srl.core.replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=1_000_000)
buf.add(obs, action, reward, done, next_obs=next_obs)

batch = buf.sample(256)
# batch.obs, .next_obs, .actions, .rewards, .dones
```

### `FlowGraph`

```python
from srl.registry.flow_graph import FlowGraph

graph = FlowGraph(["enc_a -> actor", "enc_b -> actor", "enc_a -> critic"])
order = graph.execution_order        # ["enc_a", "enc_b", "actor", "critic"]
inputs = graph.get_inputs("actor")   # ["enc_a", "enc_b"]
```

---

## Project Structure

```
SRL/
├── configs/                        # Ready-to-use YAML configs
│   ├── ppo_state.yaml
│   ├── ppo_visual.yaml
│   ├── sac_state.yaml
│   └── sac_multimodal.yaml
│
├── examples/                       # End-to-end training scripts
│   ├── train_ppo_halfcheetah.py
│   └── train_sac_ant.py
│
├── srl/
│   ├── core/
│   │   ├── base_agent.py           # Abstract BaseAgent interface
│   │   ├── base_policy.py          # Abstract BasePolicy
│   │   ├── config.py               # PPOConfig, SACConfig, DDPGConfig, A2CConfig, A3CConfig
│   │   ├── rollout_buffer.py       # On-policy buffer (lazy-init, GAE)
│   │   ├── replay_buffer.py        # Off-policy circular buffer
│   │   ├── prioritized_replay_buffer.py  # PER buffer
│   │   └── her_replay_buffer.py    # HER buffer
│   │
│   ├── networks/
│   │   ├── agent_model.py          # AgentModel (nn.Module, DAG forward pass)
│   │   ├── distributions.py        # DiagonalGaussian, SquashedGaussian
│   │   ├── encoders/
│   │   │   ├── mlp_encoder.py
│   │   │   ├── cnn_encoder.py
│   │   │   ├── recurrent.py        # LSTMEncoder wrapper
│   │   │   ├── momentum_encoder.py # EMA / momentum encoder
│   │   │   ├── frame_stack.py
│   │   │   ├── text_encoder.py     # CharCNN
│   │   │   └── augmentations.py    # Random shift, crop, colour jitter
│   │   ├── heads/
│   │   │   ├── actor_head.py       # Gaussian, SquashedGaussian, Deterministic
│   │   │   ├── critic_head.py      # Value, QFunction, TwinQ
│   │   │   └── aux_head.py         # Projection / prediction heads for aux losses
│   │   ├── layers/
│   │   │   ├── activations.py
│   │   │   ├── norms.py            # BN, LN, GN, RMSNorm, SpectralNorm
│   │   │   ├── dropout.py
│   │   │   ├── pooling.py
│   │   │   ├── init.py             # Weight init strategies
│   │   │   ├── mlp_builder.py      # build_mlp() from layer config list
│   │   │   └── cnn_builder.py      # build_cnn() from layer config list
│   │   └── representation/
│   │       ├── contrastive.py      # InfoNCE / CURL
│   │       └── autoencoder.py      # Convolutional AE
│   │
│   ├── registry/
│   │   ├── registry.py             # EncoderRegistry, HeadRegistry, LossRegistry
│   │   ├── flow_graph.py           # DAG parser + topological sort
│   │   ├── config_schema.py        # EncoderConfig, HeadConfig, AgentModelConfig
│   │   └── builder.py              # ModelBuilder.from_yaml / from_dict
│   │
│   ├── algorithms/
│   │   ├── ppo.py
│   │   ├── sac.py
│   │   ├── ddpg.py
│   │   ├── a2c.py
│   │   └── a3c.py
│   │
│   ├── losses/
│   │   ├── rl_losses.py            # PPO, SAC, DDPG, A2C loss functions
│   │   ├── aux_losses.py           # InfoNCE, reconstruction, BYOL
│   │   └── loss_composer.py        # Weighted sum + scheduling
│   │
│   ├── envs/
│   │   ├── gymnasium_wrapper.py    # obs → dict[str, ndarray]
│   │   ├── sync_vector_env.py      # SyncVectorEnv
│   │   ├── async_vector_env.py     # AsyncVectorEnv
│   │   ├── isaac_lab_wrapper.py    # Isaac Lab GPU env adapter
│   │   └── collector.py            # Trajectory collector helper
│   │
│   ├── utils/
│   │   ├── logger.py               # TensorBoard + stdout logger
│   │   ├── normalizer.py           # RunningMeanStd normaliser
│   │   ├── gae.py                  # Standalone GAE computation
│   │   ├── checkpoint.py           # CheckpointManager (safetensors / .pt)
│   │   └── callbacks.py            # LogCallback, CheckpointCallback, EarlyStopping
│   │
│   └── ros2/
│       ├── rl_node.py              # RLInferenceNode
│       └── launch/
│           └── rl_agent.launch.py
│
├── tests/
│   ├── conftest.py
│   ├── test_core.py                # Unit tests: buffers, layers, encoders, losses
│   └── test_integration.py        # Integration tests: builder, forward, PPO predict
│
├── pyproject.toml
└── README.md
```

---

## License

MIT License. See [LICENSE](LICENSE).
