# Quick Start

This page walks you through the full workflow in ~30 lines of code.

---

## 1. Install

```bash
pip install "srl-rl[mujoco]"
```

---

## 2. Pick a config

SRL uses YAML to describe the neural network architecture:

```yaml
# configs/envs/halfcheetah_sac.yaml
algo: sac
encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}
      - {out_features: 256, activation: relu, norm: layer_norm}
flows:
  - "state_enc -> actor"
  - "state_enc -> critic"
actor:
  name: actor
  type: squashed_gaussian
  action_dim: 6
critic:
  name: critic
  type: twin_q
  action_dim: 6
```

---

## 3. Train with the CLI

```bash
srl-train --config configs/envs/halfcheetah_sac.yaml \
          --env HalfCheetah-v5 \
          --algo sac \
          --steps 1000000
```

---

## 4. Train via Python API

```python
import gymnasium as gym
import torch

from srl.algorithms.sac import SAC
from srl.core.config import SACConfig
from srl.envs.gymnasium_wrapper import GymnasiumWrapper
from srl.registry.builder import ModelBuilder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model  = ModelBuilder.from_yaml("configs/envs/halfcheetah_sac.yaml")
target = ModelBuilder.from_yaml("configs/envs/halfcheetah_sac.yaml")

cfg = SACConfig(action_dim=6, buffer_size=1_000_000, batch_size=256,
                learning_starts=10_000)
agent = SAC(model, target, config=cfg, device=DEVICE)

env = GymnasiumWrapper(gym.make("HalfCheetah-v5"))
obs, _ = env.reset()

for step in range(1_000_000):
    obs_t = {k: torch.from_numpy(v).float().unsqueeze(0).to(DEVICE)
             for k, v in obs.items()}
    if step < cfg.learning_starts:
        action_np = env.act_space.sample()
    else:
        action, *_ = agent.predict(obs_t)
        action_np = action.squeeze(0).cpu().numpy()

    obs, reward, done, truncated, _ = env.step(action_np)
    agent.buffer.add(obs=obs, action=action_np,
                     reward=[reward], done=[done], truncated=[truncated],
                     next_obs=obs)
    if done or truncated:
        obs, _ = env.reset()

    if step >= cfg.learning_starts:
        agent.update()
```

---

## 5. Load a checkpoint

```python
from srl.utils.checkpoint import CheckpointManager
cm = CheckpointManager("checkpoints/sac_halfcheetah_v5")
cm.load(agent.model, step="latest")
```
