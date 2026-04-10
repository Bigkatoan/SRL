# Isaac Lab Environments

[Isaac Lab](https://isaac-sim.github.io/IsaacLab/) is NVIDIA's GPU-accelerated
robot learning framework built on Isaac Sim.

---

## Requirements

- NVIDIA GPU (RTX 3090 or better recommended)
- Isaac Sim ≥ 5.1
- Isaac Lab ≥ 0.5
- Python 3.10 or 3.11

---

## Supported environments

| Env | obs | act | n_envs | Steps |
|---|---|---|---|---|
| Isaac-Cartpole-v0 | 4 | 1 | 512 | ~500k |
| Isaac-Ant-v0 | 60 | 8 | 4 096 | ~5M |
| Isaac-Humanoid-v0 | 87 | 21 | 4 096 | ~10M |

---

## Training

```bash
# Activate Isaac Lab environment first
source /path/to/IsaacLab/_isaac_sim/setup_conda_env.sh
conda activate isaaclab

# Install SRL
pip install srl-rl

# Train
python examples/envs/train_isaaclab.py --env Isaac-Cartpole-v0  --headless
python examples/envs/train_isaaclab.py --env Isaac-Ant-v0       --headless --n-envs 4096
python examples/envs/train_isaaclab.py --env Isaac-Humanoid-v0  --headless --n-envs 4096
```

---

## IsaacLabWrapper

```python
from srl.envs.isaac_lab_wrapper import IsaacLabWrapper

env = IsaacLabWrapper("Isaac-Cartpole-v0", num_envs=512)
obs, _ = env.reset()
# obs = {"state": tensor(512, 4)}   ← GPU tensor converted to numpy

action = env.act_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

---

## PPO config for Isaac Lab

Isaac Lab envs benefit from:

- **Large `n_envs`** (512–4096): GPU parallelism
- **Short `n_steps`** (16–32): Fast inner loops
- **Large `batch_size`** (8k–32k): Efficient GPU utilisation
- **`elu` activations**: Smoother gradients than `tanh`
- **`vf_coef=1.0`**: Equal policy/value loss weighting

```yaml
train:
  total_steps: 5_000_000
  n_envs: 4096
  n_steps: 32
  batch_size: 16384
  n_epochs: 5
  lr: 5e-4
  entropy_coef: 0.005
  vf_coef: 1.0
  max_grad_norm: 1.0
```
