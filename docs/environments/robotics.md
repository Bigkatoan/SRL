# Gymnasium-Robotics Environments

Fetch manipulation tasks from [gymnasium-robotics](https://robotics.farama.org/).

Install: `pip install "srl-rl[robotics]"`

---

## Overview

All Fetch tasks use a **GoalEnv** interface — the observation is a dict:

```python
{
    "observation":    np.array(shape=(10,) or (25,)),  # robot state
    "achieved_goal":  np.array(shape=(3,)),             # current EE/object pos
    "desired_goal":   np.array(shape=(3,)),             # target pos
}
```

SRL's `GoalEnvWrapper` flattens this to `[observation | achieved_goal | desired_goal]`.

---

## Environments

| Env | Flat obs | Act | Difficulty | Target |
|---|---|---|---|---|
| FetchReach-v4 | 16 | 4 | ★☆☆☆ | success_rate > 0.90 |
| FetchPush-v4 | 31 | 4 | ★★☆☆ | success_rate > 0.80 |
| FetchPickAndPlace-v4 | 31 | 4 | ★★★☆ | success_rate > 0.70 |
| FetchSlide-v4 | 31 | 4 | ★★★★ | success_rate > 0.60 |

---

## Training

```bash
# Register envs first (automatic in train script)
python examples/envs/train_robotics.py --env FetchReach-v4
python examples/envs/train_robotics.py --env FetchPush-v4
python examples/envs/train_robotics.py --env FetchPickAndPlace-v4
python examples/envs/train_robotics.py --env FetchSlide-v4
```

---

## Using GoalEnvWrapper

```python
import gymnasium as gym
import gymnasium_robotics
gymnasium_robotics.register_robotics_envs()

from srl.envs.goal_env_wrapper import GoalEnvWrapper

env = GoalEnvWrapper(gym.make("FetchReach-v4"))
obs, info = env.reset()

# obs = {"state": array(16,)}  ← flattened [obs(10) | ag(3) | dg(3)]
# info["goal_obs"] = {"observation": ..., "achieved_goal": ..., "desired_goal": ...}

action = env.act_space.sample()
obs, reward, done, truncated, info = env.step(action)
print("success:", info.get("is_success"))
```

---

## Key hyperparameters (SAC)

| Hyperparameter | Value | Reason |
|---|---|---|
| `gamma` | 0.98 | Short-horizon tasks |
| `gradient_steps` | 40 | Compensate for sparse reward |
| `learning_starts` | 10 000 | Warm-start replay buffer |
| `batch_size` | 256 | Standard SAC |
