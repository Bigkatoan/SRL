# Gymnasium Integration

SRL dùng Gymnasium làm API chuẩn cho tất cả các environment không phải Isaac Lab.

## Wrapper

`GymnasiumWrapper` chuyển đổi Gymnasium env về dạng SRL-compatible:

```python
import gymnasium as gym
from srl.envs.gymnasium_wrapper import GymnasiumWrapper

env = GymnasiumWrapper(gym.make("HalfCheetah-v5"))
obs, _ = env.reset()
# obs = {"state": array(17,)}

action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
env.close()
```

## Các môi trường được hỗ trợ

| Family | Env IDs | env_type |
|---|---|---|
| Classic Control | Pendulum-v1, CartPole-v1 | flat |
| MuJoCo | HalfCheetah-v5, Ant-v5, Humanoid-v5 | flat |
| Box2D | LunarLanderContinuous-v2, BipedalWalker-v3 | flat |
| Gymnasium Robotics | FetchReach-v4, FetchPush-v4 | goal |
| racecar_gym | racecargym-v0 | flat |
| Visual | CarRacing-v3 | visual |

## env_type

`env_type` xác định cách observation được xử lý:

| env_type | Observation format | Dùng khi |
|---|---|---|
| `flat` | `{"state": array}` | Vector observations |
| `goal` | `{"observation": arr, "achieved_goal": arr, "desired_goal": arr}` | Goal-conditioned envs |
| `visual` | `{"pixels": array(C,H,W)}` | Pixel-based envs |

## Parallel environments

```yaml
train:
  n_envs: 8
```

## Xem thêm

- [Isaac Lab Integration](isaaclab.md) — cho GPU simulators
- [Environments Overview](../environments/index.md)
