# Gymnasium Integration

SRL uses Gymnasium as the standard API for every environment that is not an Isaac Lab
or mjlab GPU simulator.

## Wrapper

`GymnasiumWrapper` normalises a Gymnasium env into the SRL observation/action
interface:

```python
import gymnasium as gym
from srl.envs.gymnasium_wrapper import GymnasiumWrapper

env = GymnasiumWrapper(gym.make("HalfCheetah-v5"))
obs, _ = env.reset()
# obs = {"state": array(17,)}

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

The wrapper wraps the raw observation in a single-key dict. The key is `state` by
default and can be changed with the `obs_key` constructor argument (for example
`GymnasiumWrapper(env, obs_key="pixels")` for pixel observations). It also exposes
`obs_space`/`act_space` helpers and sets `num_envs = 1` so single envs and vector envs
have the same surface.

```{note}
`step()` returns `terminated` and `truncated` separately and never collapses them into
a single `done`. Off-policy algorithms bootstrap `(1 - done) * next_q`, so folding a
time-limit cutoff into `done` would bias Q-values low on any env with an episode limit.
```

## Supported environments

| Family | Example env IDs | `env_type` |
|---|---|---|
| Classic control | `Pendulum-v1`, `MountainCarContinuous-v0` | `flat` |
| MuJoCo | `HalfCheetah-v5`, `Ant-v5`, `Humanoid-v5` | `flat` |
| Box2D | `LunarLanderContinuous-v3`, `BipedalWalker-v3`, `CarRacing-v3` | `flat` |
| Gymnasium-Robotics | `FetchReach-v4`, `FetchPush-v4` | `goal` |
| racecar_gym | `SingleAgentAustria-v0`, `SingleAgentBerlin-v0` | `racecar` |

SRL targets **continuous action spaces** only, so discrete-action classics such as
`CartPole-v1` are out of scope. See [Supported Environments](../environments/index.md)
for the full table with observation/action dimensions and convergence budgets.

## `env_type`

`env_type` selects which wrapper the CLI builds and therefore how observations are
shaped:

| `env_type` | Wrapper | Observation format | Use for |
|---|---|---|---|
| `flat` | `GymnasiumWrapper` | `{"state": array}` | Vector and pixel Box observations |
| `goal` | `GoalEnvWrapper` | `{"state": [observation \| achieved_goal \| desired_goal]}` | Goal-conditioned envs |
| `racecar` | `RacecarWrapper` | `{"state": array}` (Dict obs flattened) | racecar_gym tracks |
| `isaaclab` | `IsaacLabWrapper` | one key per Isaac Lab obs group | Isaac Lab tasks |
| `mjlab` | `IsaacLabWrapper` | one key per mjlab obs group | mjlab tasks |

Anything else falls back to `flat`. Pixel-based Gymnasium envs such as `CarRacing-v3`
also use `flat` — the shipped [car_racing_ppo_visual.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/car_racing_ppo_visual.yaml)
config pairs `env_type: flat` with a `cnn` encoder.

## Parallel environments

Set `n_envs` in the `train:` block (or pass `--n-envs`) to collect from several envs at
once. The CLI picks the vector backend from `--vec-mode` (`auto` defaults to
`AsyncVectorEnv` when `n_envs > 1`):

```yaml
train:
  n_envs: 8
```

## See also

- [Isaac Lab Integration](isaaclab.md) — for GPU simulators
- [Supported Environments](../environments/index.md)
