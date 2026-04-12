# MuJoCo Environments

MuJoCo locomotion tasks are the standard benchmark for continuous-control RL.  
Install with: `pip install "srl-rl[mujoco]"`

---

## Environments at a glance

| Env | obs | act | Best algo | Target | Config |
|---|---|---|---|---|---|
| HalfCheetah-v5 | 17 | 6 | SAC | 8 000 | `halfcheetah_sac.yaml` |
| Ant-v5 | 105 | 8 | SAC | 4 000 | `ant_sac.yaml` |
| Hopper-v5 | 11 | 3 | PPO | 2 500 | `hopper_ppo.yaml` |
| Walker2d-v5 | 17 | 6 | PPO | 3 000 | `walker2d_ppo.yaml` |
| Humanoid-v5 | 348 | 17 | PPO | 5 000 | `humanoid_ppo.yaml` |
| Swimmer-v5 | 8 | 2 | SAC | 300 | `swimmer_sac.yaml` |
| Pusher-v5 | 23 | 7 | SAC | −50 | `pusher_sac.yaml` |
| Reacher-v5 | 10 | 2 | SAC | −5 | `reacher_sac.yaml` |

---

## Training

=== "SAC (off-policy)"
    ```bash
    # Best for: HalfCheetah, Ant, Swimmer, Pusher, Reacher
    python examples/envs/train_mujoco.py --env HalfCheetah-v5 --algo sac
    python examples/envs/train_mujoco.py --env Ant-v5         --algo sac
    python examples/envs/train_mujoco.py --env Swimmer-v5     --algo sac
    ```

=== "PPO (on-policy)"
    ```bash
    # Best for: Hopper, Walker2d, Humanoid
    python examples/envs/train_mujoco.py --env Hopper-v5   --algo ppo
    python examples/envs/train_mujoco.py --env Walker2d-v5 --algo ppo
    python examples/envs/train_mujoco.py --env Humanoid-v5 --algo ppo
    ```

---

## HalfCheetah-v5 Deep Dive

HalfCheetah requires the agent to maximise forward velocity.

```yaml
# configs/envs/halfcheetah_sac.yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 17
    latent_dim: 256
    layers:
      - {out_features: 256, activation: relu, norm: layer_norm}
      - {out_features: 256, activation: relu, norm: layer_norm}
actor:
  type: squashed_gaussian
  action_dim: 6
critic:
  type: twin_q
  action_dim: 6
train:
  total_steps: 1_000_000
  gamma: 0.99
  tau: 0.005
  learning_starts: 10_000
```

---

## Humanoid-v5 Deep Dive

Humanoid is the most challenging locomotion task (348-dim obs, 17-dim act,
gravity-sensitive bipedal motion).  Use PPO with a large 3-layer network and
many parallel envs:

```yaml
# configs/envs/humanoid_ppo.yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 348
    latent_dim: 512
    layers:
      - {out_features: 512, activation: tanh}
      - {out_features: 512, activation: tanh}
      - {out_features: 256, activation: tanh}
train:
  total_steps: 10_000_000
  n_envs: 16
  n_steps: 2048
  batch_size: 512
```
