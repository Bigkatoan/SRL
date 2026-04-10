# Classic Control Environments

Classic control environments are lightweight, fast, and ideal for debugging.

---

## Pendulum-v1

| Property | Value |
|---|---|
| Observation | `(3,)` — `[cos θ, sin θ, θ̇]` |
| Action | `(1,)` — torque ∈ [−2, 2] |
| Max episode steps | 200 |
| Target return | > −200 |
| Config | `configs/envs/pendulum_ppo.yaml` |

=== "Training script"
    ```bash
    python examples/envs/train_classic_control.py --env Pendulum-v1
    ```

=== "Config"
    ```yaml
    algo: ppo
    encoders:
      - {name: state_enc, type: mlp, input_dim: 3, latent_dim: 64,
         layers: [{out_features: 64, activation: tanh, norm: none},
                  {out_features: 64, activation: tanh, norm: none}]}
    ```

=== "Python API"
    ```python
    from srl.envs.gymnasium_wrapper import GymnasiumWrapper
    import gymnasium as gym
    env = GymnasiumWrapper(gym.make("Pendulum-v1"))
    obs, _ = env.reset()
    # obs = {"state": array of shape (3,)}
    ```

---

## MountainCarContinuous-v0

| Property | Value |
|---|---|
| Observation | `(2,)` — `[position, velocity]` |
| Action | `(1,)` — push ∈ [−1, 1] |
| Max episode steps | 999 |
| Target return | > 90 |
| Config | `configs/envs/mountain_car_continuous_ppo.yaml` |

!!! tip "Sparse reward"
    MountainCarContinuous uses a sparse reward — the car only gets +100 when it
    reaches the goal. High entropy bonus (`entropy_coef=0.05`) and `gamma=0.999`
    help exploration.

=== "Training script"
    ```bash
    python examples/envs/train_classic_control.py --env MountainCarContinuous-v0
    ```
