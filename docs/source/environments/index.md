# Supported Environments

SRL supports all **continuous action space** environments across four suites.

If you are working with a separate Isaac Lab task repository such as `M3bot`, see the dedicated [M3bot](m3bot.md) page for the verified machine-specific setup and runtime notes.

---

## Summary table

| Environment | Suite | Algorithm | obs dim | act dim | Steps to converge |
|---|---|---|---|---|---|
| Pendulum-v1 | Gymnasium Classic | PPO | 3 | 1 | ~200k |
| MountainCarContinuous-v0 | Gymnasium Classic | PPO | 2 | 1 | ~1M |
| BipedalWalker-v3 | Gymnasium Box2D | PPO | 24 | 4 | ~3M |
| LunarLanderContinuous-v3 | Gymnasium Box2D | PPO | 8 | 2 | ~1.5M |
| CarRacing-v3 | Gymnasium Box2D | PPO | 96x96x3 | 3 | ~5M |
| HalfCheetah-v5 | MuJoCo | PPO / **SAC** | 17 | 6 | 1–2M |
| Ant-v5 | MuJoCo | **SAC** | 105 | 8 | ~3M |
| Hopper-v5 | MuJoCo | PPO | 11 | 3 | ~1M |
| Walker2d-v5 | MuJoCo | PPO | 17 | 6 | ~2M |
| Humanoid-v5 | MuJoCo | PPO | 348 | 17 | ~10M |
| Swimmer-v5 | MuJoCo | **SAC** | 8 | 2 | ~500k |
| Pusher-v5 | MuJoCo | **SAC** | 23 | 7 | ~500k |
| Reacher-v5 | MuJoCo | **SAC** | 10 | 2 | ~200k |
| FetchReach-v4 | Gymnasium-Robotics | SAC | 16* | 4 | ~500k |
| FetchPush-v4 | Gymnasium-Robotics | SAC | 31* | 4 | ~2M |
| FetchPickAndPlace-v4 | Gymnasium-Robotics | SAC | 31* | 4 | ~5M |
| FetchSlide-v4 | Gymnasium-Robotics | SAC | 31* | 4 | ~5M |
| SingleAgentAustria-v0 | racecar_gym | PPO | ~1092** | 2 | ~2M |
| SingleAgentBerlin-v0 | racecar_gym | PPO | ~1092** | 2 | ~2M |
| Isaac-Cartpole-v0 | Isaac Lab | PPO | 4 | 1 | ~500k |
| Isaac-Ant-v0 | Isaac Lab | PPO | 60 | 8 | ~5M |
| Isaac-Humanoid-v0 | Isaac Lab | PPO | 87 | 21 | ~10M |
| Isaac-M3-Reach-v0 | M3bot / Isaac Lab | PPO | 19 | 4 | smoke-validated |
| Isaac-M3-Lift-v0 | M3bot / Isaac Lab | PPO | 28 | 5 | not yet locally verified |
| Isaac-M3-Push-v0 | M3bot / Isaac Lab | PPO | 22 | 4 | not yet locally verified |
| Isaac-M3-PickPlace-v0 | M3bot / Isaac Lab | PPO | 27 | 5 | not yet locally verified |

\* Flat concatenation: `[observation | achieved_goal | desired_goal]`  
\** Exact flattened obs dimension may vary by racecar_gym build/config.
\*** Isaac Lab envs run thousands of parallel envs on GPU.

---

## Wrappers

| Wrapper | File | Use for |
|---|---|---|
| `GymnasiumWrapper` | `srl/envs/gymnasium_wrapper.py` | Any flat Box obs env |
| `GoalEnvWrapper` | `srl/envs/goal_env_wrapper.py` | GoalEnv (Fetch, AntMaze) |
| `RacecarWrapper` | `srl/envs/racecar_wrapper.py` | racecar_gym Dict obs/action envs |
| `IsaacLabWrapper` | `srl/envs/isaac_lab_wrapper.py` | Isaac Lab GPU envs |
| `SyncVectorEnv` | `srl/envs/sync_vector_env.py` | Multiple envs, sequential |
| `AsyncVectorEnv` | `srl/envs/async_vector_env.py` | Multiple envs, parallel |

`M3bot` itself is not bundled into SRL as a built-in environment package. It is documented here because it is an active Isaac Lab task repo validated alongside SRL on the same machine.
