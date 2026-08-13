# Integrations

SRL integrates with the major RL ecosystems.

## Available integrations

| Environment | Integration | Buffer | Runner |
|---|---|---|---|
| Gymnasium (classic control, MuJoCo, Box2D) | `GymnasiumWrapper` | CPU | Sync |
| Isaac Lab (GPU sim) | `IsaacLabWrapper` | CPU (GPU via Python API) | Sync (async via Python API) |
| mjlab (GPU sim, MuJoCo-Warp) | `IsaacLabWrapper` | CPU (GPU via Python API) | Sync (async via Python API) |
| Real robot (ROS 2) | ROS 2 bridge | — | Realtime inference |

## Pages

- [Isaac Lab](isaaclab.md) — GPU-accelerated robot learning, with the full setup walkthrough
- [mjlab](mjlab.md) — the same "GPU-batched, one process" shape as Isaac Lab, without needing an Isaac Sim install
- [Gymnasium](gymnasium.md) — the standard RL environment API
- [ROS 2](../ros2.md) — deployment and sim-to-real transfer
