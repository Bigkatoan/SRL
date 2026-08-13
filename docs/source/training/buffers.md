# Replay Buffers

SRL có 2 loại replay buffer: CPU-based (default) và GPU-based (cho GPU simulators).

## CPU ReplayBuffer (default)

Standard circular buffer lưu transitions dưới dạng numpy arrays.

```python
from srl.core.replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=1_000_000)
buf.add(obs, action, reward, done, next_obs)
batch = buf.sample(256)
```

Dùng cho hầu hết mọi trường hợp.

## GPU Replay Buffer (v0.2.0)

`GPUReplayBuffer` là pre-allocated CUDA circular buffer, thiết kế cho zero-copy với Isaac Lab.

### Vấn đề với CPU buffer

```
CPU ReplayBuffer:
  Isaac Lab (CUDA) → numpy copy (host) → store → batch.to(device) → GPU
  ↑ 2 host↔device round-trips mỗi step
```

### Giải pháp

```
GPUReplayBuffer:
  Isaac Lab (CUDA) → GPUReplayBuffer (CUDA) → batch already on GPU
  ↑ 0 host↔device copies
```

### Sử dụng

```python
from srl.core.gpu_replay_buffer import GPUReplayBuffer

buf = GPUReplayBuffer(
    capacity=1_000_000,
    device="cuda:0",
)

buf.add(
    obs={"pixels": obs_tensor, "state": state_tensor},
    action=action_tensor,
    reward=reward_float,
    done=done_bool,
    next_obs={"pixels": next_pixels, "state": next_state},
)

batch = buf.sample(256)
# batch.obs["pixels"]: (256, C, H, W) on cuda:0
# batch.actions:       (256, action_dim) on cuda:0
```

### Constructor arguments

| Argument | Type | Default | Mô tả |
|---|---|---|---|
| `capacity` | int | required | Max transitions |
| `device` | str | `"cuda"` | Storage device |
| `storage_dtype` | torch.dtype | float32 | Precision |
| `n_step` | int | 1 | N-step return lookahead |
| `gamma` | float | 0.99 | Discount cho n-step |
| `num_envs` | int | 1 | Số parallel envs |

### Checkpointing

GPUReplayBuffer serialize sang CPU tensors khi lưu:

```python
state = buf.state_dict()    # CPU tensors, portable
buf.load_state_dict(state)  # restore on any device
```

## HER — Hindsight Experience Replay (goal-conditioned tasks)

`HERReplayBuffer` implements Hindsight Experience Replay
([Andrychowicz et al., 2017](https://arxiv.org/abs/1707.01495)) for
goal-conditioned environments with sparse rewards — the Fetch and AntMaze
families from `gymnasium-robotics`, for example.

On a task like `FetchReach-v4` the reward is `-1` on every step until the
gripper is within a small threshold of the goal, and `0` after. Early in
training the agent essentially never reaches the goal, so a plain
`ReplayBuffer` stores a stream of transitions that all carry reward `-1` and
the critic gets almost no signal. HER re-labels a fraction of each sampled
batch with a goal the trajectory *actually did* achieve, and recomputes the
reward against that goal — turning failed episodes into successful ones for a
different goal.

### Enabling it from YAML

Set `use_her: true` in the `train:` block of a goal-conditioned config
(`env_type: "goal"`). It is **off by default** everywhere:

```yaml
env_type: "goal"

train:
  use_her: true
  her_ratio: 0.8            # fraction of each batch that gets relabelled
  her_strategy: "future"    # future | final | episode | random
  her_max_episode_len: 50   # must be >= the env's episode length
```

Then train as usual:

```bash
srl-train --config configs/envs/fetch_reach_sac_her.yaml --env FetchReach-v4
```

A ready-made config ships as `configs/envs/fetch_reach_sac_her.yaml`. The run
prints a confirmation line at startup and logs `her/episodes` and
`her/transitions` alongside the usual metrics, so you can see relabelling is
actually engaged:

```
[srl-train] HER enabled: strategy=future ratio=0.8 obs_dim=13 goal_dim=3 max_episode_len=50
```

### Config fields

| Field | Default | Meaning |
| --- | --- | --- |
| `use_her` | `false` | Swap `ReplayBuffer` for `HERReplayBuffer`. |
| `her_ratio` | `0.8` | Fraction of a sampled batch whose `desired_goal` is relabelled. `0.8` is the paper's 4:1 ratio. |
| `her_strategy` | `"future"` | Which achieved goal to relabel with. |
| `her_max_episode_len` | `1000` | Per-episode preallocation; set it to the env's episode length to avoid wasting memory. |

Relabelling strategies:

- **`future`** — a goal achieved later in the same episode (recommended, and
  the paper's best-performing variant).
- **`final`** — the goal achieved at the end of the episode.
- **`episode`** — any goal achieved in the same episode.
- **`random`** — any goal achieved in any stored episode.

An unrecognised strategy raises at construction rather than silently
degrading to no relabelling.

### Requirements and limitations

- Requires `env_type: "goal"`, i.e. a `GoalEnvWrapper`-wrapped env exposing
  `achieved_goal`/`desired_goal` and a `compute_reward()` — the reward for a
  relabelled goal has to be recomputed, and only the env can do that. The CLI
  errors out at startup otherwise instead of quietly training without HER.
- Single environment only (`--n-envs 1`). HER stores whole episodes, and the
  CLI collects them from one un-vectorized env.
- Incompatible with `include_goal: false`.
- Currently wired for **SAC** only. DDPG/TD3 still build a plain
  `ReplayBuffer` even if these fields are set.
- Not compatible with the async / GPU-buffer fast path (`use_async`,
  `use_gpu_buffer`), which uses its own buffer.

### Using it directly

```python
from srl.core.her_replay_buffer import HERReplayBuffer

buf = HERReplayBuffer(
    capacity=1_000_000,
    obs_dim=13,       # [observation | achieved_goal]
    goal_dim=3,       # desired_goal
    action_dim=4,
    reward_fn=env.unwrapped.compute_reward,
    strategy="future",
    her_ratio=0.8,
    max_episode_len=50,
)

buf.add_transition(
    obs, achieved_goal, desired_goal, action,
    next_obs, next_achieved_goal,
    done=terminated,      # real termination only
    truncated=truncated,  # closes the episode without faking a terminal
)

batch = buf.sample(256)   # batch.obs["state"] -> (256, obs_dim + goal_dim)
```

Two details matter when driving it yourself:

- **Observation layout.** `sample()` appends the (possibly relabelled)
  desired goal itself, so the vector you pass as `obs` must *exclude* it.
  With `GoalEnvWrapper`, whose flat obs is
  `[observation | achieved_goal | desired_goal]`, store
  `[observation | achieved_goal]` — the concatenation then reproduces exactly
  the layout the encoders were built for.
- **`done` vs `truncated`.** Time-limited goal envs (every Fetch task) end by
  truncation with `terminated` always `False`. `truncated` closes the episode;
  `done` is what gets stored per timestep. Passing the time limit as `done`
  fabricates terminal states and biases the bootstrap target for the
  non-relabelled fraction of a batch.

Note that `len(buf)` counts **episodes** (the buffer is episode-indexed); use
`buf.num_transitions` for the transition count and `buf.can_sample(batch_size)`
for the readiness check.

## Kết hợp với Async Runner

```yaml
train:
  use_async: true
  use_gpu_buffer: true
```

Với cấu hình này:

1. Collector (main thread) viết CUDA tensors trực tiếp vào GPUReplayBuffer
2. Trainer (daemon thread) sample batch đã ở trên GPU
3. Không có CPU↔GPU copies trong hot path

## Xem thêm

- [GPU Replay Buffer API](../gpu_replay_buffer.md)
- [Runners & Training Loop](runners.md)
- [Isaac Lab Integration](../integrations/isaaclab.md)
