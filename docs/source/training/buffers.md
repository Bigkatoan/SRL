# Replay Buffers

SRL ships two replay buffers: a CPU/numpy one (the default) and a CUDA-resident one
for GPU simulators.

## CPU `ReplayBuffer` (default)

A standard circular buffer that stores transitions as numpy arrays.

```python
from srl.core.replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=1_000_000)
buf.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
batch = buf.sample(256)
```

`obs_shape` and `action_dim` are optional — the buffer allocates lazily from the first
transition you add. This is the right choice for almost every workload.

```{warning}
Pass the transition fields by keyword. `ReplayBuffer.add` and `GPUReplayBuffer.add`
have different positional orders (`(obs, action, reward, next_obs, done)` versus
`(obs, action, reward, done, next_obs)`), so positional calls do not transfer between
the two.
```

`done` must be **true termination only**, never OR'd with `truncated`. SAC, DDPG, and
TD3 bootstrap the target as `(1 - done) * next_q`, so folding time-limit cutoffs into
`done` biases Q-values low on every env with an episode limit. Pass `truncated`
separately.

## `GPUReplayBuffer` (v0.2.0)

`GPUReplayBuffer` is a pre-allocated CUDA circular buffer aimed at GPU-batched
simulators.

### The problem with the CPU buffer

```
CPU ReplayBuffer:
  GPU sim (CUDA) → numpy copy (host) → store → batch.to(device) → GPU
  ↑ host↔device round-trips on both the store and the sample path
```

### The fix

```
GPUReplayBuffer:
  GPU sim (CUDA) → GPUReplayBuffer (CUDA) → batch already on the GPU
  ↑ no host↔device copies
```

`add()` accepts CUDA tensors and writes them through
`Tensor.copy_(src, non_blocking=True)` on a dedicated copy stream; `sample()` returns a
`ReplayBatch` whose tensors already live on `device`, so no `.to(device)` is needed in
the training loop. It also takes numpy arrays via `torch.as_tensor`, and `add()` is
thread-safe, which is what makes it usable from the async runner's collector thread.

```{note}
Getting the end-to-end zero-copy path requires feeding the buffer CUDA tensors
yourself. The CLI's `IsaacLabWrapper` converts observations to CPU numpy before they
reach the buffer, so the copy-free path today is a Python-API one.
```

### Usage

```python
from srl.core.gpu_replay_buffer import GPUReplayBuffer

buf = GPUReplayBuffer(
    capacity=1_000_000,
    device="cuda:0",
)

buf.add(
    obs={"pixels": obs_tensor, "state": state_tensor},
    action=action_tensor,
    reward=reward_tensor,
    done=done_tensor,
    next_obs={"pixels": next_pixels, "state": next_state},
)

batch = buf.sample(256)
# batch.obs["pixels"]: (256, C, H, W) on cuda:0
# batch.actions:       (256, action_dim) on cuda:0
```

### Constructor arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `capacity` | `int` | required | Maximum number of transitions |
| `device` | `str \| torch.device` | `"cuda"` | Device all storage is allocated on |
| `n_step` | `int` | `1` | N-step return horizon (1 = standard 1-step TD) |
| `gamma` | `float` | `0.99` | Discount used for n-step returns |
| `use_fp16` | `bool` | `False` | Store float32 obs/actions as float16 to halve VRAM |
| `num_envs` | `int` | `1` | Number of parallel envs writing to the buffer |

### Checkpointing

`GPUReplayBuffer` serialises to CPU tensors, so a saved buffer restores onto any
device:

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
## Pairing with the async runner

`AsyncOffPolicyRunner` swaps the agent's CPU buffer for a `GPUReplayBuffer` when
`use_gpu_buffer=True`, carrying `capacity`, `n_step`, `gamma`, `use_fp16`, and the env
count over from the old buffer:

```python
from srl.core.config import AsyncRunnerConfig
from srl.runners import AsyncOffPolicyRunner

runner = AsyncOffPolicyRunner(
    agent=sac_agent,
    env=env,
    total_steps=1_000_000,
    runner_cfg=AsyncRunnerConfig(use_async=True, use_gpu_buffer=True),
    device="cuda:0",
)
runner.run()
```

With that configuration:

1. The collector (main thread) writes CUDA tensors straight into the `GPUReplayBuffer`.
2. The trainer (daemon thread) samples batches that are already on the GPU.
3. There are no CPU↔GPU copies in the hot path.

```{note}
`use_async` and `use_gpu_buffer` also work directly in a YAML `train:` block --
`srl-train` builds an `AsyncRunnerConfig` from those two keys automatically. See
[Async Off-Policy Runner](../async_runner.md#srl-train--yaml-usage) for the YAML form.
```

## See also

- [GPU Replay Buffer API](../gpu_replay_buffer.md)
- [Runners & Training Loop](runners.md)
- [Isaac Lab Integration](../integrations/isaaclab.md)
