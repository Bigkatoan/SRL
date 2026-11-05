# Training Block Reference

The `train:` block holds the hyperparameters the CLI reads and turns into algorithm
config dataclasses.

## How the block is consumed

Every key of `train:` falls into one of these groups (see
`srl.config.split_train_section`); **anything else is an error** with a "did you
mean" suggestion — see [strict validation](../config_reference.md#strict-validation):

1. **Trainer keys** — `total_steps`, `n_envs`, `device`, `seed`, `log_interval`,
   `eval_freq`, `eval_episodes`, `deterministic`, `save_best`, `save_interval`,
   `checkpoint_replay_buffer`, `metrics_log_interval`. Most have a matching
   `srl-train` flag (`--steps`, `--n-envs`, `--device`, ...) that wins when passed.
2. **Async-runner keys** — `use_async`, `use_gpu_buffer` (`AsyncRunnerConfig`).
3. **Algorithm hyperparameters** — the fields of the algorithm's config dataclass in
   [srl/core/config.py](https://github.com/Bigkatoan/SRL/blob/main/srl/core/config.py)
   (`PPOConfig`, `A2CConfig`, `A3CConfig`, `SACConfig`, `DDPGConfig`, `TD3Config`, or the
   `VisualPPOConfig`/`VisualSACConfig` variants). Values are type-, enum- and
   range-checked; a key that belongs to a different algorithm is reported as such.
4. **Legacy env keys** — `env_id`, `env_type`, `include_goal` are still accepted here
   with a deprecation warning; put them at the top level of the file.

`action_dim`, `num_envs` and `replay_num_envs` are derived from the environment and
overwrite whatever the YAML says.

## Trainer keys

| Field | Type | Default | Description |
|---|---|---|---|
| `total_steps` | int | 1_000_000 | Total environment steps (`--steps`) |
| `n_envs` | int | 1 | Parallel environments (`--n-envs`) |
| `device` | str | `auto` | `auto`, `cpu`, `cuda`, ... (`--device`) |
| `seed` | int | 0 | Base seed (`--seed`) |
| `log_interval` | int | 2048 | Console log interval in env steps (`--log-interval`) |
| `eval_freq` | int | 50_000 | Evaluation interval in env steps; 0 disables (`--eval-freq`) |
| `eval_episodes` | int | 10 | Episodes per evaluation (`--eval-episodes`) |
| `deterministic` | bool | false | Deterministic torch kernels (`--deterministic`) |
| `save_best` | bool | false | Keep a `best_*` checkpoint (`--save-best`, see [`--save-best`](best-checkpoint-tracking)) |
| `save_interval` | int | 100_000 | Env steps between periodic `srl-train` checkpoints (no CLI flag) |
| `checkpoint_replay_buffer` | bool | false | Store the off-policy replay buffer in every `srl-train` checkpoint, for an exact `--resume` (can be gigabytes; no CLI flag) |
| `metrics_log_interval` | int | 1000 | Off-policy only: training metrics are averaged over this many env steps before being logged (1 = every gradient step; no CLI flag) |

## Off-policy fields (SAC, DDPG, TD3)

| Field | Type | Default | Description |
|---|---|---|---|
| `lr_actor` | float | 3e-4 (SAC), 1e-4 (DDPG/TD3) | Actor learning rate |
| `lr_critic` | float | 3e-4 (SAC), 1e-3 (DDPG/TD3) | Critic learning rate |
| `lr_alpha` | float | 3e-4 | Entropy-coefficient learning rate (SAC only) |
| `buffer_size` | int | 1_000_000 | Replay buffer capacity |
| `batch_size` | int | 256 | Gradient-update batch size |
| `gamma` | float | 0.99 | Discount factor |
| `tau` | float | 0.005 | Soft target-update coefficient |
| `learning_starts` | int | 10_000 | Env steps of uniform-random actions before the policy acts |
| `update_after` | int | `learning_starts` | Env steps collected before the first gradient update |
| `train_freq` | int | 1 | Env steps between update rounds |
| `gradient_steps` | int | 1 | Gradient updates per round |
| `encoder_update_freq` | int | 1 | Encoder optimizer steps every N critic updates |
| `replay_n_step` | int | 1 | N-step return horizon: per-env sliding window; targets use the stored `gamma**k` (k shortens at episode ends) |
| `normalize_obs` | bool | false | Running mean/std normalization of flat float observations (saved in checkpoints) |
| `obs_norm_clip` | float | 5.0 | Clip range applied after observation normalization |
| `use_per` | bool | false | Prioritized replay (importance-sampling weights, priorities from \|TD error\|); not combinable with HER |
| `per_alpha` / `per_beta_start` / `per_beta_steps` / `per_eps` | float/int | 0.6 / 0.4 / 1_000_000 / 1e-6 | PER exponent, IS-weight annealing, priority epsilon |
| `use_fp16` | bool | false | Half-precision replay storage (compute stays fp32) |

SAC also accepts `alpha`, `init_alpha`, `auto_entropy_tuning`, `target_entropy`,
`min_alpha` (floor on the auto-tuned temperature, permissive `1e-8` default —
see `SACConfig.min_alpha`'s docstring in `srl/core/config.py` for why this
exists) and `target_update_interval`. DDPG and TD3 accept `action_noise`
(`gaussian` | `ou`) and `noise_sigma`; TD3 additionally accepts `policy_noise`
(0.2), `noise_clip` (0.5), and `policy_delay` (2).

```{note}
`start_steps` and `update_every` are deprecated aliases of `learning_starts` and
`train_freq`. They still load (with a `ConfigDeprecationWarning`) and keep their
exact previous meaning — `start_steps` sets the random-action warmup while
`update_after` keeps defaulting to the *original* `learning_starts`.
`srl.config.normalize_offpolicy_schedule(cfg)` is the single function that resolves
the four canonical values for a config object.
```

```{note}
`encoder_lr`, `encoder_optimize_with_critic`, `aux_loss_type`, and `aux_weight` live on
`VisualSACConfig`, not plain `SACConfig`. `srl-train` builds `VisualSACConfig`
automatically when an encoder declares a recognised `aux_type`, or when the `train:`
block itself sets one of these Visual-only fields — see
[Auxiliary Representation Learning](auxiliary.md). Without `encoder_lr` the encoder
optimizer uses `lr_critic`.
```

## On-policy fields (PPO, A2C, A3C)

| Field | Type | Default | Description |
|---|---|---|---|
| `lr` | float | 3e-4 (PPO), 7e-4 (A2C), 1e-4 (A3C) | Learning rate |
| `n_steps` | int | 2048 (PPO), 5 (A2C), 20 (A3C) | Steps collected per env per rollout |
| `batch_size` | int | 64 (PPO) | Minibatch size within a rollout |
| `n_epochs` | int | 10 | Optimisation epochs per rollout (PPO) |
| `clip_range` | float | 0.2 | PPO clip parameter |
| `clip_range_vf` | float | `null` | Value clip range; `null` disables value clipping (SB3 default) |
| `gae_lambda` | float | 0.95 (PPO), 1.0 (A2C/A3C) | GAE lambda |
| `gamma` | float | 0.99 | Discount factor |
| `entropy_coef` | float | 0.0 (PPO), 0.01 (A2C/A3C) | Entropy regularisation |
| `vf_coef` | float | 0.5 (PPO/A3C), 0.25 (A2C) | Value-loss coefficient |
| `max_grad_norm` | float | 0.5 (PPO/A2C), 40.0 (A3C) | Gradient clipping |
| `target_kl` | float | `null` | Early-stop a PPO update on KL divergence (same-epoch only — see `lr_schedule` below for a persistent alternative) |
| `lr_schedule` | str | `"fixed"` (PPO) | `"adaptive"` continuously adapts `lr` from measured KL every minibatch, for the whole run (modeled on rsl_rl PPO's default schedule) |
| `desired_kl` | float | 0.01 (PPO) | Target KL for `lr_schedule: adaptive` |
| `min_lr` / `max_lr` | float | 1e-5 / 1e-2 (PPO) | Clamp range for `lr_schedule: adaptive` |
| `kl_lr_factor` | float | 1.5 (PPO) | Multiplicative step size for `lr_schedule: adaptive` |
| `use_fp16` | bool | false | Half-precision storage/compute |

A2C also accepts `rms_prop_eps`. A3C accepts `n_workers`, but the CLI overrides it with
`--n-envs`.

## Full example — SAC on a vision task

```yaml
train:
  total_steps:                  500_000
  n_envs:                       1
  batch_size:                   128
  gamma:                        0.99
  lr_actor:                     3e-4
  lr_critic:                    3e-4
  lr_alpha:                     3e-4
  encoder_lr:                   1e-4
  tau:                          0.005
  buffer_size:                  100_000
  learning_starts:              1000
  update_after:                 1000
  train_freq:                   50
  gradient_steps:               1
  encoder_update_freq:          2
  encoder_optimize_with_critic: true
  aux_loss_type:                curl
```

`encoder_lr` is a `VisualSACConfig`-only field, so setting it here is what makes
`srl-train` auto-select `VisualSACConfig` for this run (see the note above).

## Full example — PPO on Isaac Lab

```yaml
train:
  total_steps:   5_000_000
  n_envs:        4096
  n_steps:       32
  batch_size:    16384
  n_epochs:      5
  lr:            5e-4
  entropy_coef:  0.005
  vf_coef:       1.0
  max_grad_norm: 1.0
  gae_lambda:    0.95
  gamma:         0.99
```

## See also

- [Configuration Reference](../config_reference.md)
- [Algorithms](../algorithms.md)
- [CLI Reference](../cli.md) — the flags that override `train:` keys
