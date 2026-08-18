# Training Block Reference

The `train:` block holds the hyperparameters the CLI reads and turns into algorithm
config dataclasses.

## How the block is consumed

`srl-train` uses it in two ways:

1. **`total_steps`, `n_envs`, `env_id`, and `env_type`** are read directly by the CLI
   (each is overridden by the matching command-line flag when one is passed).
2. **Everything else** is matched by name against the fields of the algorithm's config
   dataclass in [srl/core/config.py](https://github.com/Bigkatoan/SRL/blob/main/srl/core/config.py)
   — `PPOConfig`, `A2CConfig`, `A3CConfig`, `SACConfig`, `DDPGConfig`, or `TD3Config`.

```{warning}
Keys that do not match a field on the selected algorithm's dataclass are **silently
ignored**. If a hyperparameter appears to have no effect, check that it is spelled
exactly as the dataclass field and that it exists on the algorithm you are running.
```

Two consequences worth knowing:

- `device`, `seed`, `log_interval`, `eval_freq`, and `eval_episodes` are **command-line
  flags only** (`--device`, `--seed`, `--log-interval`, `--eval-freq`,
  `--eval-episodes`). They are not read from the `train:` block.
- `action_dim` and the parallel-env count are derived from the environment and
  overwrite whatever the YAML says.

## Fields read directly by the CLI

| Field | Type | Default | Description |
|---|---|---|---|
| `total_steps` | int | 1_000_000 | Total environment steps (`--steps`) |
| `n_envs` | int | 1 | Parallel environments (`--n-envs`) |
| `env_id` | str | — | Environment id; also settable at the top level of the file |
| `env_type` | str | `flat` | Wrapper selection; also settable at the top level |

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
| `learning_starts` | int | 10_000 | Random steps before learning begins |
| `start_steps` | int | `null` | Overrides `learning_starts` for the random-action warmup |
| `update_after` | int | `null` | Minimum buffer fill before the first update; falls back to `learning_starts` |
| `update_every` | int | `null` | Env steps between updates; falls back to `train_freq` |
| `gradient_steps` | int | 1 | Gradient updates per trigger |
| `encoder_update_freq` | int | 1 | Encoder optimizer steps every N critic updates |
| `replay_n_step` | int | 1 | N-step return horizon for the replay buffer |
| `use_fp16` | bool | false | Half-precision storage/compute |

SAC also accepts `alpha`, `init_alpha`, `auto_entropy_tuning`, `target_entropy`,
`target_update_interval`, `train_freq`, and the prioritised-replay fields `use_per`,
`per_alpha`, `per_beta_start`. DDPG and TD3 accept `action_noise`
(`gaussian` | `ou`) and `noise_sigma`; TD3 additionally accepts `policy_noise`
(0.2), `noise_clip` (0.5), and `policy_delay` (2).

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
| `clip_range_vf` | float | `null` | Value clip range; `null` reuses `clip_range` |
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
  start_steps:                  1000
  update_after:                 1000
  update_every:                 50
  gradient_steps:               1
  encoder_update_freq:          2
  encoder_optimize_with_critic: true
  aux_loss_type:                curl
```

`encoder_lr` is a `VisualSACConfig`-only field, so setting it here is what makes
`srl-train` auto-select `VisualSACConfig` for this run (see the note above) --
`eval_freq` and `eval_episodes` are **not** valid inside `train:`; they are
`--eval-freq`/`--eval-episodes` command-line flags only.

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
- [CLI Reference](../cli.md) — the flags that are not settable from YAML
