# Configuration Reference

SRL uses YAML files to define the model graph and the currently supported declarative parts of training.

This page is the schema-level companion to the [YAML Core Guide](yaml_core.md).

Use this page when you need exact field names and supported options. Use the YAML core guide when you need the bigger picture of how the system fits together.

---

## File structure

```text
schema_version: 1        # optional; a missing value means 1

# Optional — selects the algorithm (else inferred from the file name, else --algo)
algo: ppo | sac | ddpg | td3 | a2c | a3c | <plugin>

# Environment
env_id: <str>
env_type: flat | goal | racecar | isaaclab | mjlab | <plugin>
include_goal: <bool>     # goal envs only (default true)

# List of encoder modules
encoders:
  - name: <str>           # unique key; referenced in flows
    type: mlp | cnn | lstm | text | <registered type>
    input_name: <str>     # optional explicit obs key mapping
    input_dim: <int> | auto       # mlp/lstm; omitted = auto
    input_shape: [C, H, W] | auto # cnn; omitted = auto
    latent_dim: <int>     # output dimension
    aux_type: autoencoder | contrastive | byol   # optional
    aux_latent_dim: <int> # optional projection width
    use_momentum: <bool>
    momentum_tau: <float>
    recurrent: <bool>
    lstm_hidden: <int>
    layers:
      - {out_features: <int>, activation: <str>, norm: <str>}
    extra: {<key>: <value>}  # free-form kwargs for custom encoder types

# Data-flow graph (encoder → head)
flows:
  - "<encoder_name> -> actor"
  - "<encoder_name> -> critic"

# Actor head
actor:
  name: actor             # optional, defaults to "actor"
  type: gaussian | squashed_gaussian | deterministic | <registered type>
  action_dim: <int> | auto  # omitted = auto
  log_std_init: <float>
  log_std_min: <float>
  log_std_max: <float>

# Critic head
critic:
  name: critic            # optional, defaults to "critic"
  type: value | twin_q | q | q_function | <registered type>
  action_dim: <int> | auto  # Q-critics only; omitted = auto

# Trainer keys + algorithm hyperparameters (see "Train section")
train:
  total_steps: <int>
  n_envs: <int>
  ... algorithm-specific fields ...
```

---

## YAML-first rules

- `encoders` declare the feature extractors that enter the graph.
- `flows` declare how encoder outputs feed into downstream nodes.
- `actor` and `critic` are regular graph nodes built from `HeadConfig`.
- `train` holds trainer keys (`total_steps`, `seed`, ...) and the selected
  algorithm's config-dataclass fields.

(strict-validation)=
### Strict validation

Configs are validated **strictly** — by `srl.config.load_experiment()` for the
whole file and by `ModelBuilder.from_dict()` / `from_yaml()` for the model
section — before anything is built:

- **Unknown keys are errors**, anywhere (top level, `encoders[i]`, `actor`,
  `critic`, `train`, `visualization`, `ros2`), with a "did you mean" suggestion:

  ```text
  ConfigError: 2 configuration errors:
    - encoders[0].latnet_dim: unknown key for EncoderConfig (type='mlp') (did you mean 'latent_dim'?)
    - train.lrr: unknown key (did you mean 'lr'?)
  ```

  A `train:` key that belongs to a *different* algorithm is named as such
  (`train.tau: not a PPO hyperparameter (PPOConfig); it belongs to DDPG, SAC, TD3`).
  All problems in a file are reported together.
- **Types** are checked and YAML scalars coerced: `lr: 3e-4` (which PyYAML
  reads as a string) becomes a float, `n_steps: 2e3` becomes `2000`; `n_steps: 3.5`
  or `normalize_obs: "yes"` are errors.
- **Enumerations** are checked: `lr_schedule` (`fixed`/`adaptive`), `action_noise`
  (`gaussian`/`ou`), `her_strategy` (`future`/`final`/`episode`/`random`),
  `aux_loss_type`, `augmentation_mode`, `aux_type`, `target_entropy` (`auto` or a
  number), `env_type`, encoder/head `type` (registered types), `algo`.
- **Ranges** are checked, e.g. `gamma` and `tau` in (0, 1], learning rates > 0,
  `gae_lambda` in [0, 1], counts such as `batch_size`/`n_steps`/`buffer_size` ≥ 1,
  `min_lr <= max_lr`, `log_std_min < log_std_max`. These checks run in the config
  dataclasses' `__post_init__`, so they also apply to configs built in Python
  (`PPOConfig(gamma=1.5)` raises).
- **Flows** must reference declared encoders/heads (with a suggestion on typos).
- An encoder `name` declared more than once with a different
  `type`/`latent_dim`/`input_dim`/`input_shape` is an error (only the first
  declaration would be built).
- `aux_type: autoencoder` on an encoder without an image `input_shape` is an
  error (no decoder could be built).

`ConfigError` subclasses `ValueError`; its `.errors` attribute lists each problem.

### `schema_version`

The current schema is version **1**. A file without `schema_version` is read as
version 1 (no warning for now); any other value is rejected.

(auto-dims)=
### `auto` dimensions

`input_dim`, `input_shape` and `action_dim` may be written as `auto` or simply
omitted. They are then resolved from the environment's observation and action
spaces by `srl.config.resolve_auto_dims()` (or by passing
`obs_space=`/`act_space=` to `ModelBuilder.from_dict` / `from_yaml`):

- each encoder is matched to an observation key with the runtime routing rules
  (explicit `input_name`, then name match, single-observation broadcast, order zip);
- `mlp`/`lstm` encoders get `prod(shape)`; `cnn` encoders get the `[C, H, W]`
  shape (HWC images are transposed exactly like the env wrappers do);
- the actor and Q-style critics get the flat action dimension; a `value`
  critic gets none.

**Explicit numbers still work** and are cross-checked against the env: a
mismatch raises a `ConfigError` naming the encoder/head and both values, instead
of surfacing as a shape error in the first forward pass. Building a model whose
dims are still `auto` without env spaces raises a clear `ConfigError` before any
layer is created.

```yaml
encoders:
  - name: state_enc
    type: mlp            # input_dim omitted -> auto
    latent_dim: 256
actor:
  type: squashed_gaussian   # action_dim omitted -> auto
```

### Removed / deprecated keys

| Key | Status |
|---|---|
| top-level `losses:` | Never read by any algorithm. Ignored with a `ConfigDeprecationWarning`; set loss weights through `train:` (`vf_coef`, `entropy_coef`, `aux_weight`, ...). |
| encoder `frame_stack` | Never applied. `1` warns and is ignored; any other value is an error (stack frames in an env wrapper). |
| `train.start_steps` | Alias of `learning_starts` (random-action warmup). Folded with a warning; `update_after` keeps its previous default. |
| `train.update_every` | Alias of `train_freq`. Folded with a warning; setting both to different values is an error. |
| `train.prefill_steps`, `train.queue_maxsize` | Were never read by `AsyncOffPolicyRunner`. Ignored with a warning. |
| `train.env_id` / `train.env_type` / `train.include_goal` | Accepted with a warning; put them at the top level. |

`ConfigDeprecationWarning` is a `FutureWarning` subclass, so it is shown by default.

## Encoder types

| Type | `input_dim` | Notes |
|---|---|---|
| `mlp` | obs dimension | Fully-connected layers |
| `cnn` | `input_shape: [C, H, W]` | Convolutional encoder for pixels |
| `lstm` | obs dimension | Recurrent encoder built from an MLP + LSTM |
| `text` | — | Character CNN; accepts `vocab_size`, `embed_dim`, `channels`, `max_len` |

## Encoder fields

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | Unique node name used in `flows` |
| `type` | `str` | Encoder family or registered custom encoder key |
| `input_name` | `str \| null` | Explicit observation key for this encoder |
| `input_dim` | `int \| auto` | Vector encoders; omitted/`auto` = from the env |
| `input_shape` | `list[int] \| auto` | CNN encoders (`[C, H, W]`); omitted/`auto` = from the env |
| `latent_dim` | `int` | Encoder output width for non-recurrent paths |
| `layers` | `list` | Layer definitions passed to the encoder builder |
| `aux_type` | `str \| null` | Auxiliary head family attached to this encoder |
| `aux_latent_dim` | `int` | Projection size for contrastive or BYOL heads |
| `use_momentum` | `bool` | Wrap encoder with momentum/EMA behavior |
| `momentum_tau` | `float` | EMA coefficient |
| `recurrent` | `bool` | Wrap non-LSTM encoders with LSTM |
| `lstm_hidden` | `int` | Hidden size for recurrent wrapping |
| `extra` | `dict` | Keyword arguments for custom encoder types; type-declared keys (e.g. `text`'s `vocab_size`, `embed_dim`, `channels`, `max_len`) may also be written at the top level |

### `input_name`

`input_name` is the preferred way to map observation keys to encoders in multimodal systems.

Example:

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_name: joint_states
    input_dim: 24
    latent_dim: 128

  - name: image_enc
    type: cnn
    input_name: front_camera
    input_shape: [3, 84, 84]
    latent_dim: 256
```

When `input_name` is set:

- the runtime routes by that key first
- missing keys raise `KeyError`
- extra keys that remain unused generate a warning

## Layer entries

Layer lists accept either shorthand or explicit dictionaries depending on the encoder/head builder.

Common dictionary fields from `LayerConfig`:

| Field | Meaning |
|---|---|
| `out_features` | MLP output width |
| `out_channels` | CNN output channels |
| `kernel` | CNN kernel size |
| `stride` | CNN stride |
| `padding` | CNN padding |
| `activation` | Activation function name |
| `norm` | Normalization type |
| `dropout` | Dropout probability |
| `dropout_type` | Standard, variational, or spatial |
| `pooling` | Pooling type |
| `pooling_kernel` | Pooling kernel size |
| `residual` | Residual block flag |
| `depthwise` | Depthwise convolution flag |
| `norm_order` | `pre` or `post` normalization |
| `weight_init` | Weight init strategy |

```{warning}
CNN shorthand lists are positional in the order
`[out_channels, kernel, stride, activation, pooling]` — the third slot is **stride**
(padding then defaults to `"same"` for stride 1 and `"valid"` otherwise). `padding`
is only reachable through the dict form. See
[Heads & Flows](yaml/heads_flows.md#layer-specification).
```

## Flow graph

`flows` is a list of strings in the form `"src -> dst"`.

```yaml
flows:
  - "visual_enc -> actor"
  - "state_enc -> actor"
  - "visual_enc -> critic"
  - "state_enc -> critic"
```

Rules:

- sources and destinations must be declared nodes
- cycles are not allowed
- multiple upstream inputs are concatenated automatically
- actor and critic can consume different subsets of encoders

## Head fields

Common `HeadConfig` fields:

| Field | Meaning |
|---|---|
| `name` | Node name, typically `actor` or `critic` |
| `type` | Built-in head type or registry key |
| `action_dim` | Actor heads and Q-style critics; omitted/`auto` = from the env |
| `layers` | Optional head-specific MLP layers |
| `log_std_init` | Initial log standard deviation |
| `log_std_min` | Minimum log std clamp |
| `log_std_max` | Maximum log std clamp |
| `state_dependent_std` | Whether std is state-conditioned (gaussian heads; default `false` = learned state-independent log_std, as in CleanRL/SB3) |

---

## Actor head types

| Type | Distribution | Use with |
|---|---|---|
| `gaussian` | Normal — unbounded | PPO, A2C, A3C |
| `squashed_gaussian` | Tanh-Normal — bounded [−1, 1] | SAC |
| `deterministic` | No distribution | DDPG |

---

## Critic head types

| Type | Output | Use with |
|---|---|---|
| `value` | `V(s)` scalar | PPO, A2C, A3C |
| `twin_q` | `[Q1(s,a), Q2(s,a)]` | SAC, TD3, DDPG |
| `q_function` | `Q(s,a)` scalar (single critic) | DDPG |
| `q` | Alias for `q_function` | DDPG |

---

## Train section

The `train:` block is split by key into four groups; anything else is an error.

**Trainer keys** (`srl.config.TrainerConfig`; each also has a `srl-train` flag
that wins when passed):

| Key | Type | Default |
|---|---|---|
| `total_steps` | int ≥ 1 | 1_000_000 |
| `n_envs` | int ≥ 1 | 1 |
| `device` | str | `auto` |
| `seed` | int | 0 |
| `log_interval` | int ≥ 1 | 2048 |
| `eval_freq` | int ≥ 0 | 50_000 |
| `eval_episodes` | int ≥ 1 | 10 |
| `deterministic` | bool | false |
| `save_best` | bool | false |
| `save_interval` | int ≥ 1 | 100_000 |

**Async-runner keys** (`AsyncRunnerConfig`): `use_async`, `use_gpu_buffer`.

**Algorithm hyperparameters**: the fields of the selected algorithm's config
dataclass (`PPOConfig`, `SACConfig`, ... or the `Visual*` variant — see
[Training Block Reference](yaml/training_block.md)). Fields the trainer derives
from the environment (`action_dim`, `num_envs`, `replay_num_envs`) are
overwritten at run time; HER's `her_obs_dim`/`her_goal_dim`/`her_reward_fn`
cannot be set from YAML.

**Off-policy schedule** — canonical names, resolved in one place by
`srl.config.normalize_offpolicy_schedule()`:

| Key | Meaning | Default |
|---|---|---|
| `learning_starts` | env steps of uniform-random actions before the policy acts | 10_000 |
| `update_after` | env steps collected before the first gradient update | `learning_starts` |
| `train_freq` | env steps between update rounds | 1 |
| `gradient_steps` | gradient steps per update round | 1 |

See [halfcheetah_sac.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/halfcheetah_sac.yaml), [car_racing_ppo_visual.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/car_racing_ppo_visual.yaml), and [isaaclab_cartpole_ppo.yaml](https://github.com/Bigkatoan/SRL/blob/main/configs/envs/isaaclab_cartpole_ppo.yaml) for real examples.

## Python API

```python
from srl.config import load_experiment, dump_resolved_config, resolve_auto_dims_from_env
from srl.registry.builder import ModelBuilder

exp = load_experiment("configs/envs/halfcheetah_sac.yaml")  # ExperimentConfig, or ConfigError
exp.algo, exp.env.env_id, exp.trainer.total_steps, exp.algo_config.lr_actor

model_cfg = resolve_auto_dims_from_env(exp.model, env)  # fill `auto`, cross-check numbers
model = ModelBuilder.from_config(model_cfg)

dump_resolved_config(exp, "runs/my_run/config.yaml")    # reloads to an equal config
```

| Function | Purpose |
|---|---|
| `load_experiment(path_or_dict, *, algo=None)` | Validate a whole file → `ExperimentConfig` |
| `build_algo_config(algo, train, *, config_cls=None, model=None, overrides=None)` | Strict `train:` → algorithm dataclass |
| `build_trainer_config(train)` / `build_runner_config(train)` | Trainer keys / `AsyncRunnerConfig` |
| `normalize_offpolicy_schedule(cfg)` | Canonical `learning_starts`/`update_after`/`train_freq`/`gradient_steps` |
| `resolve_auto_dims(model_cfg, obs_space, act_space)` | Resolve/cross-check dims from spaces |
| `dump_resolved_config(cfg, path=None)` | Write the resolved config as YAML |

See the [API reference](api/config.md) for signatures.

---

## Example: Multi-modal (state + lidar)

```yaml
encoders:
  - name: state_enc
    type: mlp
    input_dim: 12
    latent_dim: 128
    layers:
      - {out_features: 128, activation: relu, norm: layer_norm}

  - name: lidar_enc
    type: mlp
    input_dim: 1080
    latent_dim: 128
    layers:
      - {out_features: 256, activation: relu, norm: none}
      - {out_features: 128, activation: relu, norm: none}

flows:
  - "state_enc -> actor"
  - "state_enc -> critic"
  - "lidar_enc -> actor"
  - "lidar_enc -> critic"
```

---

## Encoder optimizer fields (v0.2.0)

These fields appear on `SACConfig`, `DDPGConfig`, `TD3Config`, and `VisualSACConfig`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `encoder_update_freq` | `int` | `1` (state), `2` (visual) | Step the encoder optimizer every N critic updates |

`VisualSACConfig`-only fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `encoder_optimize_with_critic` | `bool` | `True` | Allow critic gradients to flow into the encoder |
| `aux_loss_type` | `str` | `"curl"` | Auxiliary encoder loss: `none \| ae \| vae \| curl \| byol \| drq \| spr \| barlow` |
| `encoder_lr` | `float` | `1e-4` | Learning rate for the dedicated encoder optimizer |

---

(asyncrunnerconfig)=
## AsyncRunnerConfig (v0.2.0)

Controls `AsyncOffPolicyRunner`. Built from the `train:` block by
`srl.config.build_runner_config()` (`use_async: true` / `use_gpu_buffer: true`),
or constructed directly and passed as the runner's `runner_cfg` argument.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `use_async` | `bool` | `False` | Split collection and training into separate threads |
| `use_gpu_buffer` | `bool` | `False` | Replace the CPU replay buffer with `GPUReplayBuffer` |

```python
from srl.core.config import AsyncRunnerConfig

runner_cfg = AsyncRunnerConfig(
    use_async      = True,
    use_gpu_buffer = True,
)
```

The former `prefill_steps` / `queue_maxsize` fields were never read and have been
removed: random-action warmup is the algorithm's `learning_starts` (the runner's
`random_steps` argument), and the runner has no collector/trainer queue.

See [async_runner.md](async_runner.md) for the full runner API.
