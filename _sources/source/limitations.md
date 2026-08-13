# Limitations

SRL already has a strong YAML-first model system, but it is not yet a fully declarative end-to-end RL framework.

This page exists to make the current system boundary explicit.

## Currently unsupported or incomplete

### Discrete-action algorithms

The current library targets continuous-action control. The built-in actor heads and algorithm docs are centered on PPO, SAC, DDPG, TD3, A2C, and A3C in continuous settings.

### Fully declarative training orchestration

YAML currently defines model structure and part of training configuration, but it does not yet define the whole training loop as a first-class graph.

### Generic reward shaping pipelines in YAML

There is no schema-supported reward wrapper or reward transformation stack yet.

### Generic environment wrapper stacks in YAML

Environment creation is still handled procedurally by the CLI and wrapper classes.

### Arbitrary custom loss registration from YAML

Built-in losses can be selected, but there is not yet a general registry that lets users declare any custom loss purely from YAML.

### Hindsight Experience Replay (HER) beyond single-env SAC

HER *is* available from YAML/CLI: set `use_her: true` in the `train:` block of
a goal-conditioned config (`env_type: "goal"`) and `srl-train` builds a
`HERReplayBuffer` and routes episodes into it. See
[Replay Buffers](training/buffers.md#her--hindsight-experience-replay-goal-conditioned-tasks).

What is *not* supported yet:

- Only **SAC** is wired. `DDPGConfig` / `TD3Config` have no `use_her` field, so
  those algorithms still train with a plain uniform `ReplayBuffer` on goal
  tasks.
- **Single environment only.** HER stores whole episodes and the CLI collects
  them from one un-vectorized env; `--n-envs > 1` is rejected at startup.
- Not compatible with the async / GPU-buffer fast path (`use_async`,
  `use_gpu_buffer`), which manages its own buffer.

### Auxiliary encoder losses: partial YAML/CLI coverage

`srl-train` now builds `VisualPPOConfig` / `VisualSACConfig` automatically, so
auxiliary encoder losses do run from YAML. It switches to the Visual variant
when any encoder declares a recognised `aux_type` (`autoencoder`,
`contrastive`, `byol`), or when the `train:` block sets a Visual-only field
(`encoder_lr`, `aux_loss_type`, `aux_weight`, `augmentation_mode`, ...), and
derives `aux_loss_type` from the declared `aux_type` so the loss matches the
head that was built. PPO reports the result as `ppo/aux_loss`, SAC as
`sac/aux_loss`. Anything the CLI cannot honour is warned about rather than
silently ignored.

What still does **not** work:

- **PPO implements reconstruction only.** `PPO.update()` feeds every aux head
  through `reconstruction_loss()` and ignores `aux_loss_type`, so
  `aux_type: contrastive` / `byol` are skipped at runtime on PPO. Use
  `aux_type: autoencoder` with PPO, or SAC for the other objectives.
- **SAC's augmentation-based losses are unusable.** `curl`, `byol`, `drq`,
  `barlow` and `augmentation_mode: aggressive` all route through
  `random_crop()`, which outputs 90% of the input resolution instead of
  preserving it. The shared encoder is built for the full-size observation, so
  the augmented view fails the encoder's flatten/projection shapes. Only `ae`,
  `vae` and `spr` avoid augmentation. `curl` is still the `VisualSACConfig`
  default, so set `aux_loss_type` explicitly (or rely on the `aux_type`-derived
  value) rather than taking the default.
- **SAC matches encoders to observations by name substring.** The off-policy
  loop stores un-remapped environment observations in the replay buffer, and
  SAC's aux helpers then look for an encoder whose name contains the raw
  observation key (or vice-versa). `GymnasiumWrapper` emits `state`, so an
  encoder named `visual_enc` never matches and the auxiliary loss silently
  computes nothing. `configs/envs/car_racing_sac_visual.yaml` names its encoder
  `state_enc` for exactly this reason.
- **`use_fp16: true` is incompatible with CNN encoders.** The replay buffer
  returns half-precision tensors while model weights stay float32.

`configs/envs/car_racing_ppo_visual.yaml` and
`configs/envs/car_racing_sac_visual.yaml` are the two shipped configs that
exercise the working paths. See [Encoders](encoders.md) for the config fields
involved.

### Multi-agent RL

The current package is not documented or structured as a multi-agent RL framework.

### ROS 2 as a full package

SRL currently exposes a ROS 2 Python API, not a full `ament` ROS 2 package with launch files and install targets.

### ROS 2 deployment parity

ROS 2 deployment exists, but the current inference node still needs stronger alignment with the observation remapping semantics used in training.

## What this means in practice

The right mental model is:

- YAML is the core abstraction for model graph definition.
- CLI covers the common operational workflows.
- Python is still required for some advanced extension points.
- Isaac Lab and ROS 2 are important targets, but they still have active engineering gaps.

If you are using SRL for research, treat it as a capable alpha-stage framework with a strong declarative model layer rather than a finished no-code training platform.