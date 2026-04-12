# Changelog

All notable changes to SRL are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning as a target release model.

## [Unreleased]

## [0.2.0] - 2026-04-12

### Added
- **Encoder optimizer fix** — SAC, DDPG, and TD3 now use a dedicated third optimizer
  (`encoder_optimizer`) so the encoder is no longer updated twice per gradient step
  (once via `actor_optimizer`, once via `critic_optimizer`).  Eliminates the effective
  double learning-rate that caused distribution-shift collapse at ~10 500 steps in
  visual tasks.
- **`encoder_update_freq`** — new field on `SACConfig`, `DDPGConfig`, and `TD3Config`
  (default `1`); `VisualSACConfig` defaults to `2` to further stabilise pixel encoders.
  The encoder optimizer only steps every N critic updates.
- **`encoder_optimize_with_critic`** — new boolean on `VisualSACConfig` (default `True`).
  Set to `False` to stop encoder gradients from flowing through the critic loss and rely
  solely on the aux loss.
- **Expanded `aux_loss_type`** — eight modes are now supported on `VisualSACConfig`:
  `none`, `ae`, `vae`, `curl`, `byol`, `drq`, `spr`, `barlow`.
- **`AsyncRunnerConfig`** dataclass — controls the new asynchronous off-policy runner.
  Fields: `use_async`, `use_gpu_buffer`, `prefill_steps`, `queue_maxsize`.
- **`AsyncOffPolicyRunner`** (`srl.runners`) — decouples data collection from gradient
  updates. The collector runs on the main thread (required for Isaac Lab CUDA-context
  safety); the trainer runs on a daemon thread with its own CUDA stream.
- **`GPUReplayBuffer`** (`srl.core.gpu_replay_buffer`) — pre-allocated CUDA circular
  buffer. Accepts CUDA tensors directly via a dedicated non-blocking copy stream, giving
  zero host↔device copies when Isaac Lab already lives on GPU. Supports dict-obs,
  n-step returns, and CPU serialisation for checkpointing.
- New aux loss functions: `vae_loss`, `drq_aug_loss`, `spr_loss`, `barlow_twins_loss`
  (all exported from `srl.losses`).
- New network heads: `VAEHead`, `LatentTransitionModel` (exported from
  `srl.networks.heads.aux_head`).
- Documentation pages: `docs/async_runner.md`, `docs/gpu_replay_buffer.md`.

### Changed
- `actor_optimizer` and `critic_optimizer` in SAC/DDPG/TD3 now contain only head
  parameters; encoder parameters live exclusively in `encoder_optimizer`.
- Checkpoint format extended with `encoder_optimizer_state` and `encoder_update_counter`
  keys; old checkpoints without these keys are loaded gracefully (backward compatible).
- Version bump: `0.1.0` → `0.2.0`.

### Fixed
- Structured CLI documentation page.
- Limitations page for current declarative and deployment boundaries.
- Structured ROS 2 YAML schema support in the config layer.
- Shared observation remapping utility used across training, runtime model execution, and ROS 2 inference.
- Initial GitHub Actions workflows for tests and linting.
- Top-level package imports are now lazy, so CLI help paths do not fail early on heavyweight runtime imports.
- ROS 2 inference now uses the same observation remapping rules as the training/runtime path.
- `python -m srl.cli.train --help` no longer fails immediately because of eager algorithm imports.
- `python -m srl.cli.visualize --help` no longer fails immediately because of eager utility imports.

## [0.1.0] - 2026-04-12

### Added
- Initial release of SRL with PPO, SAC, DDPG, TD3, A2C, and A3C.
- YAML-driven model building with flow graphs, encoders, heads, and multimodal support.
- Isaac Lab integration, benchmark scripts, checkpointing, and ROS 2 Python API.

[Unreleased]: https://github.com/Bigkatoan/SRL/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Bigkatoan/SRL/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Bigkatoan/SRL/releases/tag/v0.1.0