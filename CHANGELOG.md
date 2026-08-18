# Changelog

All notable changes to SRL are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning as a target release model.

## [Unreleased]

### Added
- **`srl-train --save-best` / `train.save_best`** — opt-in tracking of the best
  `eval/score_mean` seen so far this run, saving a `best_*.pt` checkpoint
  (alongside, never replacing, the existing periodic `ckpt_*`/final `final_*`
  checkpoints) whenever eval produces a new run-best. Off by default. Motivated
  by a real PPO run where eval score peaked partway through training and then
  declined continuously for the rest of it (entropy collapse) — with no
  best-checkpoint mechanism, only the degraded final policy was ever saved and
  the actual peak was unrecoverable. Uses its own `CheckpointManager`
  (`max_keep=1`) rather than sharing the periodic manager's save/eviction
  FIFO, so ordinary periodic checkpoint churn can never evict a best
  checkpoint before something genuinely better replaces it. On `--resume`,
  seeds from any `best_*` checkpoint already on disk so a resumed run's first
  eval isn't automatically treated as "best." See `BestCheckpointTracker` in
  `srl/utils/checkpoint.py`.
- **PPO adaptive KL-based learning-rate schedule** (`PPOConfig.lr_schedule:
  "adaptive"`, off by default at `"fixed"`) — every minibatch, adapts the
  optimizer's LR from the measured `approx_kl`: shrinks it when KL exceeds
  `desired_kl * 2.0` (divide by `kl_lr_factor`, floored at `min_lr`), grows it
  when KL is under `desired_kl / 2.0` (multiply by `kl_lr_factor`, capped at
  `max_lr`). Modeled on rsl_rl PPO's `schedule="adaptive"` (its own default),
  which is why mjlab's reference PPO training path converges and holds on a
  task where SRL's PPO, with nothing bounding per-update aggressiveness
  across a long run, was found to peak early and then decline continuously
  for millions of steps despite `approx_kl` looking superficially healthy the
  whole time (healthy relative to no target being enforced at all).
  `PPOConfig.target_kl` (pre-existing) is a much weaker, one-shot safeguard —
  a same-epoch early stop that reacts only *after* one update already
  overshot, doing nothing to prevent the next one from being just as
  aggressive. See `PPO._adapt_lr` in `srl/algorithms/ppo.py`.
- **`srl-train --visualize`** — runs one extra single env in a background thread,
  doing live deterministic inference against the *current* (still-training) model
  weights and rendering it, while the main training envs keep running headless and
  unaffected. No snapshotting or reload: the viewer reads `agent.model`/
  `agent.predict` directly, so it always reflects the live weights. mjlab uses its
  own interactive, browser-based `ViserPlayViewer`; flat/goal/racecar envs open a
  `render_mode="human"` window. Not supported for isaaclab (a process hosts exactly
  one Isaac Sim render context, shared by every isaaclab env in it, so a second
  view-only env can't be added alongside headless training envs the way it can for
  mjlab or plain Gymnasium). See `srl/utils/live_viewer.py`.
- **Config validation in `ModelBuilder`** — missing `action_dim` on actor/critic head
  types that need it, missing `input_dim`/`input_shape` on mlp/lstm/cnn encoders, an
  encoder name declared twice with different configs, and unknown encoder/head types
  now raise a clear error *before* any tensor op runs, instead of failing deep inside
  the first forward pass with a bare `mat1 and mat2 shapes cannot be multiplied` or
  `nn.Linear(None, ...)` `TypeError`. The unknown-type error also now reports the real
  combined list of built-in + registered types, instead of the registry's own
  `Registered: []` (built-ins are special-cased in `builder.py` and never touch the
  registry, so that list was always empty unless a project registers custom types).
- **RNG state in checkpoints** — `python`/`numpy`/`torch` (+ CUDA) RNG state is now
  saved and restored, so `--resume` actually continues with the same
  exploration-noise/minibatch-shuffle sequence an uninterrupted run would have had,
  not just the same model/optimizer/step. Backward compatible: checkpoints saved
  before this change simply have nothing to restore.
- `matplotlib` is now a real base dependency (previously imported by the logger but
  declared nowhere, so a plain `pip install srl-rl` silently disabled
  `enable_plots=True`'s default PNG export with no indication why); a missing install
  now also emits a one-time warning as defense-in-depth.

### Fixed
- **SAC/DDPG/TD3 bootstrap target used `terminated OR truncated`** instead of true
  termination — `(1 - done) * next_q` was zeroed on every time-limit cutoff (i.e.
  nearly every episode on nearly every task), silently biasing Q-values low with no
  crash to surface it. `truncated` is now tracked as a genuinely separate field in
  `ReplayBuffer`/`GPUReplayBuffer`/`PrioritizedReplayBuffer` (not silently dropped),
  and the three env wrappers (`GymnasiumWrapper`/`GoalEnvWrapper`/`RacecarWrapper`)
  return true `terminated` instead of pre-combining it with `truncated`.
- **PPO/A2C/A3C re-evaluated a freshly sampled action instead of the one actually
  taken** — `AgentModel.forward()`'s actor branch always called the actor head's
  plain `forward()`, which draws a fresh `rsample()` every call, so
  `update()`'s stored-batch re-evaluation compared a *new, unrelated* action's
  log-prob against the recorded one. This silently broke PPO's clipped
  importance-sampling ratio and made A2C/A3C's policy gradient an estimator of the
  wrong quantity, and is also why the `entropy_coef` hyperparameter had zero effect
  (entropy was hardcoded to 0 via a `get_distribution()` check no actor head ever
  implemented). Fixed by wiring the actor heads' existing (but previously unused)
  `evaluate_actions(z, action)` through a new `actor_action` parameter on
  `AgentModel.forward()`.
- **`async_off_policy_runner.py`** called `buffer.add(...)` positionally, which
  silently swapped `next_obs`/`done` for `ReplayBuffer` (its real argument order
  differs from `GPUReplayBuffer`'s) — crashed `use_async=True` without
  `use_gpu_buffer`. Also read `env.action_space` where `IsaacLabWrapper` only exposes
  `.act_space`, and called `.sample()` directly, which isaaclab/mjlab's lightweight
  space doesn't implement. Fixed with keyword-arg buffer calls and a new shared
  `srl/utils/spaces.py::sample_action_space()`.
- **`Collector`** claimed to support both `RolloutBuffer` and `ReplayBuffer` but only
  actually worked with the former — paired with a `ReplayBuffer` it crashed
  immediately. Now dispatches on the buffer's actual type.
- **`HERReplayBuffer`** never stored per-timestep `done`, fabricating
  `dones = (rewards == 0.0)` for every sampled transition including the fraction that
  keeps the original (non-relabelled) goal — silently discarding real termination
  info whenever reward isn't purely sparse. Now stores real `done` and only uses the
  reward heuristic for the HER-relabelled fraction, where it's actually correct.
- **`apply_obs_remap`'s broadcast rule** had no case for "1 obs key, N>1 unnamed
  encoders" (only the N=1 rename case existed) — crashed the library's own flagship
  example configs (`halfcheetah_ppo.yaml`, `halfcheetah_sac.yaml`, two encoders, no
  explicit `input_name`) on the very first `agent.predict()`/`update()` call.
- **`ConvDecoderHead` construction** in the aux-loss (autoencoder) wiring passed a
  nonexistent `out_channels` kwarg — crashed any `aux_type: autoencoder` config
  (including the shipped `car_racing_ppo_visual.yaml`) at model-build time. Fixed to
  pass `output_shape` (the encoder's own `input_shape` — the decoder reconstructs
  what the encoder read in).
- Two `nn.ModuleDict.get(...)` calls in PPO's aux-loss wiring (`ModuleDict` has no
  `.get()`) — crashed constructing *any* `VisualPPOConfig` PPO agent for an aux-loss
  config before a single gradient step.
- `PrioritizedReplayBuffer._write()`'s override was missing the `truncated`
  parameter the base class's `add()` now always passes — every `add()` call raised
  `TypeError`.

### Changed
- CI now runs the *entire* `tests/` suite (previously only 2 of 7 test files), plus
  substantial new coverage: an end-to-end `srl-train` integration test against a fake
  mjlab/isaaclab-shaped env, one-gradient-step tests for all 6 algorithms, and
  add/sample round-trip tests for every buffer class.
- Lint cleaned up (168 `ruff` errors → 0, 58 files reformatted with `black`);
  `ruff`/`black` versions and an explicit `[tool.ruff.lint] select` are now pinned in
  `pyproject.toml` so the enforced rule set can't silently drift with the installed
  tool version.

- **mjlab integration** — `--env mjlab:<task>` / `env_type: mjlab`, a lighter-weight
  alternative to the Isaac Lab integration for projects using
  [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo-Warp, GPU-batched) instead of
  real Isaac Sim. Task ids resolve through mjlab's own registry (auto-discovered from
  any project's `mjlab.tasks` entry point, the same mechanism `import mjlab` itself
  uses), and the env is wrapped with the existing `IsaacLabWrapper` — no new wrapper
  class needed, the two envs' APIs are close enough. See
  `docs/integrations/mjlab.md`.

### Fixed
- **Double obs-remap `KeyError`** — `srl/cli/train.py`'s rollout loops remap the obs
  dict to encoder names before calling `agent.predict()`/`model()`, and
  `AgentModel.forward()` remapped it *again* internally, expecting the *original* raw
  obs key — which no longer existed once the first pass had renamed it away. Any YAML
  config using an encoder's explicit `input_name` hit this on every forward pass.
  `AgentModel._remap_obs_dict` now short-circuits when the incoming dict's keys
  already exactly match the encoder names.
- **Isaac Lab/mjlab eval action shape** — `_evaluate_agent` unconditionally squeezed
  the batch dimension off actions for single-env evaluation. isaaclab/mjlab envs
  always expect a batched `(1, action_dim)` action, even at `num_envs=1`; the squeeze
  produced a 1-D action and the env's action manager raised `IndexError`.
- **Isaac Lab/mjlab off-policy warmup** — random-action warmup called `.sample()` on
  the action space directly, which isaaclab/mjlab's lightweight `Box`-style space
  dataclass doesn't implement (`AttributeError`); the fallback also needs to handle
  unbounded (`[-inf, inf]`) action spaces, which those envs declare by convention
  (action terms do their own internal scale/clip) and which `np.random.uniform` can't
  sample directly (`OverflowError`). Falls back to `[-1, 1]` on any non-finite bound.
  Separately, `env.action_space` on these envs is already the *batched*
  `(num_envs, action_dim)` space, not one env's space — stacking N samples of it
  produced an `(N, num_envs, action_dim)` array instead of `(num_envs, action_dim)`.
  `IsaacLabWrapper` now also exposes `single_act_space` (falling back to `act_space`
  for envs that don't distinguish the two) for exactly this per-env sampling case.

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