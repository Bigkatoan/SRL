# Changelog

All notable changes to SRL are documented in this file.

The format follows Keep a Changelog and the project uses Semantic Versioning as a target release model.

## [Unreleased]

### Phase 2: Python API, strict configs, one registry

**Python API**
- Every algorithm has `from_config(yaml, env=...)`, `learn(total_timesteps, env=None, eval_env=None, callbacks=None, log_dir=None, progress=False)`, `predict(obs) -> (action, state)` (numpy in, env-unit action out), `save(dir)` and `Cls.load(dir_or_checkpoint, env=None)`. `srl-train` checkpoints load with `Cls.load(path)`.
- New `srl.training` package: `OnPolicyTrainer`, `OffPolicyTrainer`, `Evaluator`, callbacks (`CheckpointCallback`, `EvalCallback`, `EarlyStopping`, `LogCallback`) with ordered hooks. `srl-train` and `from_config` share one construction path (`srl.core.agent_io.build_agent_from_experiment`); the CLI is a thin layer over `agent.learn()`.
- `BaseAgent` / `OnPolicyAgent` / `OffPolicyAgent` hold the shared agent code. `agent.save(dir)` also writes `config.yaml`, the fully resolved experiment (`srl.config.dump_resolved_config`); `srl-train` writes the same file to `<logdir>/<run>/config.yaml`.

**Checkpoint format v1, `srl.load_policy`, deployment**
- `agent.save(dir)` and every `srl-train` checkpoint (`ckpt_<step>/`, `final_<step>/`, `best_<step>/`) write one directory format, `srl-agent` v1: `meta.json` (format/versions, obs/action spec, normalizers), `config.yaml` (resolved experiment, re-validated on load), `model.safetensors` (strict key check), `normalizers.json`, optional `train_state.pt` (optimizers, RNG, counters) and `replay_buffer/`. Nothing in it needs pickle: all `.pt` files load with `torch.load(weights_only=True)`. Missing files, a foreign `format` or a newer `format_version` raise `CheckpointFormatError`.
- v0 directories (`agent.pt`) and single-file `*.pt` checkpoints still load (`weights_only=False` with a `LegacyCheckpointWarning`).
- `srl.load_policy(path, device="cpu")` -> `srl.Policy`: inference-only (`policy(obs)` = `agent.predict(obs, deterministic=True)[0]`), from an agent directory, an `srl-train` checkpoint directory (best / final / latest) or a legacy file; `Policy.export_torchscript(path)`; `python -m srl.policy info|export` (`srl-policy`).
- ROS 2: `srl.ros2.rl_node` loads the policy with `srl.load_policy` (`policy_path` parameter; topics from the checkpoint's `ros2:` block), so `auto`-dim configs, normalizers and action rescaling work; the inference core is the ROS-free `PolicyRunner`. `resolve_msg_type` accepts `pkg/Msg` names. Template package updated.
- Periodic `srl-train` checkpoints no longer embed the replay buffer (it was rewritten, multi-GB, on every save); `train.checkpoint_replay_buffer: true` restores the old behaviour. `CheckpointManager(..., include_replay_buffer=...)`.
- `agent.save(dir, include_train_state=False)` writes an inference-only checkpoint. `AgentModel.act` works with dict-returning heads.
- `CheckpointManager` no longer swallows safetensors write errors.
- Auxiliary-loss config diagnostics are `warnings.warn` (`UserWarning`); a train key of another algorithm says to remove it.

**Strict configs (`srl.config`)**
- `load_experiment(path_or_dict, algo=None)` validates the whole YAML (model, env keys, `train:`); unknown keys, wrong types, bad enums/ranges raise one `ConfigError` listing every problem, with "did you mean" hints. Config dataclasses validate in `__post_init__`. Files carry `schema_version: 1`.
- `input_dim: auto` / `input_shape: auto` / `action_dim: auto` are resolved from the env's spaces (the env is now built before the model); explicit values are cross-checked against the env. The simple shipped configs (Pendulum, Hopper/HalfCheetah/Walker2d, Fetch) use `auto`.
- `srl.config.build_algo_config` / `normalize_offpolicy_schedule` / `build_runner_config` are the only YAML -> config builders (the lenient CLI copies are gone). `train.save_interval` is a trainer key.
- One algorithm registry: `srl.registry.AlgorithmRegistry` (`AlgorithmSpec`); built-ins register like plugins, entry-point groups `srl.algorithms` / `srl.encoders` / `srl.heads` / `srl.envs`. `srl.algorithms.ALGORITHMS` is a read-only view of it; plugin env types are built through `EnvRegistry`.

**Removed / deprecated**
- Removed: `LossRegistry`/`register_loss` (losses are fixed per algorithm), `RecurrentPPOConfig`, `srl.core.BasePolicy`, `srl.envs.collector.Collector`, `srl.training.make_algo_config`, the private `srl.cli.train._run_*` / eval / config helpers (public equivalents in `srl.training` / `srl.config`).
- Deprecated (warn, then ignored or folded): top-level `losses:` blocks (never read; removed from the shipped YAMLs), `frame_stack: 1` (any other value is an error), `start_steps` -> `learning_starts`, `update_every` -> `train_freq`, `train.env_id`/`env_type` -> top level, `AsyncRunnerConfig.prefill_steps`/`queue_maxsize` (ignored), `agent.predict(tensor_dict)` 4-tuple (use `agent.act`), `agent.save("file.pt")`, guessing the algorithm from the file name. `srl.training.ConfigurationError` is an alias of `srl.config.ConfigError`.

**Fixed**
- PPO `target_kl` applied one more gradient step after the KL limit was exceeded; it now stops before stepping (SB3/CleanRL order).
- Gaussian actor heads created a second, never-trained `log_std_param` next to the distribution's real `dist.log_std` (old checkpoints still load).
- `pendulum_ppo.yaml` / the PPO Pendulum learning test: lr 3e-4, batch 256 (was rl-zoo's 1e-3 / 64, which without gSDE blew up approx_kl once the log-std shrank and collapsed the learned policy). The learning test runs seeds 0-2 and asserts on the mean of the last 3 evals.
- `EarlyStopping` monitors `eval/score_mean` and actually stops training; `srl-train` requires `algo:` (YAML) or `--algo`.

### Breaking / behaviour changes (correctness audit, planned 0.3.0)
- Continuous actions now live in a normalized `[-1, 1]` domain: `RescaleActionWrapper` maps them to env bounds at `env.step`; warmup samples uniformly in `[-1, 1]`.
- Gaussian actor heads default to state-independent std (`state_dependent_std: false`); old checkpoints trained with the implicit state-dependent std need `state_dependent_std: true`.
- PPO enables `normalize_obs` / `normalize_reward` by default and `clip_range_vf` defaults to `None` (= no value clipping).
- `eval/success_mean` is the per-episode final-step success rate (was averaged over all timesteps).

### Fixed (correctness audit)
- PPO trained only its heads: encoders were always detached from the RL loss and stayed at random init.
- On-policy training with `n_envs=1` never reset the env after the first episode.
- n-step replay dropped (n-1)/n of transitions; it is now a per-env sliding window (CPU and GPU), and SAC/TD3/DDPG bootstrap with per-transition `gamma**k`.
- GPU replay buffer mixed n-step returns across envs and episodes.
- SAC deterministic actions returned the pre-tanh mean; now `tanh(mean)`. PPO/A2C/A3C deterministic actions are the mode clipped to bounds.
- CPU vector envs dropped the true terminal observation on auto-reset (wrong truncation bootstrap); added `info["final_obs"]` and `srl.envs.get_final_obs`.
- TD3/DDPG actor-only encoders never trained; TD3 trained in eval mode after the first `predict()`.
- HER relabelled transitions were marked terminal when reward == 0; "future" goals now sampled from t+1..T.
- A3C ignored `truncated` (time-limit envs never reset) and seeded workers by rank only.
- PPO value clipping used max of batch means instead of elementwise max; GAE ran over empty slots of partial rollouts.
- Seeding: `seed_offset` ignored and action spaces unseeded; `AsyncOffPolicyRunner` warmup shape and resume step; DDPG noise shared across envs.

### Added
- `srl.utils.set_random_seed`, `--deterministic` flag.
- `normalize_obs` / `obs_norm_clip` running normalizer for SAC/TD3/DDPG (on in Fetch configs); PPO VecNormalize-style obs/reward normalization.
- `use_per` prioritized replay (importance weights, priority updates) for SAC/TD3/DDPG.
- Fetch push/slide/pick-and-place configs enable HER (future, k=4); `pendulum_ppo.yaml` uses rl-zoo-like hyperparameters.
- ~120 regression tests, config-parse matrix over every YAML, and `slow` learning tests (SAC/PPO Pendulum, FetchReach+HER).
- Phase 1 benchmark harness: rl-zoo MuJoCo configs (`configs/benchmarks/mujoco/`), resumable matrix runner `scripts/benchmark/run_matrix.py`, `report.py` (mean ± std, IQM + bootstrap CI, learning curves vs SB3 zoo / CleanRL in `reference.yaml`), detached launcher `launch_phase1.sh`, and the `Benchmarks` docs page.

### Changed
- Off-policy `srl-train` averages training metrics over `train.metrics_log_interval` env steps (default 1000) before logging, instead of logging ~13 scalars per gradient step (~2 GB of TensorBoard/JSONL/CSV and >1 GB RSS per 1M-step run). Set it to 1 for the old behaviour.

### Fixed
- **PPO GAE off-by-one in `RolloutBuffer.compute_returns_and_advantages`** —
  `self._dones[t]` (as written by every real caller's own `add()` sequence,
  e.g. `srl.cli.train._run_on_policy`) means "did stepping slot t's obs end
  the episode", i.e. whether slot t+1's obs is a fresh reset. The GAE
  recursion gated slot t's bootstrap on `self._dones[t + 1]` instead of
  `self._dones[t]` — correct for a reference implementation (e.g. CleanRL)
  that stores dones the OPPOSITE way, but this class's own convention was
  never adjusted for when the formula was ported in. Net effect, verified
  with a hand-traced example: the step right before a real termination
  wrongly bootstraps through `self._values[t+1]` — a completely unrelated,
  freshly-reset episode's first-state value — while the step before *that*
  gets its (harmless, legitimate) bootstrap wrongly zeroed. Not
  mjlab/isaaclab-specific — this fires any time an episode boundary falls
  inside a rollout window, on every env type PPO trains on, and JAVIS's own
  training logs show early-training episode lengths (15-53 steps) well
  under typical `n_steps` (24-2048) configs, so it fires constantly rather
  than rarely. Also fixes a related latent bug: the boundary case's
  `next_non_terminal` always defaulted to "not terminal" (`last_dones`
  parameter existed but no real caller ever passed it), now correctly
  reads the buffer's own last-written `done` flag instead. See
  `srl/core/rollout_buffer.py`'s updated docstring for the full derivation
  and `tests/test_rollout_buffer_done_indexing.py` for a regression test
  that fails against the old code and passes against the fix.
- **mjlab/isaaclab `auto_reset` silently corrupting time-limit
  bootstrapping** — `_make_cli_env` never set `env_cfg.auto_reset = False`
  for either backend, so `env.step()`'s returned observation for any
  sub-env that just terminated OR timed out was already the NEXT episode's
  first frame (mjlab's own docs: with `auto_reset=True`, "the returned
  observation is the post-reset state"). Fine for choosing the next action,
  wrong for value bootstrapping: PPO's GAE and SAC/DDPG/TD3's Q-target both
  ended up bootstrapping off a completely unrelated, freshly-reset
  episode's state at every time-limit truncation, not just at the (rarer)
  off-by-one interaction above. `IsaacLabWrapper` now probes (once, before
  construction) whether the concrete env class's `reset()` accepts
  `env_ids` (mjlab's does, verified against its source; real Isaac Lab is
  assumed to as well per mjlab's own "mirrors the API closely enough"
  claim, but this repo has no real Isaac Lab install to verify that
  against, so it's a genuine capability probe, not an assumption) and, only
  when true, disables the env's own auto-reset and does the equivalent
  partial reset itself immediately after every `step()`, exposing the true
  pre-reset observation via `info["true_final_obs"]`. `_run_on_policy` uses
  it to add a `gamma * V(true_final_obs)` correction to the reward at every
  truncation (and now treats `terminated | truncated` as a GAE chain-break,
  matching the off-by-one fix above); `_run_off_policy` and
  `AsyncOffPolicyRunner` (both its sync and async collector loops) use it
  as the stored `next_obs` for off-policy bootstrapping. Env types whose
  `reset()` doesn't support `env_ids` (or that never reached this bug in
  the first place — a single-env `GymnasiumWrapper` never auto-resets
  internally, so its returned observation on termination was already
  correct) are entirely unaffected: this degrades to the exact prior
  behavior rather than guessing wrong.
- **PPO entropy bonus with nothing pulling it back down** — real 40M-step
  verification run of the two fixes above (`javis_mjlab_ppo.yaml`,
  `lr_schedule="adaptive"` already on): peaks far higher than the original
  bug (10.05 @ step 11M vs. 3.18 @ step 10M) and holds it much longer, but
  still declines afterward, ending BELOW the original bug's own eventual
  plateau (0.77 vs. 1.10-1.23 @ step 40M). Traced directly: `ppo/entropy`
  climbs continuously and monotonically for the ENTIRE run (-2.76 @
  ~100k steps to +1.45 @ 40M, never turning over), the opposite of the
  earlier entropy-COLLAPSE failure mode `lr_schedule="adaptive"` was built
  for, but the same underlying shape of problem: nothing bounds a
  monotonic drift over a long enough run. `log_std` (`DiagonalGaussian`)
  confirmed correctly clamped to `[log_std_min, log_std_max]` and nowhere
  near that ceiling -- the entropy bonus's constant, one-directional pull
  (unlike SAC's auto-tuned `alpha`, which targets a specific entropy level
  and can push either direction) just kept winning against the policy
  gradient's counter-pressure. New `PPOConfig.entropy_coef_anneal_steps`
  (default `None` -- off) linearly anneals `entropy_coef` down to
  `entropy_coef_final` (default `0.0`) over that many GRADIENT steps,
  wired through `LossComposer`'s existing (previously unused for this
  term) `schedule="linear_decay"` support. Re-verified with a second real
  40M-step run (`entropy_coef` 0.005 -> 0.0005 annealed over the first
  ~65% of the budget): a much longer-sustained elevated plateau (average
  ~4.2 across steps 6M-28M, vs. the un-annealed fix's brief 10-19M
  spike-then-crash), then settles into a genuinely STABLE second plateau
  for the last 12M steps (1.45-1.85, tight band, not still declining at
  step 40M the way the un-annealed run was) -- ends at 1.45, ~2x the
  un-annealed fix's final value and above the original bug's own
  plateau. Also found (while tracing the entropy growth) a real but
  likely non-causal bug in `GaussianActorHead`: with
  `state_dependent_std=False` it computes its own `log_std_param` and
  passes it to `self.dist(mean, log_std)`, but `DiagonalGaussian`
  silently discards that argument and substitutes its own separate
  internal `log_std` parameter instead -- `GaussianActorHead.
  log_std_param` is genuinely dead weight, never receiving a gradient.
  Not fixed in this PR (the actually-used parameter works correctly
  independent of this); worth a small, separate follow-up.

### Added
- **SAC temperature floor** (`SACConfig.min_alpha`, permissive `1e-8` default —
  off in effect for existing configs) — clamps the auto-tuned temperature
  (`log_alpha`) after each alpha-optimizer step so it can't collapse below a
  configured floor. `log_alpha` was otherwise an unclamped free parameter:
  found on a real 10M-step SAC (FlashSAC config: `batch_norm` +
  `weight_norm_projection` + `lr_alpha=3e-5`, already a fix for *immediate*
  alpha collapse) run against JAVIS's mjlab balance task that alpha still
  eventually collapsed (to ~3e-4, the same order of magnitude as an
  unfixed-baseline's full collapse) around step 6-7M, immediately followed
  by numerical-explosion episode returns (as extreme as -7.1e6) from a
  known unclamped-reward-term physics-divergence exploit that a
  fully-collapsed alpha's missing entropy regularization lets the actor
  find — and it recurred a second time later in the same run. The 10x-lower
  `lr_alpha` delayed collapse (it was immediate in the unfixed baseline)
  but did not prevent it over a long enough horizon. See `SAC.update()`'s
  temperature-update step in `srl/algorithms/sac.py`.
- **PPO adaptive KL-based learning-rate schedule** (`PPOConfig.lr_schedule:
  "adaptive"`, off by default at `"fixed"`) — every minibatch, adapts the
  optimizer's LR from the measured `approx_kl`: shrinks it when KL exceeds
  `desired_kl * 2.0` (divide by `kl_lr_factor`, floored at `min_lr`), grows it
  when KL is under `desired_kl / 2.0` (multiply by `kl_lr_factor`, capped at
  `max_lr`). Modeled on rsl_rl PPO's `schedule="adaptive"` (its own default,
  and what mjlab's own rsl_rl-based reference PPO training path uses on this
  same task). `PPOConfig.target_kl` (pre-existing) is a much weaker, one-shot
  safeguard — a same-epoch early stop that reacts only *after* one update
  already overshot, doing nothing to bound the next one. Real-GPU-verified:
  substantially raises and delays the peak `eval/score_mean` a long PPO run
  reaches before declining (e.g. 1.87→3.18 peak height, 3.5M→10M steps to
  peak, across configs tested) — but did **not**, even combined with also
  matching rsl_rl's actor std configuration, fully eliminate an eventual
  decline within a 20-40M step budget on the task this was found on; see
  [Bigkatoan/SRL#39](https://github.com/Bigkatoan/SRL/pull/39) for full
  results and honest caveats (including that rsl_rl's own reference config
  trains for ~295M steps, ~15x what was tested here). Naively setting
  `max_lr` to rsl_rl's raw hardcoded constant (`1e-2`, 10x this task's tuned
  base `lr`) was actively *worse* than not using this feature at all — cap
  `max_lr` near your own tuned `lr` unless you've independently verified a
  higher ceiling is safe for your setup. See `PPO._adapt_lr` in
  `srl/algorithms/ppo.py`.
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

### Infrastructure
- **Packaging:** PEP 639 `license = "MIT"` + `license-files` (setuptools>=77);
  version is now derived from git tags by setuptools-scm (`srl/_version.py`,
  `srl.__version__` still works); removed the `racecar-gym` direct-URL
  dependency that PyPI rejects (install it manually, see installation docs);
  added Python 3.13 and `Typing :: Typed` classifiers; dev extras split into
  `test`, `lint`, `docs`, `tracking` (`dev` = all of them).
- **CI (`test.yml`):** CPU-only torch wheels, uv with caching, read-only
  token permissions, concurrency cancellation, Python 3.10–3.13 on Ubuntu plus
  macOS/Windows smoke jobs, coverage upload to Codecov (`fail_under = 65`).
  New nightly / `run-slow`-label `learning` job runs `pytest -m slow`.
- **Lint:** ruff + black now also cover `examples/` and `templates/`; added a
  non-blocking mypy job (baseline ~281 errors). `ruff format` is configured
  but black stays the enforced formatter until the migration PR.
- **Docs:** built with `-W --keep-going` on every PR; deployed through GitHub
  Pages actions without publishing `.doctrees`. Fixed the broken
  `--save-best` anchor, the `.cuda()`-only index example and the non-existent
  `CheckpointManager.load(step=...)` call in the quickstart.
- **Releases:** tag-driven `release.yml` (build, `twine check --strict`,
  TestPyPI → PyPI via trusted publishing, GitHub Release); see `RELEASING.md`.
- **Tests:** `tests/conftest.py` registers `slow`/`gpu`/`integration`
  markers (`--strict-markers`, 300 s per-test timeout);
  `tests/test_configs_parse.py` builds every shipped config and checks it
  against its env; `tests/test_learning.py` adds slow end-to-end learning
  gates (SAC/PPO Pendulum, SAC+HER FetchReach).
- **Community:** `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `CITATION.cff`, issue forms, PR template, pre-commit config and Dependabot.
- **Scripts:** removed the stale root `run_full_matrix_benchmark.sh` wrapper
  and `scripts/run_deep_env_tests.sh` (ran deleted tests); the matrix
  benchmark script no longer hard-codes a dev-machine venv path.

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