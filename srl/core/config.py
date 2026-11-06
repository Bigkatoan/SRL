"""Algorithm and model hyperparameter dataclasses.

Every config validates its enum (``Literal``) fields and numeric ranges in
``__post_init__`` (see :func:`srl.config._validate.validate_constraints`), so
``PPOConfig(gamma=1.5)`` or ``DDPGConfig(action_noise="uo")`` raise a
:class:`~srl.config.ConfigError` immediately instead of misbehaving at
training time. Building one from a YAML ``train:`` block should go through
:func:`srl.config.build_algo_config`, which additionally rejects unknown keys.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from srl.config._validate import validate_constraints

LrSchedule = Literal["fixed", "adaptive"]
ActionNoise = Literal["gaussian", "ou"]
HerStrategy = Literal["future", "final", "episode", "random"]
AuxLossType = Literal["none", "ae", "vae", "curl", "byol", "drq", "spr", "barlow"]
AugmentationMode = Literal["drq", "curl", "aggressive"]

# Legacy off-policy schedule aliases. Kept as fields so existing Python code
# constructing e.g. ``SACConfig(start_steps=...)`` keeps working, but
# srl.config.build_algo_config() folds them into the canonical names
# (learning_starts / update_after / train_freq) with a deprecation warning,
# and srl.config.normalize_offpolicy_schedule() is the one place that
# resolves them for configs built directly in Python.
_ALIAS = {"deprecated_alias": True}


def _check_lr_bounds(self) -> list[tuple[str, str]]:
    errors = []
    if getattr(self, "min_lr", 0) > getattr(self, "max_lr", float("inf")):
        errors.append(("min_lr", f"must be <= max_lr ({self.max_lr!r}), got {self.min_lr!r}"))
    lo, hi = getattr(self, "min_entropy_coef", None), getattr(self, "max_entropy_coef", None)
    if lo is not None and hi is not None and lo > hi:
        errors.append(("min_entropy_coef", f"must be <= max_entropy_coef ({hi!r}), got {lo!r}"))
    return errors


class _Validated:
    """Mixin: run SRL's enum/range validation after dataclass ``__init__``."""

    __srl_check__ = _check_lr_bounds

    def __post_init__(self) -> None:
        validate_constraints(self)


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm configs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig(_Validated):
    lr: float = 3e-4
    n_steps: int = 2048  # steps per env per rollout
    num_envs: int = 1
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None  # None = no value-function clipping (SB3 default)
    entropy_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    use_fp16: bool = False
    # VecNormalize-style normalisation (on by default, as in CleanRL's
    # ppo_continuous_action / rl-zoo MuJoCo PPO). `normalize_obs`: running
    # mean/var per flat (1-D per sample) observation key -- image keys are
    # never touched -- updated on every collected observation, applied inside
    # PPO.predict() (so eval / deployment see the same transform) and stored
    # normalised in the rollout buffer. `normalize_reward`: rewards are
    # divided by the running std of the discounted return (not the reward
    # itself, and not mean-centred). Both statistics are saved in
    # checkpoints. Episode returns logged by the trainer stay raw.
    normalize_obs: bool = True
    normalize_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    norm_epsilon: float = 1e-8
    # Adaptive KL-based learning-rate schedule (off by default -- `lr` stays
    # fixed for the whole run unless `lr_schedule="adaptive"`). Modeled on
    # rsl_rl's PPO (`schedule="adaptive"`, the default there), which applies
    # this every minibatch for the entire run: `target_kl` above is only a
    # same-epoch early stop (breaks out of *this* update's minibatch loop
    # once KL gets too large), a much weaker, one-shot safeguard that does
    # nothing to prevent the next update from being just as aggressive.
    # Nothing bounding per-update aggressiveness across the whole run is
    # what let a real 20M-step PPO run on JAVIS's mjlab balance task drift
    # into a declining policy over millions of steps with `approx_kl`
    # staying inside a superficially "healthy" band throughout -- healthy
    # relative to no target at all, not to a `desired_kl` actually being
    # enforced. See srl/algorithms/ppo.py's PPO.update() for where this is
    # applied.
    lr_schedule: LrSchedule = "fixed"
    desired_kl: float = 0.01
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    kl_lr_factor: float = 1.5
    # Entropy coefficient annealing (off by default -- `entropy_coef` stays
    # fixed for the whole run unless `entropy_coef_anneal_steps` is set).
    #
    # CORRECTION to this field's original design rationale: it was added
    # after real runs against JAVIS's mjlab balance task showed `ppo/
    # entropy` climbing continuously for tens of millions of steps, read
    # (at the time) as an unbounded ENTROPY EXPLOSION with nothing
    # counteracting the entropy bonus's constant upward pull. That
    # reading was wrong -- traced (both by re-deriving from source and by
    # a direct empirical check constructing a real PPO agent and
    # inspecting `update()`'s own return value) to a sign bug in
    # INTERPRETING the logged metric, not in the mechanism itself:
    # `entropy_loss(ent) = -ent.mean()` (srl/losses/rl_losses.py) is what
    # gets logged as `ppo/entropy` (LossComposer.compute() -> Logger.
    # record_metrics() is a straight, unmodified passthrough of that
    # value) -- i.e. `ppo/entropy` in every log/metrics.jsonl is the
    # NEGATIVE of the true mean entropy, not the entropy itself.
    # Confirmed against real data too: this task's actor
    # (log_std_init=0.0, action_dim=2) has a theoretical initial entropy
    # of ~2.838 (2*0.5*ln(2*pi*e) at std=1) -- every real run's very
    # first logged `ppo/entropy` value is ~-2.76, matching -2.838, not
    # +2.838. So the real trajectory (climbing -2.76 -> +6.6 in raw log
    # terms) is true entropy DECLINING from +2.76 toward -6.6 -- this is
    # entropy COLLAPSE (policy std shrinking toward log_std_min), the
    # ORIGINAL failure hypothesis this file's `lr_schedule="adaptive"`
    # comment above describes, not a distinct "opposite" failure mode.
    # The entropy_coef MECHANISM itself was never sign-confused (a
    # positive `entropy_coef` genuinely pushes true entropy up, standard
    # usage) -- only the earlier narrative describing what was
    # happening to it was backwards. Consequently, LOWERING
    # `entropy_coef_final` toward/to 0.0 (tested on real 60M-step runs:
    # 0.0005 held up longest, hard 0.0 collapsed fastest and regressed
    # below every other configuration tried on this task, 0.0001 was
    # in between) makes sense under the corrected picture too --
    # removing the entropy bonus removes protection AGAINST collapse,
    # not "removes a runaway pull" as originally framed. Left in place
    # (rather than removed) as a still-useful, simpler alternative to
    # `target_entropy` below for configs that don't need the extra
    # complexity; a NON-zero `entropy_coef_final` is what actually
    # helps here, and only real GPU verification tells you how much.
    #
    # When `entropy_coef_anneal_steps` is set, `entropy_coef` is linearly
    # annealed from its configured value down to `entropy_coef_final`
    # over that many GRADIENT steps (`self._global_step` below -- one
    # increment per minibatch, i.e. the same unit `lr_schedule=
    # "adaptive"`'s own internal bookkeeping already uses, NOT env steps
    # -- `n_epochs * total_env_steps / batch_size` if you need to
    # convert from an env-step budget), then held at `entropy_coef_final`
    # for the rest of the run. See `PPO.__init__`'s composer setup for
    # where this is wired through `LossComposer`'s `schedule=
    # "linear_decay"` support.
    #
    # Ignored when `target_entropy` (below) is set -- the two mechanisms
    # are mutually exclusive; `target_entropy` takes over `entropy_coef`
    # entirely in that case.
    entropy_coef_final: float = 0.0
    entropy_coef_anneal_steps: int | None = None
    # Adaptive, TARGET-seeking entropy coefficient (off by default --
    # `None`). A fixed or annealed `entropy_coef` (above) is a
    # one-directional dial: it can only ever push true entropy up, never
    # pull it back down, so it either isn't enough to prevent collapse
    # (too low) or, in principle, could over-correct (too high) -- there
    # is no setting that automatically corrects in BOTH directions the
    # way SAC's auto-tuned `alpha` does for its own entropy term. This
    # ports that same idea to PPO: `entropy_coef` becomes a learned
    # parameter (`log_entropy_coef`, exponentiated to stay positive),
    # updated via its own small optimizer toward whatever value makes
    # measured entropy track `target_entropy` -- symmetric, self-
    # correcting, no manual floor/schedule tuning. Dual-ascent update
    # (same shape as SAC's temperature loss, re-derived here rather than
    # copied verbatim since SAC's exact formula assumes its own sign/
    # target conventions): `loss = log_entropy_coef * (entropy.detach()
    # - target_entropy)`, minimized by gradient descent -- entropy above
    # target shrinks the coefficient (less push, it's already high
    # enough), entropy below target grows it (push harder). Verify: at
    # entropy == target_entropy the loss gradient is zero, a genuine
    # equilibrium, unlike a fixed/annealed coefficient which has none.
    # `target_entropy` should be a true-entropy value (this task's own
    # data: score was still healthy while true entropy was declining
    # from its ~2.8 init through roughly 0, and degraded once it fell
    # below there toward -1 and beyond -- 0.0 is a reasonable starting
    # target, corresponding to roughly std~=0.24 per action dimension
    # for a 2-action-dim Gaussian, not yet verified as optimal on a real
    # long run). `min_entropy_coef`/`max_entropy_coef` bound the learned
    # coefficient itself (same safety-net role as SAC's `min_alpha`),
    # and `log_std_min`/`log_std_max` (actor head config, NOT here)
    # remain a direct, mechanism-independent bound on worst-case policy
    # std regardless of whether this or the coefficient tuning above is
    # what's actually controlling it day to day.
    target_entropy: float | None = None
    entropy_coef_lr: float = 1e-3
    min_entropy_coef: float = 1e-6
    max_entropy_coef: float = 0.05


@dataclass
class A2CConfig(_Validated):
    lr: float = 7e-4
    n_steps: int = 5
    num_envs: int = 1
    # Not read by A2C.update() -- A2C always takes one gradient step over the
    # full rollout (n_steps * num_envs transitions), matching the canonical
    # algorithm. Kept only for backward compatibility with existing code
    # that constructs A2CConfig(batch_size=...) explicitly.
    batch_size: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0
    entropy_coef: float = 0.01
    vf_coef: float = 0.25
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-5
    use_fp16: bool = False


@dataclass
class A3CConfig(_Validated):
    lr: float = 1e-4
    n_steps: int = 20
    gamma: float = 0.99
    gae_lambda: float = 1.0
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 40.0
    n_workers: int = 4
    batch_size: int = 20
    # Base seed; worker `rank` uses `seed + rank` (torch RNG + env.reset).
    # Not a train: key of its own -- it is the trainer-level `train.seed`
    # (set by from_config / srl-train, and by learn(seed=...)).
    seed: int = field(default=0, metadata={"yaml": False})


@dataclass
class SACConfig(_Validated):
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # soft target update coefficient
    action_dim: int = 0  # required for automatic target entropy
    target_update_interval: int = 1
    learning_starts: int = 10_000
    start_steps: int | None = field(default=None, metadata=_ALIAS)  # -> learning_starts
    update_after: int | None = None
    update_every: int | None = field(default=None, metadata=_ALIAS)  # -> train_freq
    train_freq: int = 1
    gradient_steps: int = 1
    alpha: float | None = None
    init_alpha: float = 0.2
    auto_entropy_tuning: bool = True
    target_entropy: Literal["auto"] | float = "auto"  # "auto" → -action_dim
    # Floor on the auto-tuned temperature (never applied when
    # auto_entropy_tuning=False -- alpha is fixed at `alpha` then anyway).
    # `log_alpha` is otherwise an unclamped free parameter: if the policy's
    # measured entropy stays above `target_entropy` for long enough, the
    # temperature-loss gradient keeps pushing alpha down without bound --
    # and as alpha shrinks, the entropy bonus in the actor's loss weakens,
    # which can let actual policy entropy fall too, a self-reinforcing
    # spiral with nothing to arrest it. Found on a real 10M-step run
    # (JAVIS's `Javis-Payload-Rough` mjlab task): alpha held a healthy
    # ~0.2-1.9-score plateau, then collapsed from a healthy magnitude down
    # to ~3e-4 (same order of magnitude as an earlier, *unfixed* baseline's
    # 0.0002 collapse) between steps ~6M-7M, immediately followed by
    # numerical-explosion episode returns as extreme as -7.1e6 -- entropy
    # regularization was gone, and the actor found the same
    # `pitch_rate_l2`-triggered divergent-physics-state exploit a prior,
    # unfixed config's fully-collapsed alpha had already been diagnosed to
    # trigger. Permissive default (1e-8) preserves every existing run's
    # behavior byte-for-byte; set a real floor (e.g. 1e-3) to opt in.
    min_alpha: float = 1e-8
    # Prioritized Experience Replay (CPU ReplayBuffer only; not with HER or
    # the GPU buffer). Critic loss is IS-weighted, priorities = |TD error|.
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 1_000_000
    per_eps: float = 1e-6
    # Running mean/std observation normalizer (clip to +-obs_norm_clip),
    # applied to flat vector inputs only and saved in checkpoints. Off by
    # default for state tasks; the Fetch/HER configs turn it on.
    normalize_obs: bool = False
    obs_norm_clip: float = 5.0
    use_fp16: bool = False
    replay_n_step: int = 1
    replay_num_envs: int = 1
    # Encoder update frequency: encoder_optimizer steps every N critic updates.
    # 1 = every critic step (default, same as pre-v0.2 behaviour for state tasks).
    encoder_update_freq: int = 1

    # ------------------------------------------------------------------
    # Weight Normalization (FlashSAC, arXiv:2604.04539 section 4.2)
    # ------------------------------------------------------------------
    # Opt-in and off by default -- projecting weights after every optimizer
    # step changes training numerics for every SAC run that turns it on, so
    # it must be requested explicitly rather than silently applied to
    # existing configs. When True, immediately after each of the
    # actor/critic/encoder optimizers' `.step()` calls, every `nn.Linear`
    # weight row (each output unit's incoming weight vector) is projected
    # onto the unit L2-norm sphere, and every normalization layer's affine
    # vector(s) (LayerNorm/BatchNorm/GroupNorm gamma+beta, RMSNorm gamma) is
    # rescaled to L2 norm sqrt(d) where d is that vector's length. This
    # bounds uncontrolled weight growth that would otherwise inflate
    # Q-value variance and amplify bootstrapped estimation error --
    # motivated by, and most relevant when combined with, large-batch/
    # few-gradient-step SAC configurations (see `batch_size`/
    # `gradient_steps` above), where each individual update is much larger
    # and less frequent than SAC's textbook (256, ~1-per-env-step) regime.
    weight_norm_projection: bool = False

    # ------------------------------------------------------------------
    # Hindsight Experience Replay (goal-conditioned tasks only)
    # ------------------------------------------------------------------
    # Opt-in from YAML: `use_her: true` in the train block of a goal config
    # swaps the plain ReplayBuffer for HERReplayBuffer. Only meaningful with
    # `env_type: "goal"` -- the CLI errors out otherwise, since HER needs the
    # achieved/desired goals that only GoalEnvWrapper surfaces.
    use_her: bool = False
    her_ratio: float = 0.8
    her_strategy: HerStrategy = "future"
    her_max_episode_len: int = 1000
    # Filled in by the CLI from the environment, not from YAML: HER needs the
    # goal split and the env's own sparse reward function to recompute rewards
    # for relabelled goals.
    her_obs_dim: int = field(default=0, metadata={"yaml": False})
    her_goal_dim: int = field(default=0, metadata={"yaml": False})
    her_reward_fn: Callable | None = field(
        default=None, repr=False, compare=False, metadata={"yaml": False}
    )


@dataclass
class DDPGConfig(_Validated):
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    action_dim: int = 0  # required for OU noise
    learning_starts: int = 10_000
    start_steps: int | None = field(default=None, metadata=_ALIAS)  # -> learning_starts
    update_after: int | None = None
    update_every: int | None = field(default=None, metadata=_ALIAS)  # -> train_freq
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: ActionNoise = "gaussian"
    noise_sigma: float = 0.1
    # Prioritized Experience Replay (CPU ReplayBuffer only; not with HER or
    # the GPU buffer). Critic loss is IS-weighted, priorities = |TD error|.
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 1_000_000
    per_eps: float = 1e-6
    # Running mean/std observation normalizer (clip to +-obs_norm_clip),
    # applied to flat vector inputs only and saved in checkpoints. Off by
    # default for state tasks; the Fetch/HER configs turn it on.
    normalize_obs: bool = False
    obs_norm_clip: float = 5.0
    use_fp16: bool = False
    replay_n_step: int = 1
    replay_num_envs: int = 1
    encoder_update_freq: int = 1


@dataclass
class TD3Config(_Validated):
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    action_dim: int = 0
    encoder_update_freq: int = 1
    learning_starts: int = 10_000
    start_steps: int | None = field(default=None, metadata=_ALIAS)  # -> learning_starts
    update_after: int | None = None
    update_every: int | None = field(default=None, metadata=_ALIAS)  # -> train_freq
    # TD3 had no train_freq field, so `train_freq:` in a TD3 YAML was silently
    # dropped and the trainer fell back to updating every step.
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: ActionNoise = "gaussian"
    noise_sigma: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    # Prioritized Experience Replay (CPU ReplayBuffer only; not with HER or
    # the GPU buffer). Critic loss is IS-weighted, priorities = |TD error|.
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_steps: int = 1_000_000
    per_eps: float = 1e-6
    # Running mean/std observation normalizer (clip to +-obs_norm_clip),
    # applied to flat vector inputs only and saved in checkpoints. Off by
    # default for state tasks; the Fetch/HER configs turn it on.
    normalize_obs: bool = False
    obs_norm_clip: float = 5.0
    use_fp16: bool = False
    replay_n_step: int = 1
    replay_num_envs: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# Extended vision / recurrent configs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VisualPPOConfig(PPOConfig):
    encoder_lr: float = 1e-4  # ~0.3× policy lr
    # PPO's aux path only implements reconstruction ("ae"); the remaining
    # values are accepted for symmetry with VisualSACConfig.
    aux_loss_type: AuxLossType = "curl"
    aux_weight: float = 0.1
    augmentation_mode: AugmentationMode = "curl"
    latent_dim: int = 256


@dataclass
class VisualSACConfig(SACConfig):
    encoder_lr: float = 1e-4
    # Encoder update frequency for vision SAC defaults to 2 (DrQ-v2 style).
    encoder_update_freq: int = 2
    # When True, encoder receives gradients from critic loss in addition to aux
    # loss.  When False, encoder is detached from critic backward pass and
    # learns *only* through the selected aux_loss_type.
    encoder_optimize_with_critic: bool = True
    # Unsupervised / self-supervised auxiliary loss for the visual encoder.
    # "none"    – no aux loss (pure RL signal via critic when
    #             encoder_optimize_with_critic=True)
    # "ae"      – pixel reconstruction (autoencoder, MSE)
    # "vae"     – variational autoencoder (MSE recon + KL divergence)
    # "curl"    – contrastive InfoNCE with momentum encoder (CURL)
    # "byol"    – BYOL bootstrap + momentum encoder
    # "drq"     – augmented Q-consistency (DrQ-v2)
    # "spr"     – self-predictive latent forward model (SPR)
    # "barlow"  – Barlow Twins redundancy reduction
    aux_loss_type: AuxLossType = "curl"
    aux_weight: float = 0.1
    augmentation_mode: AugmentationMode = "curl"
    latent_dim: int = 256
    momentum_tau: float = 0.99  # momentum encoder EMA rate


@dataclass
class AsyncRunnerConfig(_Validated):
    """Optional async data-collection / training separation.

    Read from the YAML ``train:`` block (``use_async`` / ``use_gpu_buffer``)
    by :func:`srl.config.build_runner_config`. The former ``prefill_steps``
    and ``queue_maxsize`` fields were never read by
    :class:`~srl.runners.AsyncOffPolicyRunner` and have been removed: the
    random-action warmup is the algorithm's ``learning_starts`` and the
    runner has no collector/trainer queue.
    """

    use_async: bool = False
    use_gpu_buffer: bool = False
