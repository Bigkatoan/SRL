"""Algorithm and model hyperparameter dataclasses."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────────
# Algorithm configs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    lr: float = 3e-4
    n_steps: int = 2048  # steps per env per rollout
    num_envs: int = 1
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None  # None = same as clip_range
    entropy_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    use_fp16: bool = False
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
    lr_schedule: str = "fixed"  # "fixed" | "adaptive"
    desired_kl: float = 0.01
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    kl_lr_factor: float = 1.5
    # Entropy coefficient annealing (off by default -- `entropy_coef` stays
    # fixed for the whole run unless `entropy_coef_anneal_steps` is set).
    # Unlike SAC's auto-tuned `alpha` (which targets a specific entropy
    # level and can push either direction), PPO's entropy bonus here is a
    # constant, one-directional pull toward MORE entropy every single
    # minibatch, with nothing that naturally pulls it back down once
    # "enough" exploration has been reached. On a real 40M-step PPO run
    # against JAVIS's mjlab balance task (with lr_schedule="adaptive"
    # already enabled -- see above), `ppo/entropy` climbed continuously
    # for the entire run (-2.76 at step ~100k to +1.07 by step ~27.6M,
    # still rising, log_std correctly clamped to [log_std_min,
    # log_std_max] but nowhere near that ceiling yet) while eval score
    # declined from a step-11M peak -- i.e. the OPPOSITE of the earlier
    # entropy-COLLAPSE failure mode this file's `lr_schedule="adaptive"`
    # comment above describes, but plausibly the same class of problem
    # (nothing bounding a monotonic drift over a long enough run): a
    # policy getting steadily noisier hurts closed-loop control on a
    # physical balance task regardless of which direction the drift goes.
    # When `entropy_coef_anneal_steps` is set, `entropy_coef` is linearly
    # annealed from its configured value down to `entropy_coef_final`
    # (default 0.0) over that many GRADIENT steps (`self._global_step`
    # below -- one increment per minibatch, i.e. the same unit
    # `lr_schedule="adaptive"`'s own internal bookkeeping already uses,
    # NOT env steps -- `n_epochs * total_env_steps / batch_size` if you
    # need to convert from an env-step budget), then held at
    # `entropy_coef_final` for the rest of the run: real exploration
    # pressure early, no constant countervailing force once the policy
    # gradient should be sharpening the policy back down late. See
    # `PPO.__init__`'s composer setup for where this is wired through
    # `LossComposer`'s existing (already-implemented, previously unused
    # for this term) `schedule="linear_decay"` support.
    entropy_coef_final: float = 0.0
    entropy_coef_anneal_steps: int | None = None


@dataclass
class A2CConfig:
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
class A3CConfig:
    lr: float = 1e-4
    n_steps: int = 20
    gamma: float = 0.99
    gae_lambda: float = 1.0
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 40.0
    n_workers: int = 4
    batch_size: int = 20


@dataclass
class SACConfig:
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
    start_steps: int | None = None
    update_after: int | None = None
    update_every: int | None = None
    train_freq: int = 1
    gradient_steps: int = 1
    alpha: float | None = None
    init_alpha: float = 0.2
    auto_entropy_tuning: bool = True
    target_entropy: str | float = "auto"  # "auto" → -action_dim
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
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
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
    her_strategy: str = "future"  # "future" | "final" | "episode" | "random"
    her_max_episode_len: int = 1000
    # Filled in by the CLI from the environment, not from YAML: HER needs the
    # goal split and the env's own sparse reward function to recompute rewards
    # for relabelled goals.
    her_obs_dim: int = 0
    her_goal_dim: int = 0
    her_reward_fn: Callable | None = field(default=None, repr=False, compare=False)


@dataclass
class DDPGConfig:
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    action_dim: int = 0  # required for OU noise
    learning_starts: int = 10_000
    start_steps: int | None = None
    update_after: int | None = None
    update_every: int | None = None
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: str = "gaussian"  # "gaussian" | "ou"
    noise_sigma: float = 0.1
    use_per: bool = False
    use_fp16: bool = False
    replay_n_step: int = 1
    replay_num_envs: int = 1
    encoder_update_freq: int = 1


@dataclass
class TD3Config:
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    action_dim: int = 0
    encoder_update_freq: int = 1
    learning_starts: int = 10_000
    start_steps: int | None = None
    update_after: int | None = None
    update_every: int | None = None
    gradient_steps: int = 1
    action_noise: str = "gaussian"
    noise_sigma: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    use_fp16: bool = False
    replay_n_step: int = 1
    replay_num_envs: int = 1


# ──────────────────────────────────────────────────────────────────────────────
# Extended vision / recurrent configs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VisualPPOConfig(PPOConfig):
    encoder_lr: float = 1e-4  # ~0.3× policy lr
    aux_loss_type: str = "curl"  # "curl" | "ae" | "none"
    aux_weight: float = 0.1
    augmentation_mode: str = "curl"  # "drq" | "curl" | "aggressive"
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
    aux_loss_type: str = "curl"
    aux_weight: float = 0.1
    augmentation_mode: str = "curl"  # "drq" | "curl" | "aggressive"
    latent_dim: int = 256
    momentum_tau: float = 0.99  # momentum encoder EMA rate


@dataclass
class AsyncRunnerConfig:
    """Optional async data-collection / training separation."""

    use_async: bool = False
    use_gpu_buffer: bool = False
    # Number of transitions the collector pre-fills before starting updates.
    prefill_steps: int = 0
    # Internal queue depth between collector and trainer (async mode only).
    queue_maxsize: int = 2


@dataclass
class RecurrentPPOConfig(PPOConfig):
    lstm_hidden: int = 256
    burn_in_steps: int = 32
