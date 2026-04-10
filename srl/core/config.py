"""Algorithm and model hyperparameter dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    lr: float = 3e-4
    n_steps: int = 2048          # steps per env per rollout
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


@dataclass
class A2CConfig:
    lr: float = 7e-4
    n_steps: int = 5
    num_envs: int = 1
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
    tau: float = 0.005            # soft target update coefficient
    action_dim: int = 0           # required for automatic target entropy
    target_update_interval: int = 1
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    init_alpha: float = 0.2
    auto_entropy_tuning: bool = True
    target_entropy: str | float = "auto"  # "auto" → -action_dim
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    use_fp16: bool = False


@dataclass
class DDPGConfig:
    lr_actor: float = 1e-4
    lr_critic: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    action_dim: int = 0           # required for OU noise
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1
    action_noise: str = "gaussian"  # "gaussian" | "ou"
    noise_sigma: float = 0.1
    use_per: bool = False
    use_fp16: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Extended vision / recurrent configs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VisualPPOConfig(PPOConfig):
    encoder_lr: float = 1e-4      # ~0.3× policy lr
    aux_loss_type: str = "curl"   # "curl" | "ae" | "none"
    aux_weight: float = 0.1
    augmentation_mode: str = "curl"  # "drq" | "curl" | "aggressive"
    latent_dim: int = 256


@dataclass
class VisualSACConfig(SACConfig):
    encoder_lr: float = 1e-4
    aux_loss_type: str = "curl"
    aux_weight: float = 0.1
    augmentation_mode: str = "curl"
    latent_dim: int = 256
    momentum_tau: float = 0.99   # momentum encoder update rate


@dataclass
class RecurrentPPOConfig(PPOConfig):
    lstm_hidden: int = 256
    burn_in_steps: int = 32
