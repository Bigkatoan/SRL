"""RL loss functions — PPO clip, SAC, DDPG, A2C / TD."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ppo_clip_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped surrogate PPO policy loss."""
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def ppo_value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_eps: float | None = 0.2,
) -> torch.Tensor:
    """Value function loss with optional value-clipping.

    With clipping (``old_values`` given and ``clip_eps`` not None) this is the
    PPO2 / SB3 / CleanRL formulation: the *elementwise* max of the unclipped
    and clipped squared errors, then the mean over the batch. (Taking the max
    of the two batch means instead is a different, weaker objective.)
    ``clip_eps=None`` disables value clipping entirely.
    """
    if old_values is None or clip_eps is None:
        return F.mse_loss(values, returns)
    v_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
    loss_unclipped = (values - returns) ** 2
    loss_clipped = (v_clipped - returns) ** 2
    return torch.max(loss_unclipped, loss_clipped).mean()


def entropy_loss(entropy: torch.Tensor) -> torch.Tensor:
    return -entropy.mean()


def a2c_policy_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    return -(log_probs * advantages.detach()).mean()


def a2c_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(values, returns)


def sac_policy_loss(
    log_probs: torch.Tensor,
    q_values: torch.Tensor,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """SAC actor loss: min_π [α log π - Q(s,a)]."""
    return (alpha * log_probs - q_values).mean()


def sac_temperature_loss(
    log_probs: torch.Tensor,
    log_alpha: torch.Tensor,
    target_entropy: float,
) -> torch.Tensor:
    """Dual objective for automatic entropy tuning."""
    return -(log_alpha * (log_probs + target_entropy).detach()).mean()


def sac_q_loss(
    q1: torch.Tensor,
    q2: torch.Tensor,
    target_q: torch.Tensor,
) -> torch.Tensor:
    """Twin Q-network regression loss."""
    return F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)


def ddpg_q_loss(
    q: torch.Tensor,
    target_q: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(q, target_q)


def ddpg_policy_loss(q: torch.Tensor) -> torch.Tensor:
    """DDPG actor loss: max_a Q(s,a)."""
    return -q.mean()


def td_error(
    q: torch.Tensor,
    reward: torch.Tensor,
    next_q: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    target = (reward + gamma * next_q * (1.0 - done)).detach()
    return F.mse_loss(q, target)
