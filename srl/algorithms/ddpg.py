"""DDPG (Deep Deterministic Policy Gradient) — off-policy, deterministic.

Changes in v0.2.0
-----------------
* Three separate optimizers: critic_optimizer, actor_optimizer, encoder_optimizer.
  Encoder is updated only through critic backward — actor backward never touches
  encoder weights, eliminating the effective 2× encoder LR bug.
* encoder_update_freq (DDPGConfig): encoder_optimizer steps every N critic updates.

Fixes
-----
* Actor-only encoders now train (own optimizer, stepped off the actor loss).
* train/eval mode is set atomically per forward under a lock (``predict()`` on
  the async collector thread used to flip the shared model's mode mid-update).
* Exploration noise is per env (one independent OU/Gaussian process per row
  of the action batch) and :meth:`DDPG.reset_noise` resets it at episode ends.
* n-step bootstrap uses the per-transition ``gamma**k`` discount.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from srl.algorithms._common import (
    _unique_encoder_params,  # noqa: F401 -- re-exported for backwards compat
    _zero_param_grads,
    is_weighted_mse,
    maybe_update_priorities,
)
from srl.core.base_agent import OffPolicyAgent
from srl.core.config import DDPGConfig
from srl.losses.rl_losses import ddpg_policy_loss


def _reset_mask(n: int, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.ones(n, dtype=torch.bool)
    m = torch.as_tensor(np.asarray(env_ids))
    if m.dtype == torch.bool:
        return m.reshape(-1)[:n] if m.numel() >= n else torch.ones(n, dtype=torch.bool)
    out = torch.zeros(n, dtype=torch.bool)
    out[m.long().reshape(-1)] = True
    return out


class OrnsteinUhlenbeckNoise:
    """OU process for temporally correlated exploration noise.

    Keeps one independent process per env: ``sample(n)`` returns an
    ``(n, action_dim)`` draw (``sample()`` -> ``(action_dim,)`` for the
    single-env case). ``reset(env_ids)`` restarts the processes of the given
    envs (all when ``None``) at ``mu`` -- call it at each episode end, or the
    correlated noise carries over from the previous episode.
    """

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu * torch.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state: torch.Tensor | None = None  # (n_envs, action_dim)

    def reset(self, env_ids=None) -> None:
        if self.state is None:
            return
        mask = _reset_mask(self.state.shape[0], env_ids)
        self.state[mask] = self.mu

    def sample(self, n: int | None = None) -> torch.Tensor:
        rows = 1 if n is None else int(n)
        if self.state is None or self.state.shape[0] != rows:
            self.state = self.mu.expand(rows, -1).clone()
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn_like(x)
        self.state = x + dx
        return self.state[0].clone() if n is None else self.state.clone()


class GaussianActionNoise:
    """Independent Gaussian exploration noise (stateless, so reset is a no-op)."""

    def __init__(self, action_dim: int, sigma: float = 0.1):
        self.action_dim = action_dim
        self.sigma = sigma

    def reset(self, env_ids=None) -> None:
        return None

    def sample(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return torch.randn(self.action_dim) * self.sigma
        return torch.randn(int(n), self.action_dim) * self.sigma


class DDPG(OffPolicyAgent):
    """Deep Deterministic Policy Gradient.

    Parameters
    ----------
    model:
        AgentModel with a DeterministicActorHead and QFunctionHead.
    target_model:
        Optional target network (same architecture); deep-copied from
        *model* when omitted. ``DDPG(model, config)`` also works.
    config:
        DDPGConfig.
    device:
        PyTorch device.
    env:
        Optional training env (raw Gymnasium envs are wrapped).
    """

    algo_name = "ddpg"
    config_cls = DDPGConfig

    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module | None = None,
        config: DDPGConfig | None = None,
        device: str | torch.device = "cpu",
        *,
        env=None,
    ) -> None:
        super().__init__(model, target_model, config, device, env=env)

    def _build_exploration(self) -> None:
        action_dim = self.cfg.action_dim or 1
        if self.cfg.action_noise == "ou":
            self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=self.cfg.noise_sigma)
        else:
            self.noise = GaussianActionNoise(action_dim, sigma=self.cfg.noise_sigma)

    def reset_noise(self, env_ids=None) -> None:
        """Reset exploration noise for the envs whose episode just ended.

        *env_ids*: ``None`` (all), a boolean mask over envs, or indices.
        """
        self.noise.reset(env_ids)

    def _select_action(self, actor_out, deterministic: bool):
        action = actor_out.get("action") if isinstance(actor_out, dict) else actor_out
        if not deterministic and action is not None:
            if action.dim() > 1:
                noise = self.noise.sample(action.shape[0])
            else:
                noise = self.noise.sample()
            action = (action + noise.to(device=action.device, dtype=action.dtype)).clamp(-1.0, 1.0)
        return action, None

    def update(self) -> dict[str, float]:
        if not self._can_update():
            return {}

        batch, obs, next_obs, actions, rewards, dones, discount, is_weights = self._sample_batch()

        with torch.no_grad():
            next_actor_out = self.target_model(next_obs, compute_critic=False)["actor_out"]
            if isinstance(next_actor_out, dict):
                next_action = next_actor_out.get("action")
            else:
                next_action = next_actor_out
            next_q_raw = self.target_model(next_obs, action=next_action, compute_actor=False)[
                "value"
            ]
            # TwinQHead returns (q1, q2); take min to prevent overestimation
            if isinstance(next_q_raw, tuple):
                next_q = torch.min(next_q_raw[0], next_q_raw[1])
            else:
                next_q = next_q_raw
            target_q = rewards.float() + discount * (1.0 - dones.float()) * next_q

        q_raw = self._forward_model("train", obs, action=actions, compute_actor=False)["value"]
        # TwinQHead: compute loss for both Q networks
        if isinstance(q_raw, tuple):
            critic_loss = is_weighted_mse(q_raw[0], target_q, is_weights) + is_weighted_mse(
                q_raw[1], target_q, is_weights
            )
            td_error = 0.5 * ((q_raw[0] - target_q).abs() + (q_raw[1] - target_q).abs())
        else:
            critic_loss = is_weighted_mse(q_raw, target_q, is_weights)
            td_error = (q_raw - target_q).abs()
        maybe_update_priorities(self.buffer, batch, td_error)

        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        if self.actor_only_encoder_optimizer is not None:
            self.actor_only_encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Encoder step — every encoder_update_freq critic updates
        self._encoder_update_counter += 1
        freq = self.cfg.encoder_update_freq
        if self.encoder_optimizer is not None and (self._encoder_update_counter % freq == 0):
            self.encoder_optimizer.step()

        # Zero critic-reachable encoder grads before actor backward
        _zero_param_grads(self._encoder_param_list)

        # Actor
        actor_out = self._forward_model("train", obs, compute_critic=False)["actor_out"]
        if isinstance(actor_out, dict):
            new_action = actor_out.get("action")
        else:
            new_action = actor_out
        q_actor_raw = self._forward_model("train", obs, action=new_action, compute_actor=False)[
            "value"
        ]
        # For actor loss use Q1 (or mean of twin) — maximise Q
        q_actor = q_actor_raw[0] if isinstance(q_actor_raw, tuple) else q_actor_raw
        actor_loss = ddpg_policy_loss(q_actor)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.actor_only_encoder_optimizer is not None:
            self.actor_only_encoder_optimizer.step()

        # Zero encoder grads again — clean for next update() call
        _zero_param_grads(self._encoder_param_list)

        self._soft_update_target()
        self._global_step += 1

        return {
            "ddpg/critic_loss": critic_loss.item(),
            "ddpg/actor_loss": actor_loss.item(),
            "ddpg/target_q_mean": target_q.mean().item(),
        }
