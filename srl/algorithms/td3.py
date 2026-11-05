"""TD3 (Twin Delayed DDPG) — off-policy, deterministic continuous control.

Changes in v0.2.0
-----------------
* Three separate optimizers: critic_optimizer, actor_optimizer, encoder_optimizer.
  Encoder is updated only through critic backward; actor backward never touches
  encoder weights, eliminating the effective 2× encoder LR bug.
* encoder_update_freq (TD3Config): encoder_optimizer steps every N critic updates,
  independently of policy_delay (actor update schedule).

Fixes
-----
* Actor-only encoders (separate ``actor_state_enc`` flows) now train, through
  their own optimizer stepped off the actor loss (same split as SAC).
* ``predict()`` no longer leaves the model in eval mode for every later
  ``update()``: every forward sets its mode atomically under a lock.
* n-step bootstrap uses the per-transition ``gamma**k`` discount.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.algorithms._common import (
    _unique_encoder_params,  # noqa: F401 -- re-exported for backwards compat
    _zero_param_grads,
    is_weighted_mse,
    maybe_update_priorities,
)
from srl.core.base_agent import OffPolicyAgent
from srl.core.config import TD3Config
from srl.losses.rl_losses import ddpg_policy_loss


class TD3(OffPolicyAgent):
    """Twin Delayed DDPG.

    Parameters
    ----------
    model:
        AgentModel with a DeterministicActorHead and a TwinQ critic.
    target_model:
        Optional target network (same architecture); deep-copied from
        *model* when omitted. ``TD3(model, config)`` also works.
    config:
        :class:`~srl.core.config.TD3Config`.
    device:
        PyTorch device.
    env:
        Optional training env (raw Gymnasium envs are wrapped).
    """

    algo_name = "td3"
    config_cls = TD3Config

    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module | None = None,
        config: TD3Config | None = None,
        device: str | torch.device = "cpu",
        *,
        env=None,
    ) -> None:
        super().__init__(model, target_model, config, device, env=env)
        self._update_count = 0

    def _select_action(self, actor_out, deterministic: bool):
        action = actor_out.get("action") if isinstance(actor_out, dict) else actor_out
        if not deterministic and action is not None:
            noise = torch.randn_like(action) * self.cfg.noise_sigma
            action = (action + noise).clamp(-1.0, 1.0)
        return action, None

    def update(self) -> dict[str, float]:
        if not self._can_update():
            return {}

        batch, obs, next_obs, actions, rewards, dones, discount, is_weights = self._sample_batch()

        with torch.no_grad():
            target_actor_out = self.target_model(next_obs, compute_critic=False)["actor_out"]
            next_action = (
                target_actor_out.get("action")
                if isinstance(target_actor_out, dict)
                else target_actor_out
            )
            target_noise = torch.randn_like(next_action) * self.cfg.policy_noise
            target_noise = target_noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (next_action + target_noise).clamp(-1.0, 1.0)

            target_q = self.target_model(next_obs, action=next_action, compute_actor=False)["value"]
            if isinstance(target_q, tuple):
                target_q = torch.min(*target_q)
            backup = rewards + discount * (1.0 - dones) * target_q

        q_out = self._forward_model("train", obs, action=actions, compute_actor=False)["value"]
        if isinstance(q_out, tuple):
            q1, q2 = q_out
            critic_loss = is_weighted_mse(q1, backup, is_weights) + is_weighted_mse(
                q2, backup, is_weights
            )
            td_error = 0.5 * ((q1 - backup).abs() + (q2 - backup).abs())
            q1_mean = q1.mean()
            q2_mean = q2.mean()
        else:
            critic_loss = is_weighted_mse(q_out, backup, is_weights)
            td_error = (q_out - backup).abs()
            q1_mean = q_out.mean()
            q2_mean = q_out.mean()
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
        encoder_step_due = self._encoder_update_counter % freq == 0
        if self.encoder_optimizer is not None and encoder_step_due:
            self.encoder_optimizer.step()

        # Zero critic-reachable encoder grads before actor backward
        _zero_param_grads(self._encoder_param_list)

        actor_loss_value = None
        if self._update_count % self.cfg.policy_delay == 0:
            actor_out = self._forward_model("train", obs, compute_critic=False)["actor_out"]
            policy_action = actor_out.get("action") if isinstance(actor_out, dict) else actor_out
            q_actor = self._forward_model("train", obs, action=policy_action, compute_actor=False)[
                "value"
            ]
            if isinstance(q_actor, tuple):
                q_actor = q_actor[0]
            actor_loss = ddpg_policy_loss(q_actor)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            # Actor-only encoders: actor_loss is their only training signal.
            if self.actor_only_encoder_optimizer is not None:
                self.actor_only_encoder_optimizer.step()
                self.actor_only_encoder_optimizer.zero_grad()
            self._soft_update_target()
            actor_loss_value = actor_loss.item()

        # Zero encoder grads again — clean for next update() call
        _zero_param_grads(self._encoder_param_list)

        self._global_step += 1
        self._update_count += 1

        metrics = {
            "td3/critic_loss": critic_loss.item(),
            "td3/q1_mean": q1_mean.item(),
            "td3/q2_mean": q2_mean.item(),
            "td3/target_q_mean": backup.mean().item(),
        }
        if actor_loss_value is not None:
            metrics["td3/actor_loss"] = actor_loss_value
        return metrics

    def _algo_payload(self) -> dict[str, object]:
        return {**self._base_payload(), "update_count": self._update_count}

    def _load_algo_payload(self, payload: dict[str, object]) -> None:
        self._load_base_payload(payload)
        self._update_count = int(payload.get("update_count", 0))
