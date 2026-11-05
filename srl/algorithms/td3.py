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

import threading

import torch
import torch.nn as nn

from srl.algorithms._common import (
    _soft_update,
    _unique_encoder_params,  # noqa: F401 -- re-exported for backwards compat
    _zero_param_grads,
    batch_discounts,
    build_encoder_optimizers,
    forward_with_mode,
    is_weighted_mse,
    maybe_build_obs_normalizer,
    maybe_update_priorities,
    prepare_obs,
)
from srl.core.base_agent import BaseAgent
from srl.core.config import TD3Config
from srl.core.prioritized_replay_buffer import build_replay_buffer
from srl.losses.rl_losses import ddpg_policy_loss


class TD3(BaseAgent):
    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module,
        config: TD3Config | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.target_model = target_model
        self.cfg = config or TD3Config()
        self._device = torch.device(device)

        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        for parameter in self.target_model.parameters():
            parameter.requires_grad = False
        # Target net is only ever used for inference.
        self.target_model.eval()
        self._model_lock = threading.Lock()

        # ------------------------------------------------------------------
        # Optimizers: critic head, actor head, critic-reachable encoders
        # (trained via critic loss only), actor-only encoders (trained via
        # actor loss -- their only gradient source).
        # ------------------------------------------------------------------
        _encoder_lr = getattr(self.cfg, "encoder_lr", self.cfg.lr_critic)
        (
            self._encoder_param_list,
            self._actor_only_encoder_param_list,
            _,
            self.encoder_optimizer,
            self.actor_only_encoder_optimizer,
        ) = build_encoder_optimizers(self.model, _encoder_lr)
        self.actor_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters()),
            lr=self.cfg.lr_actor,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.critic.parameters()),
            lr=self.cfg.lr_critic,
        )
        self._encoder_update_counter: int = 0

        self.buffer = build_replay_buffer(self.cfg)
        self.obs_normalizer = maybe_build_obs_normalizer(self.cfg)
        self._global_step = 0
        self._update_count = 0

    def _forward_model(self, mode: str, *args, **kwargs):
        return forward_with_mode(self.model, self._model_lock, mode, *args, **kwargs)

    def predict(self, obs, hidden=None, deterministic=False):
        with torch.no_grad():
            obs = prepare_obs(self.model, self.obs_normalizer, obs)
            result = self._forward_model("eval", obs, hidden_states=hidden, compute_critic=False)
        actor_out = result["actor_out"]
        action = actor_out.get("action") if isinstance(actor_out, dict) else actor_out
        if not deterministic and action is not None:
            noise = torch.randn_like(action) * self.cfg.noise_sigma
            action = (action + noise).clamp(-1.0, 1.0)
        return action, None, None, result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        batch = self.buffer.sample(self.cfg.batch_size)
        obs = {k: v.to(self.device) for k, v in batch.obs.items()}
        next_obs = {k: v.to(self.device) for k, v in batch.next_obs.items()}
        obs = prepare_obs(self.model, self.obs_normalizer, obs, update_stats=True)
        next_obs = prepare_obs(self.model, self.obs_normalizer, next_obs)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        dones = batch.dones.to(self.device)
        discount = batch_discounts(batch, self.cfg.gamma, self.device)
        is_weights = batch.weights.to(self.device) if batch.weights is not None else None

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
            _soft_update(self.model, self.target_model, self.cfg.tau)
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

    def save(self, path: str) -> None:
        torch.save(self.checkpoint_payload(), path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_checkpoint_payload(ckpt)

    def checkpoint_payload(self) -> dict[str, object]:
        payload = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "replay_buffer_state": self.buffer.state_dict(),
            "algo_step": self._global_step,
            "update_count": self._update_count,
            "encoder_update_counter": self._encoder_update_counter,
        }
        if self.encoder_optimizer is not None:
            payload["encoder_optimizer_state"] = self.encoder_optimizer.state_dict()
        if self.actor_only_encoder_optimizer is not None:
            payload["actor_only_encoder_optimizer_state"] = (
                self.actor_only_encoder_optimizer.state_dict()
            )
        if self.obs_normalizer is not None:
            payload["obs_normalizer_state"] = self.obs_normalizer.state_dict()
        return payload

    def load_checkpoint_payload(self, payload: dict[str, object]) -> None:
        model_state = payload.get("model_state", payload.get("model"))
        if model_state is not None:
            self.model.load_state_dict(model_state)
        target_state = payload.get("target_model_state", payload.get("target"))
        if target_state is not None:
            self.target_model.load_state_dict(target_state)
        actor_optimizer_state = payload.get("actor_optimizer_state")
        if actor_optimizer_state is not None:
            self.actor_optimizer.load_state_dict(actor_optimizer_state)
        critic_optimizer_state = payload.get("critic_optimizer_state")
        if critic_optimizer_state is not None:
            self.critic_optimizer.load_state_dict(critic_optimizer_state)
        replay_buffer_state = payload.get("replay_buffer_state")
        if replay_buffer_state is not None:
            self.buffer.load_state_dict(replay_buffer_state)
        self._global_step = int(payload.get("algo_step", payload.get("step", 0)))
        self._update_count = int(payload.get("update_count", 0))
        self._encoder_update_counter = int(payload.get("encoder_update_counter", 0))
        enc_opt = payload.get("encoder_optimizer_state")
        if self.encoder_optimizer is not None and enc_opt is not None:
            self.encoder_optimizer.load_state_dict(enc_opt)
        actor_enc_opt = payload.get("actor_only_encoder_optimizer_state")
        if self.actor_only_encoder_optimizer is not None and actor_enc_opt is not None:
            self.actor_only_encoder_optimizer.load_state_dict(actor_enc_opt)
        norm_state = payload.get("obs_normalizer_state")
        if self.obs_normalizer is not None and norm_state is not None:
            self.obs_normalizer.load_state_dict(norm_state)


def _encoder_params_for_head(model: nn.Module, head_name: str) -> list[nn.Parameter]:
    encoder_names = model.encoder_names_for_head(head_name)
    params: list[nn.Parameter] = []
    for encoder_name in encoder_names:
        params.extend(list(model.encoders[encoder_name].parameters()))
    return params
