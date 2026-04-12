"""TD3 (Twin Delayed DDPG) — off-policy, deterministic continuous control.

Changes in v0.2.0
-----------------
* Three separate optimizers: critic_optimizer, actor_optimizer, encoder_optimizer.
  Encoder is updated only through critic backward; actor backward never touches
  encoder weights, eliminating the effective 2× encoder LR bug.
* encoder_update_freq (TD3Config): encoder_optimizer steps every N critic updates,
  independently of policy_delay (actor update schedule).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.core.base_agent import BaseAgent
from srl.core.config import TD3Config
from srl.core.replay_buffer import ReplayBuffer
from srl.losses.rl_losses import ddpg_policy_loss, sac_q_loss


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

        actor_encoder_params = _encoder_params_for_head(self.model, "actor")
        critic_encoder_params = _encoder_params_for_head(self.model, "critic")

        # ------------------------------------------------------------------
        # Three-optimizer design (v0.2.0)
        # ------------------------------------------------------------------
        self._encoder_param_list: list[nn.Parameter] = _unique_encoder_params(self.model)
        _encoder_lr = getattr(self.cfg, "encoder_lr", self.cfg.lr_critic)
        self.encoder_optimizer: torch.optim.Optimizer | None = (
            torch.optim.Adam(self._encoder_param_list, lr=_encoder_lr)
            if self._encoder_param_list
            else None
        )
        self.actor_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters()),
            lr=self.cfg.lr_actor,
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.critic.parameters()),
            lr=self.cfg.lr_critic,
        )
        self._encoder_update_counter: int = 0

        self.buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            num_envs=self.cfg.replay_num_envs,
            n_step=self.cfg.replay_n_step,
            gamma=self.cfg.gamma,
            use_fp16=self.cfg.use_fp16,
        )
        self._global_step = 0
        self._update_count = 0

    def predict(self, obs, hidden=None, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden)
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
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        dones = batch.dones.to(self.device)

        with torch.no_grad():
            target_actor_out = self.target_model(next_obs)["actor_out"]
            next_action = target_actor_out.get("action") if isinstance(target_actor_out, dict) else target_actor_out
            target_noise = torch.randn_like(next_action) * self.cfg.policy_noise
            target_noise = target_noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (next_action + target_noise).clamp(-1.0, 1.0)

            target_q = self.target_model(next_obs, action=next_action)["value"]
            if isinstance(target_q, tuple):
                target_q = torch.min(*target_q)
            backup = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        q_out = self.model(obs, action=actions)["value"]
        if isinstance(q_out, tuple):
            q1, q2 = q_out
            critic_loss = sac_q_loss(q1, q2, backup.detach())
            q1_mean = q1.mean()
            q2_mean = q2.mean()
        else:
            critic_loss = torch.nn.functional.mse_loss(q_out, backup.detach())
            q1_mean = q_out.mean()
            q2_mean = q_out.mean()

        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Encoder step — every encoder_update_freq critic updates
        self._encoder_update_counter += 1
        freq = self.cfg.encoder_update_freq
        if self.encoder_optimizer is not None and (self._encoder_update_counter % freq == 0):
            self.encoder_optimizer.step()

        # Zero encoder grads before actor backward
        _zero_param_grads(self._encoder_param_list)

        actor_loss_value = None
        if self._update_count % self.cfg.policy_delay == 0:
            actor_out = self.model(obs)["actor_out"]
            policy_action = actor_out.get("action") if isinstance(actor_out, dict) else actor_out
            q_actor = self.model(obs, action=policy_action)["value"]
            if isinstance(q_actor, tuple):
                q_actor = q_actor[0]
            actor_loss = ddpg_policy_loss(q_actor)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
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


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data * tau)


def _unique_encoder_params(model: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    params: list[nn.Parameter] = []
    encoders = getattr(model, "encoders", {})
    for enc in encoders.values():
        for p in enc.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                params.append(p)
    return params


def _zero_param_grads(params: list[nn.Parameter]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _encoder_params_for_head(model: nn.Module, head_name: str) -> list[nn.Parameter]:
    encoder_names = getattr(model, "encoder_names_for_head")(head_name)
    params: list[nn.Parameter] = []
    for encoder_name in encoder_names:
        params.extend(list(model.encoders[encoder_name].parameters()))
    return params