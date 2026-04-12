"""DDPG (Deep Deterministic Policy Gradient) — off-policy, deterministic.

Changes in v0.2.0
-----------------
* Three separate optimizers: critic_optimizer, actor_optimizer, encoder_optimizer.
  Encoder is updated only through critic backward — actor backward never touches
  encoder weights, eliminating the effective 2× encoder LR bug.
* encoder_update_freq (DDPGConfig): encoder_optimizer steps every N critic updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.core.base_agent import BaseAgent
from srl.core.config import DDPGConfig
from srl.core.replay_buffer import ReplayBuffer
from srl.losses.rl_losses import ddpg_policy_loss, ddpg_q_loss


class OrnsteinUhlenbeckNoise:
    """OU process for temporally correlated exploration noise."""

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * torch.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = self.mu.clone()

    def sample(self) -> torch.Tensor:
        if self.state is None:
            self.state = self.mu.clone()
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn_like(x)
        self.state = x + dx
        return self.state


class GaussianActionNoise:
    """Independent Gaussian exploration noise."""

    def __init__(self, action_dim: int, sigma: float = 0.1):
        self.action_dim = action_dim
        self.sigma = sigma

    def reset(self) -> None:
        return None

    def sample(self) -> torch.Tensor:
        return torch.randn(self.action_dim) * self.sigma


class DDPG(BaseAgent):
    """Deep Deterministic Policy Gradient.

    Parameters
    ----------
    model:
        AgentModel with a DeterministicActorHead and QFunctionHead.
    target_model:
        Target network (same architecture).
    config:
        DDPGConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module,
        config: DDPGConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.target_model = target_model
        self.cfg = config or DDPGConfig()
        self._device = torch.device(device)

        self.model.to(self.device)
        self.target_model.to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        for p in self.target_model.parameters():
            p.requires_grad = False

        actor_encoder_params = _encoder_params_for_head(self.model, "actor")
        critic_encoder_params = _encoder_params_for_head(self.model, "critic")

        # ------------------------------------------------------------------
        # Three-optimizer design (v0.2.0): encoder separated from heads
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

        action_dim = self.cfg.action_dim or 1
        if self.cfg.action_noise == "ou":
            self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=self.cfg.noise_sigma)
        else:
            self.noise = GaussianActionNoise(action_dim, sigma=self.cfg.noise_sigma)

        self.buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            num_envs=self.cfg.replay_num_envs,
            n_step=self.cfg.replay_n_step,
            gamma=self.cfg.gamma,
            use_fp16=self.cfg.use_fp16,
        )
        self._global_step = 0

    def predict(self, obs, hidden=None, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden)
        actor_out = result["actor_out"]
        if isinstance(actor_out, dict):
            action = actor_out.get("action")
        else:
            action = actor_out
        if not deterministic and action is not None:
            action = action + self.noise.sample().to(self.device)
            action = action.clamp(-1.0, 1.0)
        return action, None, None, result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError

    def update(self) -> dict[str, float]:
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        self.model.train()
        batch = self.buffer.sample(self.cfg.batch_size)

        obs = {k: v.to(self.device) for k, v in batch.obs.items()}
        next_obs = {k: v.to(self.device) for k, v in batch.next_obs.items()}
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        dones = batch.dones.to(self.device)

        with torch.no_grad():
            next_actor_out = self.target_model(next_obs)["actor_out"]
            if isinstance(next_actor_out, dict):
                next_action = next_actor_out.get("action")
            else:
                next_action = next_actor_out
            next_q_raw = self.target_model(next_obs, action=next_action)["value"]
            # TwinQHead returns (q1, q2); take min to prevent overestimation
            if isinstance(next_q_raw, tuple):
                next_q = torch.min(next_q_raw[0], next_q_raw[1])
            else:
                next_q = next_q_raw
            target_q = rewards.float() + self.cfg.gamma * (1.0 - dones.float()) * next_q

        q_raw = self.model(obs, action=actions)["value"]
        # TwinQHead: compute loss for both Q networks
        if isinstance(q_raw, tuple):
            critic_loss = ddpg_q_loss(q_raw[0], target_q) + ddpg_q_loss(q_raw[1], target_q)
            q_for_log = q_raw[0]
        else:
            critic_loss = ddpg_q_loss(q_raw, target_q)
            q_for_log = q_raw

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

        # Actor
        actor_out = self.model(obs)["actor_out"]
        if isinstance(actor_out, dict):
            new_action = actor_out.get("action")
        else:
            new_action = actor_out
        q_actor_raw = self.model(obs, action=new_action)["value"]
        # For actor loss use Q1 (or mean of twin) — maximise Q
        q_actor = q_actor_raw[0] if isinstance(q_actor_raw, tuple) else q_actor_raw
        actor_loss = ddpg_policy_loss(q_actor)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Zero encoder grads again — clean for next update() call
        _zero_param_grads(self._encoder_param_list)

        _soft_update(self.model, self.target_model, self.cfg.tau)
        self._global_step += 1

        return {
            "ddpg/critic_loss": critic_loss.item(),
            "ddpg/actor_loss": actor_loss.item(),
        }

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
        self._encoder_update_counter = int(payload.get("encoder_update_counter", 0))
        enc_opt = payload.get("encoder_optimizer_state")
        if self.encoder_optimizer is not None and enc_opt is not None:
            self.encoder_optimizer.load_state_dict(enc_opt)


def _soft_update(src, tgt, tau):
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
