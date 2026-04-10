"""SAC (Soft Actor-Critic) — off-policy, continuous actions."""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.core.base_agent import BaseAgent
from srl.core.config import SACConfig
from srl.core.replay_buffer import ReplayBuffer
from srl.losses.rl_losses import (
    sac_policy_loss,
    sac_q_loss,
    sac_temperature_loss,
)
from srl.utils.checkpoint import CheckpointManager


class SAC(BaseAgent):
    """Soft Actor-Critic with automatic entropy tuning.

    Parameters
    ----------
    model:
        :class:`~srl.networks.agent_model.AgentModel` with a SquashedGaussian
        actor and TwinQ critic.
    target_model:
        Same architecture as *model*, used as the target network. Will be
        soft-updated with ``tau``.
    config:
        :class:`~srl.core.config.SACConfig`.
    device:
        PyTorch device.
    """

    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module,
        config: SACConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.target_model = target_model
        self.cfg = config or SACConfig()
        self._device = torch.device(device)

        self.model.to(self._device)
        self.target_model.to(self._device)

        # Copy weights to target, freeze
        self.target_model.load_state_dict(self.model.state_dict())
        for p in self.target_model.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.cfg.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=self.cfg.lr_critic
        )

        # Automatic entropy tuning
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        action_dim = self.cfg.action_dim
        self.target_entropy: float = -float(action_dim) if action_dim else -1.0

        self.buffer = ReplayBuffer(
            capacity=self.cfg.buffer_size,
            num_envs=1,
        )

        self._global_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def predict(
        self,
        obs: dict[str, torch.Tensor],
        hidden: dict | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict]:
        self.model.eval()
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden)
        actor_out = result["actor_out"]
        if isinstance(actor_out, dict):
            action = actor_out.get("mean", actor_out.get("action")) if deterministic else actor_out.get("action")
            log_prob = actor_out.get("log_prob")
        elif isinstance(actor_out, tuple):
            action, log_prob = actor_out
        else:
            action, log_prob = actor_out, None
        return action, log_prob, None, result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError("Use a TrainingLoop or call update() directly.")

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

        # --- Critic update ---
        with torch.no_grad():
            next_result = self.target_model(next_obs)
            next_actor_out = next_result["actor_out"]
            if isinstance(next_actor_out, dict):
                next_action = next_actor_out.get("action")
                next_log_prob = next_actor_out.get("log_prob")
            elif isinstance(next_actor_out, tuple):
                next_action, next_log_prob = next_actor_out
            else:
                next_action, next_log_prob = next_actor_out, torch.zeros(rewards.shape, device=self.device)

            next_q = self.target_model(next_obs, action=next_action)["value"]
            if isinstance(next_q, tuple):
                next_q = torch.min(*next_q)
            target_q = (
                rewards + self.cfg.gamma * (1.0 - dones) * (next_q - self.alpha * next_log_prob)
            )

        result = self.model(obs, action=actions)
        q_out = result["value"]
        if isinstance(q_out, tuple):
            q1, q2 = q_out
            critic_loss = sac_q_loss(q1, q2, target_q.detach())
        else:
            critic_loss = torch.nn.functional.mse_loss(q_out, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        result_actor = self.model(obs)
        actor_out = result_actor["actor_out"]
        if isinstance(actor_out, dict):
            new_action = actor_out.get("action")
            log_prob = actor_out.get("log_prob")
        elif isinstance(actor_out, tuple):
            new_action, log_prob = actor_out
        else:
            new_action, log_prob = actor_out, torch.zeros(rewards.shape, device=self.device)

        q_actor = self.model(obs, action=new_action)["value"]
        if isinstance(q_actor, tuple):
            q_actor = torch.min(*q_actor)

        actor_loss = sac_policy_loss(log_prob, q_actor, self.alpha.detach())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Temperature update ---
        temp_loss = sac_temperature_loss(log_prob.detach(), self.log_alpha, self.target_entropy)
        self.alpha_optimizer.zero_grad()
        temp_loss.backward()
        self.alpha_optimizer.step()

        # --- Soft update target ---
        _soft_update(self.model, self.target_model, self.cfg.tau)
        self._global_step += 1

        return {
            "sac/critic_loss": critic_loss.item(),
            "sac/actor_loss": actor_loss.item(),
            "sac/alpha": self.alpha.item(),
            "sac/temp_loss": temp_loss.item(),
        }

    def save(self, path: str) -> None:
        torch.save({
            "model": self.model.state_dict(),
            "target": self.target_model.state_dict(),
            "log_alpha": self.log_alpha.data,
            "step": self._global_step,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target"])
        self.log_alpha.data = ckpt["log_alpha"]
        self._global_step = ckpt.get("step", 0)


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data * tau)
