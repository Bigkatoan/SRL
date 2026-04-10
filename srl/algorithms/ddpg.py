"""DDPG (Deep Deterministic Policy Gradient) — off-policy, deterministic."""

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

        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.cfg.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=self.cfg.lr_critic
        )

        action_dim = self.cfg.action_dim or 1
        self.noise = OrnsteinUhlenbeckNoise(action_dim)

        self.buffer = ReplayBuffer(capacity=self.cfg.buffer_size, num_envs=1)
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
            next_q = self.target_model(next_obs, action=next_action)["value"]
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        q = self.model(obs, action=actions)["value"]
        critic_loss = ddpg_q_loss(q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor
        actor_out = self.model(obs)["actor_out"]
        if isinstance(actor_out, dict):
            new_action = actor_out.get("action")
        else:
            new_action = actor_out
        q_actor = self.model(obs, action=new_action)["value"]
        actor_loss = ddpg_policy_loss(q_actor)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        _soft_update(self.model, self.target_model, self.cfg.tau)
        self._global_step += 1

        return {
            "ddpg/critic_loss": critic_loss.item(),
            "ddpg/actor_loss": actor_loss.item(),
        }

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "target": self.target_model.state_dict(), "step": self._global_step}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target"])
        self._global_step = ckpt.get("step", 0)


def _soft_update(src, tgt, tau):
    for sp, tp in zip(src.parameters(), tgt.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data * tau)
