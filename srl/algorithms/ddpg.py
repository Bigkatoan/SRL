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

import threading

import numpy as np
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
from srl.core.config import DDPGConfig
from srl.core.prioritized_replay_buffer import build_replay_buffer
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
        self.target_model.eval()
        self._model_lock = threading.Lock()

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

        action_dim = self.cfg.action_dim or 1
        if self.cfg.action_noise == "ou":
            self.noise = OrnsteinUhlenbeckNoise(action_dim, sigma=self.cfg.noise_sigma)
        else:
            self.noise = GaussianActionNoise(action_dim, sigma=self.cfg.noise_sigma)

        self.buffer = build_replay_buffer(self.cfg)
        self.obs_normalizer = maybe_build_obs_normalizer(self.cfg)
        self._global_step = 0

    def _forward_model(self, mode: str, *args, **kwargs):
        return forward_with_mode(self.model, self._model_lock, mode, *args, **kwargs)

    def reset_noise(self, env_ids=None) -> None:
        """Reset exploration noise for the envs whose episode just ended.

        *env_ids*: ``None`` (all), a boolean mask over envs, or indices.
        """
        self.noise.reset(env_ids)

    def predict(self, obs, hidden=None, deterministic=False):
        with torch.no_grad():
            obs = prepare_obs(self.model, self.obs_normalizer, obs)
            result = self._forward_model("eval", obs, hidden_states=hidden, compute_critic=False)
        actor_out = result["actor_out"]
        if isinstance(actor_out, dict):
            action = actor_out.get("action")
        else:
            action = actor_out
        if not deterministic and action is not None:
            if action.dim() > 1:
                noise = self.noise.sample(action.shape[0])
            else:
                noise = self.noise.sample()
            action = (action + noise.to(device=action.device, dtype=action.dtype)).clamp(-1.0, 1.0)
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

        _soft_update(self.model, self.target_model, self.cfg.tau)
        self._global_step += 1

        return {
            "ddpg/critic_loss": critic_loss.item(),
            "ddpg/actor_loss": actor_loss.item(),
            "ddpg/target_q_mean": target_q.mean().item(),
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
