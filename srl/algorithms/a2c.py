"""A2C (Advantage Actor-Critic) — synchronous on-policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.core.base_agent import BaseAgent
from srl.core.config import A2CConfig
from srl.core.rollout_buffer import RolloutBuffer
from srl.losses.loss_composer import LossComposer
from srl.losses.rl_losses import a2c_policy_loss, a2c_value_loss, entropy_loss


class A2C(BaseAgent):
    """Synchronous Advantage Actor-Critic."""

    def __init__(
        self,
        model: nn.Module,
        config: A2CConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.cfg = config or A2CConfig()
        self._device = torch.device(device)
        self.model.to(self._device)

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.cfg.lr,
            alpha=0.99,
            eps=self.cfg.rms_prop_eps,
        )

        self.buffer = RolloutBuffer(
            capacity=self.cfg.n_steps,
            num_envs=getattr(self.cfg, "num_envs", 1),
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
            device=self.device,
        )

        self.composer = LossComposer()
        self.composer.add("policy", weight=1.0)
        self.composer.add("value", weight=self.cfg.vf_coef)
        self.composer.add("entropy", weight=self.cfg.entropy_coef)

        self._global_step = 0

    def predict(self, obs, hidden=None, deterministic=False):
        self.model.eval()
        with torch.no_grad():
            result = self.model(obs, hidden_states=hidden)
        actor_out = result["actor_out"]
        if isinstance(actor_out, dict):
            action = (
                actor_out.get("mean", actor_out.get("action"))
                if deterministic
                else actor_out.get("action")
            )
            log_prob = actor_out.get("log_prob")
        elif isinstance(actor_out, tuple):
            action, log_prob = actor_out
        else:
            action, log_prob = actor_out, None
        return action, log_prob, result["value"], result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError

    def update(self) -> dict[str, float]:
        self.model.train()
        metrics_accum: dict[str, list] = {}

        for mini in self.buffer.get_batches(self.cfg.batch_size):
            obs = {k: v.to(self.device) for k, v in mini.obs.items()}
            actions = mini.actions.to(self.device)
            # actor_action=actions: re-evaluate the action actually taken under
            # the current policy, not a freshly resampled one -- see PPO.update()
            # for why (same actor-head evaluate_actions() mechanism).
            result = self.model(obs, actor_action=actions)
            actor_out = result["actor_out"]

            if isinstance(actor_out, dict):
                log_prob = actor_out.get("log_prob")
                ent = actor_out.get("entropy")
                if ent is None:
                    dist = actor_out.get("dist")
                    ent = dist.entropy() if dist is not None else torch.zeros(1, device=self.device)
            elif isinstance(actor_out, tuple):
                _, log_prob = actor_out
                ent = torch.zeros(1, device=self.device)
            else:
                log_prob = actor_out
                ent = torch.zeros(1, device=self.device)

            adv = mini.advantages.to(self.device)

            pol_loss = a2c_policy_loss(log_prob, adv)
            val_loss = a2c_value_loss(result["value"].squeeze(-1), mini.returns.to(self.device))
            ent_loss = entropy_loss(ent)

            total, info = self.composer.compute(
                step=self._global_step,
                policy=pol_loss,
                value=val_loss,
                entropy=ent_loss,
            )

            self.optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            self._global_step += 1

            for k, v in info.items():
                metrics_accum.setdefault(k, []).append(v)

        self.buffer.reset()
        return {k: sum(v) / len(v) for k, v in metrics_accum.items()}

    def save(self, path: str) -> None:
        torch.save(self.checkpoint_payload(), path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_checkpoint_payload(ckpt)

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "algo_step": self._global_step,
        }

    def load_checkpoint_payload(self, payload: dict[str, object]) -> None:
        model_state = payload.get("model_state", payload.get("model"))
        if model_state is not None:
            self.model.load_state_dict(model_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self._global_step = int(payload.get("algo_step", payload.get("step", 0)))
