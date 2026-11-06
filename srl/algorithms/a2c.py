"""A2C (Advantage Actor-Critic) — synchronous on-policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from srl.core.base_agent import OnPolicyAgent
from srl.core.config import A2CConfig
from srl.losses.loss_composer import LossComposer
from srl.losses.rl_losses import a2c_policy_loss, a2c_value_loss, entropy_loss


class A2C(OnPolicyAgent):
    """Synchronous Advantage Actor-Critic.

    Parameters
    ----------
    model:
        :class:`~srl.networks.agent_model.AgentModel` with a Gaussian actor
        and a Value critic.
    config:
        :class:`~srl.core.config.A2CConfig`.
    device:
        PyTorch device.
    env:
        Optional training env (raw Gymnasium envs are wrapped).
    """

    algo_name = "a2c"
    config_cls = A2CConfig

    def __init__(
        self,
        model: nn.Module,
        config: A2CConfig | None = None,
        device: str | torch.device = "cpu",
        *,
        env=None,
    ) -> None:
        super().__init__(model, config, device, env=env)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.cfg.lr,
            alpha=0.99,
            eps=self.cfg.rms_prop_eps,
        )
        self.buffer = self._build_rollout_buffer()

        self.composer = LossComposer()
        self.composer.add("policy", weight=1.0)
        self.composer.add("value", weight=self.cfg.vf_coef)
        self.composer.add("entropy", weight=self.cfg.entropy_coef)

    def update(self) -> dict[str, float]:
        self.model.train()
        metrics_accum: dict[str, list] = {}

        # A2C (unlike PPO) takes exactly one synchronous gradient step over
        # the *entire* just-collected rollout -- there is no clipped
        # surrogate objective here to keep repeated minibatch updates safe
        # against a policy that has already moved since the advantages were
        # computed. `self.cfg.batch_size` doesn't scale with `num_envs`
        # (rollout size is `n_steps * num_envs`), so using it directly could
        # split one rollout into many sequential updates against
        # increasingly stale advantages/log-probs -- confirmed by direct
        # repro to blow up policy loss within a few thousand steps at
        # num_envs=16 with the documented default `batch_size=5`. Always use
        # the full rollout as a single batch, matching the canonical
        # algorithm; `batch_size` is kept on `A2CConfig` for backward
        # compatibility but is no longer read here.
        full_batch_size = self.cfg.n_steps * max(getattr(self.cfg, "num_envs", 1), 1)
        for mini in self.buffer.get_batches(full_batch_size):
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

    def _algo_payload(self) -> dict[str, object]:
        return {**self._base_payload(), "optimizer_state": self.optimizer.state_dict()}

    def _load_algo_payload(self, payload: dict[str, object]) -> None:
        self._load_base_payload(payload)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
