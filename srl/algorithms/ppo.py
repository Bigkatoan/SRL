"""PPO (Proximal Policy Optimisation) — on-policy, continuous actions."""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

from srl.core.base_agent import BaseAgent
from srl.core.config import PPOConfig
from srl.core.rollout_buffer import RolloutBuffer
from srl.losses.aux_losses import reconstruction_loss
from srl.losses.loss_composer import LossComposer
from srl.losses.rl_losses import (
    entropy_loss,
    ppo_clip_loss,
    ppo_value_loss,
)
from srl.utils.checkpoint import CheckpointManager
from srl.utils.normalizer import RunningNormalizer


class PPO(BaseAgent):
    """Proximal Policy Optimisation.

    Parameters
    ----------
    model:
        :class:`~srl.networks.agent_model.AgentModel` with a Gaussian /
        SquashedGaussian actor and a Value critic.
    config:
        :class:`~srl.core.config.PPOConfig`.
    device:
        PyTorch device.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.cfg = config or PPOConfig()
        self._device = torch.device(device)
        self.model.to(self._device)

        # If a config exposes a separate encoder learning rate (VisualPPOConfig),
        # create a head-only optimizer (actor+critic) and optionally a separate
        # encoder/aux optimizer that will be used only for auxiliary losses.
        if hasattr(self.cfg, "encoder_lr"):
            head_params = []
            if getattr(self.model, "actor", None) is not None:
                head_params += list(self.model.actor.parameters())
            if getattr(self.model, "critic", None) is not None:
                head_params += list(self.model.critic.parameters())
            # Fallback: if no actor/critic present, fall back to all parameters
            if not head_params:
                head_params = list(self.model.parameters())

            self.optimizer = torch.optim.Adam(head_params, lr=self.cfg.lr, eps=1e-5)
            self._head_params = head_params

            # Build encoder/aux optimizer when aux modules exist
            aux_params: list = []
            if getattr(self.model, "aux_modules", None):
                for aux in self.model.aux_modules.values():
                    aux_params += list(aux.parameters())
                # Include encoders corresponding to aux modules (names like '<enc>_aux')
                for aux_name in self.model.aux_modules.keys():
                    if aux_name.endswith("_aux"):
                        enc_name = aux_name[:-4]
                        # nn.ModuleDict has no .get() -- membership + indexing instead.
                        enc = (
                            self.model.encoders[enc_name]
                            if enc_name in self.model.encoders
                            else None
                        )
                        if enc is not None:
                            aux_params += list(enc.parameters())
            if aux_params:
                enc_lr = getattr(self.cfg, "encoder_lr", self.cfg.lr)
                self.encoder_optimizer = torch.optim.Adam(aux_params, lr=enc_lr)
            else:
                self.encoder_optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
            self._head_params = list(self.model.parameters())
            self.encoder_optimizer = None

        self.buffer = RolloutBuffer(
            capacity=self.cfg.n_steps,
            num_envs=getattr(self.cfg, "num_envs", 1),
            gamma=self.cfg.gamma,
            lam=self.cfg.gae_lambda,
            device=self.device,
        )

        self.checkpoint_manager: CheckpointManager | None = None
        self._global_step = 0

        # Optional observation/reward normalisation
        self._obs_normalizer: RunningNormalizer | None = None
        self._ret_normalizer: RunningNormalizer | None = None

        self.composer = LossComposer()
        self.composer.add("policy", weight=1.0)
        self.composer.add("value", weight=self.cfg.vf_coef)
        self.composer.add("entropy", weight=self.cfg.entropy_coef)

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

        # Actor out: dict from GaussianActorHead/SquashedGaussianActorHead,
        # or (action, log_prob) tuple from older heads
        if isinstance(actor_out, dict):
            if deterministic:
                action = actor_out.get("mean", actor_out.get("action"))
            else:
                action = actor_out.get("action")
            log_prob = actor_out.get("log_prob")
        elif isinstance(actor_out, tuple):
            action, log_prob = actor_out
        else:
            action, log_prob = actor_out, None

        value = result["value"]
        return action, log_prob, value, result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError(
            "Call PPO via a TrainingLoop or pass pre-collected batches to update()."
        )

    def update(self) -> dict[str, float]:
        """Run *n_epochs × n_batches* gradient updates on the filled buffer."""
        self.model.train()
        metrics_accum: dict[str, list[float]] = {}
        stop_early = False

        for _ in range(self.cfg.n_epochs):
            for mini in self.buffer.get_batches(self.cfg.batch_size):
                obs = {k: v.to(self.device) for k, v in mini.obs.items()}
                actions = mini.actions.to(self.device)
                # Detach encoder latents so policy/value updates do not backpropagate
                # into encoder parameters (encoders are updated only via aux optimizer).
                # actor_action=actions: re-evaluate the SAME action taken during
                # rollout under the current policy (via the actor head's
                # evaluate_actions()) instead of a freshly resampled one -- the
                # PPO ratio exp(new_log_prob - old_log_prob) is only meaningful
                # when both log-probs refer to the same action.
                result = self.model(obs, detach_encoders=True, actor_action=actions)
                actor_out = result["actor_out"]

                if isinstance(actor_out, dict):
                    log_prob_eval = actor_out.get("log_prob")
                    ent_raw = actor_out.get("entropy")
                    if ent_raw is None:
                        dist = actor_out.get("dist")
                        ent_raw = (
                            dist.entropy()
                            if dist is not None
                            else torch.zeros(1, device=self.device)
                        )
                elif isinstance(actor_out, tuple):
                    _, log_prob_eval = actor_out
                    ent_raw = torch.zeros(1, device=self.device)
                else:
                    log_prob_eval = actor_out
                    ent_raw = torch.zeros(1, device=self.device)
                ent = ent_raw

                adv = mini.advantages.to(self.device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pol_loss = ppo_clip_loss(
                    log_prob_eval,
                    mini.log_probs.to(self.device),
                    adv,
                    clip_eps=self.cfg.clip_range,
                )
                value_clip = (
                    self.cfg.clip_range_vf
                    if self.cfg.clip_range_vf is not None
                    else self.cfg.clip_range
                )
                val_loss = ppo_value_loss(
                    result["value"].squeeze(-1),
                    mini.returns.to(self.device),
                    old_values=mini.values.to(self.device),
                    clip_eps=value_clip,
                )
                ent_loss = entropy_loss(ent)
                with torch.no_grad():
                    log_ratio = log_prob_eval - mini.log_probs.to(self.device)
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()

                total, info = self.composer.compute(
                    step=self._global_step,
                    policy=pol_loss,
                    value=val_loss,
                    entropy=ent_loss,
                )
                info["approx_kl"] = float(approx_kl.item())

                self.optimizer.zero_grad()
                total.backward()
                # Clip grads only for head params
                try:
                    nn.utils.clip_grad_norm_(self._head_params, self.cfg.max_grad_norm)
                except Exception as exc:
                    # Fallback to full model params if head list unavailable
                    warnings.warn(
                        f"Gradient clipping on _head_params failed ({exc!r}); "
                        "clipping all model parameters instead.",
                        stacklevel=2,
                    )
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                for k, v in info.items():
                    metrics_accum.setdefault(k, []).append(v)
                self._global_step += 1
                if self.cfg.target_kl is not None and approx_kl.item() > self.cfg.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

            # --- Auxiliary losses / encoder updates ---
            # Run auxiliary reconstruction losses (if any) and update encoders separately.
            if getattr(self, "encoder_optimizer", None) is not None and getattr(
                self.model, "aux_modules", None
            ):
                # Build a batch-level aux loss across available aux heads
                aux_loss: torch.Tensor | None = None
                for aux_name, aux_module in self.model.aux_modules.items():
                    # Expect aux_name like '<enc_name>_aux'
                    enc_name = aux_name[:-4] if aux_name.endswith("_aux") else None
                    # Resolve associated observation key
                    input_key = None
                    if enc_name and hasattr(self.model, "encoder_input_names"):
                        input_key = self.model.encoder_input_names.get(enc_name)
                    if input_key is None:
                        if enc_name and enc_name in mini.obs:
                            input_key = enc_name
                        elif len(mini.obs) == 1:
                            input_key = next(iter(mini.obs.keys()))
                        else:
                            # Cannot determine associated obs for this aux head
                            continue
                    obs_tensor = mini.obs.get(input_key)
                    if obs_tensor is None:
                        continue
                    obs_tensor = obs_tensor.to(self.device)
                    # Encoder forward (will create grads for encoder parameters)
                    # nn.ModuleDict has no .get() -- membership + indexing instead.
                    enc = (
                        self.model.encoders[enc_name]
                        if enc_name and enc_name in self.model.encoders
                        else None
                    )
                    if enc is None:
                        continue
                    latent = enc(obs_tensor)
                    # Aux head forward
                    recon = aux_module(latent)
                    # Try compute reconstruction loss if shapes match
                    try:
                        loss = reconstruction_loss(recon, obs_tensor)
                    except Exception as exc:
                        warnings.warn(
                            f"Skipping aux loss for '{aux_name}': reconstruction_loss "
                            f"raised {exc!r} (unsupported aux head / loss combination, "
                            "or a shape mismatch worth checking).",
                            stacklevel=2,
                        )
                        continue
                    aux_loss = loss if aux_loss is None else aux_loss + loss

                if aux_loss is not None:
                    weight = float(getattr(self.cfg, "aux_weight", 1.0))
                    total_aux = aux_loss * weight
                    # Surface the aux loss so it is observable in the console /
                    # TensorBoard metrics -- otherwise there is no way to tell
                    # from a run whether the encoder is actually learning.
                    metrics_accum.setdefault("aux_loss", []).append(float(aux_loss.item()))
                    metrics_accum.setdefault("aux_loss_weighted", []).append(
                        float(total_aux.item())
                    )
                    self.encoder_optimizer.zero_grad()
                    total_aux.backward()
                    # Clip encoder/aux gradients (use params from optimizer)
                    try:
                        enc_params = self.encoder_optimizer.param_groups[0]["params"]
                        nn.utils.clip_grad_norm_(enc_params, self.cfg.max_grad_norm)
                    except Exception as exc:
                        # Not fatal (the un-clipped step below still runs), but
                        # worth surfacing -- an unclipped encoder step can let
                        # gradient explosions go unnoticed.
                        warnings.warn(
                            f"Gradient clipping on encoder_optimizer params failed "
                            f"({exc!r}); proceeding with an unclipped encoder step.",
                            stacklevel=2,
                        )
                    self.encoder_optimizer.step()

        self.buffer.reset()
        return {k: sum(v) / len(v) for k, v in metrics_accum.items()}

    def save(self, path: str) -> None:
        import torch

        torch.save(self.checkpoint_payload(), path)

    def load(self, path: str) -> None:
        import torch

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_checkpoint_payload(ckpt)

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "encoder_optimizer_state": (
                self.encoder_optimizer.state_dict()
                if getattr(self, "encoder_optimizer", None) is not None
                else None
            ),
            "algo_step": self._global_step,
        }

    def load_checkpoint_payload(self, payload: dict[str, Any]) -> None:
        model_state = payload.get("model_state", payload.get("model"))
        if model_state is not None:
            self.model.load_state_dict(model_state)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        encoder_opt_state = payload.get("encoder_optimizer_state")
        if encoder_opt_state is not None and getattr(self, "encoder_optimizer", None) is not None:
            self.encoder_optimizer.load_state_dict(encoder_opt_state)
        self._global_step = int(payload.get("algo_step", payload.get("step", 0)))
