"""PPO (Proximal Policy Optimisation) — on-policy, continuous actions."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from srl.core.base_agent import OnPolicyAgent
from srl.core.config import PPOConfig
from srl.losses.aux_losses import reconstruction_loss
from srl.losses.loss_composer import LossComposer
from srl.losses.rl_losses import (
    entropy_loss,
    ppo_clip_loss,
    ppo_value_loss,
)
from srl.utils.normalizer import RunningNormalizer


class PPO(OnPolicyAgent):
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
    env:
        Optional training env (raw Gymnasium envs are wrapped).
    """

    algo_name = "ppo"
    config_cls = PPOConfig

    def __init__(
        self,
        model: nn.Module,
        config: PPOConfig | None = None,
        device: str | torch.device = "cpu",
        *,
        env=None,
    ) -> None:
        super().__init__(model, config, device, env=env)

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
            enc_lr = float(getattr(self.cfg, "encoder_lr", self.cfg.lr))
            if aux_params:
                self.encoder_optimizer = torch.optim.Adam(aux_params, lr=enc_lr)
                param_groups = [{"params": head_params, "lr": self.cfg.lr}]
                self._head_params = head_params
            else:
                # No auxiliary objective owns the encoders, so the RL loss
                # itself must train them (at `encoder_lr`). Without this the
                # encoders were excluded from every optimizer and frozen at
                # their random init for the whole run.
                self.encoder_optimizer = None
                head_ids = {id(p) for p in head_params}
                enc_params = [
                    p
                    for enc in self.model.encoders.values()
                    for p in enc.parameters()
                    if id(p) not in head_ids
                ]
                param_groups = [{"params": head_params, "lr": self.cfg.lr}]
                if enc_params:
                    param_groups.append(
                        {
                            "params": enc_params,
                            "lr": enc_lr,
                            # Keeps the encoder:head LR ratio fixed under
                            # lr_schedule="adaptive" (see _adapt_lr).
                            "lr_scale": enc_lr / float(self.cfg.lr),
                        }
                    )
                self._head_params = head_params + enc_params
            self.optimizer = torch.optim.Adam(param_groups, lr=self.cfg.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)
            self._head_params = list(self.model.parameters())
            self.encoder_optimizer = None

        # Detach encoder latents in the policy/value pass ONLY when a separate
        # auxiliary optimizer owns encoder training (VisualPPO with aux heads).
        # In every other case the PPO loss is the encoders' only training
        # signal -- detaching there silently froze them (verified dW == 0).
        self._detach_encoders = self.encoder_optimizer is not None

        self.buffer = self._build_rollout_buffer()

        # Tracks the live LR under lr_schedule="adaptive" -- self.cfg.lr stays
        # the *initial* value (also what a fresh --resume with no checkpoint
        # would start from); this is what actually drives the optimizer once
        # adaptation starts moving it away from that initial value. See
        # PPOConfig.lr_schedule's docstring for why this exists at all.
        self._current_lr = float(self.cfg.lr)

        # Optional observation/reward normalisation (PPOConfig.normalize_obs /
        # normalize_reward). Obs normalisers are created lazily per flat obs
        # key on first sight; the return normaliser tracks the running
        # variance of the per-env discounted return (VecNormalize).
        self._obs_normalizers: dict[str, RunningNormalizer] = {}
        self._obs_norm_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._ret_normalizer: RunningNormalizer | None = (
            RunningNormalizer(
                (1,), clip=self.cfg.clip_reward, eps=self.cfg.norm_epsilon, initial_count=1e-4
            )
            if self.cfg.normalize_reward
            else None
        )
        self._running_return: np.ndarray | None = None

        self.composer = LossComposer()
        self.composer.add("policy", weight=1.0)
        self.composer.add("value", weight=self.cfg.vf_coef)
        # `target_entropy` and the fixed/annealed `entropy_coef` are
        # mutually exclusive -- see PPOConfig.target_entropy's docstring.
        # When target_entropy is active, the entropy term is NOT
        # registered with the composer at all: its weight is a learned
        # parameter updated every minibatch in update(), which the
        # composer's step-driven schedule machinery has no slot for, so
        # it's applied manually instead (see update()).
        self._target_entropy_active = self.cfg.target_entropy is not None
        if self._target_entropy_active:
            self.log_entropy_coef = torch.nn.Parameter(
                torch.log(torch.tensor(max(self.cfg.entropy_coef, self.cfg.min_entropy_coef))),
            )
            self.entropy_coef_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                [self.log_entropy_coef], lr=self.cfg.entropy_coef_lr
            )
        else:
            self.log_entropy_coef = None
            self.entropy_coef_optimizer = None
            if self.cfg.entropy_coef_anneal_steps:
                # See PPOConfig.entropy_coef_anneal_steps's docstring for
                # why this exists. `LossComposer.add`'s `schedule`/
                # `total_steps`/`min_weight` machinery already implements
                # exactly this decay -- it just sat unused for this term
                # until now.
                self.composer.add(
                    "entropy",
                    weight=self.cfg.entropy_coef,
                    schedule="linear_decay",
                    total_steps=self.cfg.entropy_coef_anneal_steps,
                    min_weight=self.cfg.entropy_coef_final,
                )
            else:
                self.composer.add("entropy", weight=self.cfg.entropy_coef)

    # ------------------------------------------------------------------
    # Observation / reward normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_flat_obs(value: Any) -> bool:
        """Flat vector obs ((D,) or batched (N, D)); never images/uint8."""
        if isinstance(value, torch.Tensor):
            return value.ndim in (1, 2) and value.is_floating_point()
        arr = np.asarray(value)
        return arr.ndim in (1, 2) and arr.dtype != np.uint8

    def _obs_key_map(self, obs: dict) -> dict:
        """Remap raw env keys onto encoder names (idempotent), so the
        normalisers are keyed identically no matter whether a caller already
        remapped (the training loop does, eval paths may not)."""
        remap = getattr(self.model, "_remap_obs_dict", None)
        return remap(obs) if callable(remap) else obs

    def update_obs_normalizer(self, obs: dict[str, Any]) -> None:
        """Update the running obs statistics with one collected observation
        dict (numpy, unbatched (D,) or batched (N, D)). Call BEFORE
        :meth:`act` on that observation (VecNormalize order)."""
        if not self.cfg.normalize_obs:
            return
        for key, value in self._obs_key_map(obs).items():
            if not self._is_flat_obs(value):
                continue
            arr = np.asarray(value, dtype=np.float64)
            norm = self._obs_normalizers.get(key)
            if norm is None:
                norm = RunningNormalizer(
                    (arr.shape[-1],),
                    clip=self.cfg.clip_obs,
                    eps=self.cfg.norm_epsilon,
                    initial_count=1e-4,
                )
                self._obs_normalizers[key] = norm
            norm.update(arr.reshape(-1, arr.shape[-1]))
            self._obs_norm_cache.pop(key, None)

    def normalize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Normalise an obs dict (numpy arrays or tensors) with the current
        statistics, without updating them. Keys without statistics (images,
        unseen keys) pass through unchanged. No-op when normalize_obs=False."""
        if not self.cfg.normalize_obs or not self._obs_normalizers:
            return obs
        out = dict(self._obs_key_map(obs))
        for key, value in out.items():
            norm = self._obs_normalizers.get(key)
            if norm is None or not self._is_flat_obs(value):
                continue
            if isinstance(value, torch.Tensor):
                cached = self._obs_norm_cache.get(key)
                if cached is None or cached[0].device != value.device:
                    mean = torch.as_tensor(norm.mean, dtype=torch.float32, device=value.device)
                    std = torch.as_tensor(
                        np.sqrt(norm.var + norm.eps), dtype=torch.float32, device=value.device
                    )
                    cached = (mean, std)
                    self._obs_norm_cache[key] = cached
                mean, std = cached
                out[key] = torch.clamp((value - mean) / std, -norm.clip, norm.clip)
            else:
                out[key] = norm.normalize(np.asarray(value, dtype=np.float64)).astype(np.float32)
        return out

    def normalize_reward(self, reward: Any, episode_end: Any) -> np.ndarray:
        """Scale a per-env reward vector by the running std of the discounted
        return (VecNormalize). `episode_end` (terminated OR truncated) resets
        that env's running return AFTER this step's update. Must be called
        exactly once per collected step, on the RAW env reward. Returns the
        reward unchanged (as float32 array) when normalize_reward=False."""
        r = np.asarray(reward, dtype=np.float64).reshape(-1)
        if self._ret_normalizer is None:
            return r.astype(np.float32)
        if self._running_return is None or self._running_return.shape != r.shape:
            self._running_return = np.zeros_like(r)
        self._running_return = self._running_return * self.cfg.gamma + r
        self._ret_normalizer.update(self._running_return.reshape(-1, 1))
        std = float(np.sqrt(self._ret_normalizer.var[0] + self._ret_normalizer.eps))
        scaled = np.clip(r / std, -self.cfg.clip_reward, self.cfg.clip_reward)
        ended = np.asarray(episode_end, dtype=bool).reshape(-1)
        self._running_return[ended] = 0.0
        return scaled.astype(np.float32)

    def _normalizer_payload(self) -> dict[str, Any]:
        return {
            "obs": {k: v.state_dict() for k, v in self._obs_normalizers.items()},
            "ret": self._ret_normalizer.state_dict() if self._ret_normalizer is not None else None,
        }

    def _load_normalizer_payload(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        self._obs_normalizers = {}
        self._obs_norm_cache = {}
        for key, state in (payload.get("obs") or {}).items():
            norm = RunningNormalizer(
                tuple(np.asarray(state["mean"]).shape),
                clip=self.cfg.clip_obs,
                eps=self.cfg.norm_epsilon,
            )
            norm.load_state_dict(state)
            self._obs_normalizers[key] = norm
        ret_state = payload.get("ret")
        if ret_state is not None and self._ret_normalizer is not None:
            self._ret_normalizer.load_state_dict(ret_state)

    def _preprocess_obs(self, obs: dict) -> dict:
        # act() applies the running obs normalisation itself, so eval and
        # deployment see the same transform the policy was trained on.
        return self.normalize_obs(obs)

    def _adapt_lr(self, kl_mean: float) -> None:
        """Adapt `self._current_lr` from this minibatch's measured KL and push
        it onto every `self.optimizer` param group, mirroring rsl_rl PPO's
        `schedule="adaptive"` (`desired_kl` default 0.01 there too): too far
        above `desired_kl` and the last update was too aggressive, so shrink
        the LR; comfortably below it and updates are overly conservative, so
        grow it back. Runs every minibatch, for the whole run -- unlike
        `target_kl`'s same-epoch early stop, this actually keeps subsequent
        updates bounded instead of only reacting after one already overshot.
        """
        desired = self.cfg.desired_kl
        factor = self.cfg.kl_lr_factor
        if kl_mean > desired * 2.0:
            self._current_lr = max(self.cfg.min_lr, self._current_lr / factor)
        elif kl_mean < desired / 2.0 and kl_mean > 0.0:
            self._current_lr = min(self.cfg.max_lr, self._current_lr * factor)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self._current_lr * float(param_group.get("lr_scale", 1.0))

    def _adapt_entropy_coef(self, true_entropy: torch.Tensor) -> float:
        """Dual-ascent update of `self.log_entropy_coef` toward whatever
        value makes measured entropy track `self.cfg.target_entropy` --
        see PPOConfig.target_entropy's docstring for the full derivation.
        Only called when `self._target_entropy_active`. `true_entropy`
        must be the model's own real, correctly-signed mean entropy for
        this minibatch (already detached) -- never `entropy_loss`'s
        negated value.

        Returns the scalar dual loss (for logging), matching `_adapt_lr`'s
        sibling role: both are the "measure something this minibatch,
        nudge a scalar knob toward a target" adaptation, just for
        different knobs (LR vs. entropy_coef) and different measured
        quantities (KL vs. entropy).
        """
        entropy_coef_loss = self.log_entropy_coef * (true_entropy - self.cfg.target_entropy)
        self.entropy_coef_optimizer.zero_grad()
        entropy_coef_loss.backward()
        self.entropy_coef_optimizer.step()
        with torch.no_grad():
            self.log_entropy_coef.clamp_(
                math.log(self.cfg.min_entropy_coef),
                math.log(self.cfg.max_entropy_coef),
            )
        return float(entropy_coef_loss.item())

    def update(self) -> dict[str, float]:
        """Run *n_epochs × n_batches* gradient updates on the filled buffer."""
        self.model.train()
        metrics_accum: dict[str, list[float]] = {}
        stop_early = False

        for _ in range(self.cfg.n_epochs):
            for mini in self.buffer.get_batches(self.cfg.batch_size):
                obs = {k: v.to(self.device) for k, v in mini.obs.items()}
                actions = mini.actions.to(self.device)
                # Encoder latents are detached only when an aux optimizer owns
                # encoder training (see __init__'s _detach_encoders).
                # actor_action=actions: re-evaluate the SAME action taken during
                # rollout under the current policy (via the actor head's
                # evaluate_actions()) instead of a freshly resampled one -- the
                # PPO ratio exp(new_log_prob - old_log_prob) is only meaningful
                # when both log-probs refer to the same action.
                result = self.model(
                    obs, detach_encoders=self._detach_encoders, actor_action=actions
                )
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
                # clip_range_vf=None -> no value clipping (SB3 default).
                value_clip = self.cfg.clip_range_vf
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

                if self._target_entropy_active:
                    # Entropy term applied manually with the CURRENT learned
                    # coefficient (detached -- this parameter is trained by
                    # its own dual-ascent step below, never through the
                    # main policy/value backward pass), not through the
                    # composer's step-driven schedule machinery.
                    current_entropy_coef = self.log_entropy_coef.exp().detach()
                    total, info = self.composer.compute(
                        step=self._global_step, policy=pol_loss, value=val_loss
                    )
                    total = total + current_entropy_coef * ent_loss
                    info["entropy"] = float(ent_loss.item())
                    info["entropy_weight"] = float(current_entropy_coef.item())
                else:
                    total, info = self.composer.compute(
                        step=self._global_step,
                        policy=pol_loss,
                        value=val_loss,
                        entropy=ent_loss,
                    )
                info["approx_kl"] = float(approx_kl.item())

                # target_kl: `approx_kl` is measured on this minibatch BEFORE
                # its gradient step (old rollout policy vs. the current one),
                # so once it exceeds the target the policy has already moved
                # too far -- stop before applying another step (SB3/CleanRL
                # order). Stepping first and breaking after, as this loop
                # used to, always applied one extra update past the limit.
                if self.cfg.target_kl is not None and approx_kl.item() > self.cfg.target_kl:
                    for k, v in info.items():
                        metrics_accum.setdefault(k, []).append(v)
                    stop_early = True
                    break

                if self.cfg.lr_schedule == "adaptive":
                    self._adapt_lr(float(approx_kl.item()))
                info["lr"] = self._current_lr

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

                if self._target_entropy_active:
                    # `ent` is the model's own real, correctly-signed
                    # entropy tensor from THIS minibatch's forward pass
                    # above -- never pass entropy_loss's negated value
                    # here, that would silently re-introduce the exact
                    # sign confusion that motivated this feature.
                    entropy_coef_loss = self._adapt_entropy_coef(ent.detach().mean())
                    info["entropy_coef_loss"] = entropy_coef_loss
                    info["true_entropy"] = float(ent.detach().mean().item())

                for k, v in info.items():
                    metrics_accum.setdefault(k, []).append(v)
                self._global_step += 1
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

    def _algo_payload(self) -> dict[str, Any]:
        return {
            **self._base_payload(),
            "optimizer_state": self.optimizer.state_dict(),
            "encoder_optimizer_state": (
                self.encoder_optimizer.state_dict()
                if getattr(self, "encoder_optimizer", None) is not None
                else None
            ),
            "normalizers": self._normalizer_payload(),
            # optimizer_state above already carries each param group's
            # resumed "lr" -- this is saved/restored separately anyway
            # because _current_lr is PPO's own bookkeeping of that same
            # value (see _adapt_lr), read on the *next* adaptation before
            # the optimizer is touched. Without restoring it too, a resumed
            # run's first adapted step would compute from cfg.lr (the
            # *initial* value) instead of wherever the schedule had actually
            # drifted to, then stomp the correctly-resumed optimizer LR with
            # that wrong starting point.
            "current_lr": self._current_lr,
            # log_entropy_coef/its optimizer: only meaningful under
            # target_entropy (None otherwise) -- same "why save this
            # separately from the main model/optimizer state" reasoning
            # as current_lr above: a resumed run needs to continue from
            # wherever the dual-ascent adaptation actually left the
            # coefficient, not restart it from cfg.entropy_coef.
            "log_entropy_coef": (
                self.log_entropy_coef.detach().cpu() if self.log_entropy_coef is not None else None
            ),
            "entropy_coef_optimizer_state": (
                self.entropy_coef_optimizer.state_dict()
                if self.entropy_coef_optimizer is not None
                else None
            ),
        }

    def _load_algo_payload(self, payload: dict[str, Any]) -> None:
        self._load_base_payload(payload)
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        encoder_opt_state = payload.get("encoder_optimizer_state")
        if encoder_opt_state is not None and getattr(self, "encoder_optimizer", None) is not None:
            self.encoder_optimizer.load_state_dict(encoder_opt_state)
        self._load_normalizer_payload(payload.get("normalizers"))
        if "current_lr" in payload:
            self._current_lr = float(payload["current_lr"])
        log_entropy_coef = payload.get("log_entropy_coef")
        if log_entropy_coef is not None and self.log_entropy_coef is not None:
            with torch.no_grad():
                self.log_entropy_coef.copy_(log_entropy_coef.to(self.device))
        entropy_coef_opt_state = payload.get("entropy_coef_optimizer_state")
        if entropy_coef_opt_state is not None and self.entropy_coef_optimizer is not None:
            self.entropy_coef_optimizer.load_state_dict(entropy_coef_opt_state)
