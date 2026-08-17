"""SAC (Soft Actor-Critic) — off-policy, continuous actions.

Changes in v0.2.0
-----------------
* Three separate optimizers: critic_optimizer, actor_optimizer, encoder_optimizer.
  Encoder is updated only through critic backward (DrQ-v2 style) — actor backward
  never touches encoder weights.  This eliminates the effective 2× encoder LR bug.
* encoder_update_freq (SACConfig): encoder_optimizer steps every N critic updates.
  Default 1 for state tasks (no change); VisualSACConfig overrides to 2.
* encoder_optimize_with_critic (VisualSACConfig): when False encoder is fully
  detached from critic backward — learns only via aux_loss_type.
* aux_loss_type (VisualSACConfig): "none"|"ae"|"vae"|"curl"|"byol"|"drq"|"spr"|"barlow".
  Aux gradients accumulate on encoder params across freq steps and are consumed
  together with (or instead of) critic gradients when encoder_optimizer.step() fires.

Changes since v0.2.0
---------------------
* Fixed: an encoder that feeds ONLY the actor (a `flows:` graph with separate
  per-head encoders, e.g. `["actor_state_enc -> actor", "critic_state_enc ->
  critic"]`) used to never receive a gradient update that any optimizer
  actually consumed -- critic_loss's backward pass never touches it
  (compute_actor=False forward passes correctly prune encoders that don't
  feed the critic), and the one place a real gradient into it existed
  (actor_loss's own backward pass) was unconditionally zeroed before
  encoder_optimizer.step() ever ran again. Confirmed directly: such an
  encoder's weights never moved from random init across real update() calls.
  Now split via `_partition_encoder_params`: critic-reachable encoders keep
  the original "trained only through critic backward" behaviour unchanged;
  actor-only encoders get their own `actor_only_encoder_optimizer`, stepped
  off actor_loss's backward pass (their only valid signal) instead.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from srl.core.base_agent import BaseAgent
from srl.core.config import SACConfig, VisualSACConfig
from srl.core.replay_buffer import ReplayBuffer
from srl.losses.aux_losses import (
    barlow_twins_loss,
    byol_loss,
    drq_aug_loss,
    info_nce_loss,
    reconstruction_loss,
    spr_loss,
    vae_loss,
)
from srl.losses.rl_losses import (
    sac_policy_loss,
    sac_q_loss,
    sac_temperature_loss,
)
from srl.networks.encoders.augmentations import augment


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
        :class:`~srl.core.config.SACConfig` or
        :class:`~srl.core.config.VisualSACConfig`.
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

        # ------------------------------------------------------------------
        # Three-optimizer design (v0.2.0), split by reachability (v0.2.1)
        # ------------------------------------------------------------------
        # `_encoder_param_list` used to be every encoder parameter,
        # unconditionally trained through `encoder_optimizer` off of
        # critic_loss's backward pass alone (DrQ-v2 style: "encoder is
        # updated only through critic backward"). That rule is correct for
        # a SHARED encoder (or a critic-only one) -- but for a config with
        # a SEPARATE, actor-only encoder (e.g. a `flows:` graph like
        # `["actor_state_enc -> actor", "critic_state_enc -> critic"]`,
        # each head with its own dedicated encoder), it silently starves
        # that encoder of any training signal at all: critic_loss's
        # backward pass never touches it (compute_actor=False forward
        # passes correctly prune encoders that don't feed the critic, so
        # it's never even part of that graph), and the ONE place a real
        # gradient into it exists -- actor_loss's own backward pass, which
        # does run through it -- gets explicitly zeroed out before any
        # optimizer.step() ever consumes it (see [5]/[8] below), by design,
        # to keep actor_loss from contaminating a *shared* encoder's critic
        # -driven training. For an actor-only encoder that rule has no
        # shared encoder to protect, and left unconditionally applied it
        # means the encoder's weights never move from random init --
        # confirmed directly: 20 real `update()` calls against a tiny
        # actor-only-encoder fixture left every one of its parameter
        # tensors bit-for-bit identical to their initial values, while the
        # critic's own (correctly-trained) encoder moved normally.
        #
        # Fix: partition encoder params by whether they're reachable from
        # the critic head (`encoder_names_for_head`, which already exists
        # for exactly this "which encoders does this head actually need"
        # question). Critic-reachable (shared or critic-only) encoders keep
        # the original behaviour untouched -- `encoder_optimizer` trained
        # off critic_loss alone, actor_loss's contribution zeroed before it
        # can leak in. Actor-only encoders get their own dedicated
        # `actor_only_encoder_optimizer`, stepped off actor_loss's backward
        # pass (their only valid signal) instead of being zeroed.
        _critic_reachable, _actor_only, _actor_only_modules = _partition_encoder_params(self.model)
        self._encoder_param_list: list[nn.Parameter] = _critic_reachable
        self._actor_only_encoder_param_list: list[nn.Parameter] = _actor_only
        # Modules (not just flat params) for the actor-only subset -- kept
        # in case a subclass or future feature needs to walk module
        # structure (e.g. weight-norm-style projection) rather than a flat
        # parameter list.
        self._actor_only_encoder_modules: list[nn.Module] = _actor_only_modules

        # encoder_lr: read from VisualSACConfig if available, else fall back to
        # lr_critic (conservative default for state-based configs).
        _encoder_lr = getattr(self.cfg, "encoder_lr", self.cfg.lr_critic)

        self.encoder_optimizer: torch.optim.Optimizer | None = (
            torch.optim.Adam(self._encoder_param_list, lr=_encoder_lr)
            if self._encoder_param_list
            else None
        )
        self.actor_only_encoder_optimizer: torch.optim.Optimizer | None = (
            torch.optim.Adam(self._actor_only_encoder_param_list, lr=_encoder_lr)
            if self._actor_only_encoder_param_list
            else None
        )

        # Actor head params only — encoder excluded
        self.actor_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters()),
            lr=self.cfg.lr_actor,
        )
        # Critic head params only — encoder excluded
        self.critic_optimizer = torch.optim.Adam(
            list(self.model.critic.parameters()),
            lr=self.cfg.lr_critic,
        )

        self._encoder_update_counter: int = 0

        # Visual / aux settings (only present in VisualSACConfig)
        self._is_visual: bool = isinstance(self.cfg, VisualSACConfig)
        self._aux_loss_type: str = getattr(self.cfg, "aux_loss_type", "none")
        self._aux_weight: float = getattr(self.cfg, "aux_weight", 0.0)
        self._aug_mode: str = getattr(self.cfg, "augmentation_mode", "curl")
        self._enc_with_critic: bool = getattr(self.cfg, "encoder_optimize_with_critic", True)

        # ------------------------------------------------------------------
        # Automatic entropy tuning
        # ------------------------------------------------------------------
        alpha_value = self.cfg.alpha if self.cfg.alpha is not None else self.cfg.init_alpha
        init_alpha = max(float(alpha_value), 1e-8)
        self.log_alpha = torch.tensor(
            [math.log(init_alpha)],
            requires_grad=self.cfg.auto_entropy_tuning,
            device=self._device,
        )
        self.alpha_optimizer = (
            torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)
            if self.cfg.auto_entropy_tuning
            else None
        )
        action_dim = self.cfg.action_dim
        if self.cfg.target_entropy == "auto":
            self.target_entropy = -float(action_dim) if action_dim else -1.0
        else:
            self.target_entropy = float(self.cfg.target_entropy)

        if getattr(self.cfg, "use_her", False):
            self.buffer = self._build_her_buffer()
        else:
            self.buffer = ReplayBuffer(
                capacity=self.cfg.buffer_size,
                num_envs=self.cfg.replay_num_envs,
                n_step=self.cfg.replay_n_step,
                gamma=self.cfg.gamma,
                use_fp16=self.cfg.use_fp16,
            )

        self._global_step = 0

    def _build_her_buffer(self):
        """Construct the HER buffer for a goal-conditioned run.

        ``her_obs_dim`` / ``her_goal_dim`` / ``her_reward_fn`` are environment
        facts, so the CLI fills them in on the config before constructing the
        agent (see ``srl/cli/train.py``). Building SAC with ``use_her=True``
        but without them is a wiring mistake, not a recoverable state -- HER
        cannot relabel without a goal split, and silently falling back to a
        plain ReplayBuffer is exactly the failure mode this feature exists to
        remove -- so fail loudly.
        """
        from srl.core.her_replay_buffer import HERReplayBuffer

        obs_dim = int(getattr(self.cfg, "her_obs_dim", 0) or 0)
        goal_dim = int(getattr(self.cfg, "her_goal_dim", 0) or 0)
        reward_fn = getattr(self.cfg, "her_reward_fn", None)
        missing = [
            name
            for name, value in (
                ("her_obs_dim", obs_dim),
                ("her_goal_dim", goal_dim),
                ("her_reward_fn", reward_fn),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "SACConfig.use_her=True requires " + ", ".join(missing) + " to be set. "
                "These are derived from the environment; use a goal config "
                '(env_type: "goal") via srl-train, which fills them in, or set '
                "them yourself when constructing SAC directly."
            )
        return HERReplayBuffer(
            capacity=self.cfg.buffer_size,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            action_dim=int(self.cfg.action_dim),
            reward_fn=reward_fn,
            strategy=str(getattr(self.cfg, "her_strategy", "future")),
            her_ratio=float(getattr(self.cfg, "her_ratio", 0.8)),
            max_episode_len=int(getattr(self.cfg, "her_max_episode_len", 1000)),
            gamma=self.cfg.gamma,
            device="cpu",
        )

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
            # compute_critic=False: predict() is the env-collection path --
            # called on every single environment step -- and never reads
            # result["value"] (see the `None` returned for it below). Without
            # the flag every action-selection call would also run the critic
            # encoder(s) and a full TwinQHead forward on a dummy zero action,
            # for a result nothing downstream uses.
            result = self.model(obs, hidden_states=hidden, compute_critic=False)
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
        return action, log_prob, None, result["new_hidden"]

    def learn(self, total_timesteps: int) -> None:
        raise NotImplementedError("Use a TrainingLoop or call update() directly.")

    def update(self) -> dict[str, float]:
        # HERReplayBuffer is episode-indexed, so its __len__ counts episodes,
        # not transitions -- gating it on `len(buffer) >= batch_size` would
        # stall training for batch_size *episodes*. It exposes can_sample()
        # for exactly this reason; plain ReplayBuffer keeps the length check.
        _can_sample = getattr(self.buffer, "can_sample", None)
        if _can_sample is not None:
            if not _can_sample(self.cfg.batch_size):
                return {}
        elif len(self.buffer) < self.cfg.batch_size:
            return {}

        self.model.train()
        batch = self.buffer.sample(self.cfg.batch_size)

        obs = {k: v.to(self.device) for k, v in batch.obs.items()}
        next_obs = {k: v.to(self.device) for k, v in batch.next_obs.items()}
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        dones = batch.dones.to(self.device)

        freq = self.cfg.encoder_update_freq

        # ------------------------------------------------------------------
        # [1] Zero encoder grads — clean slate for this update step
        # ------------------------------------------------------------------
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad()
        if self.actor_only_encoder_optimizer is not None:
            self.actor_only_encoder_optimizer.zero_grad()

        # ------------------------------------------------------------------
        # [2] Critic forward + backward
        #     Encoder receives gradients from critic iff enc_with_critic=True.
        # ------------------------------------------------------------------
        with torch.no_grad():
            # compute_critic=False: only next_action/next_log_prob are used
            # from this call. Without the flag, AgentModel.forward() would
            # also run the critic encoder(s) and a full TwinQHead forward on
            # a dummy zero action just to throw the result away below.
            next_result = self.model(next_obs, compute_critic=False)
            next_actor_out = next_result["actor_out"]
            if isinstance(next_actor_out, dict):
                next_action = next_actor_out.get("action")
                next_log_prob = next_actor_out.get("log_prob")
            elif isinstance(next_actor_out, tuple):
                next_action, next_log_prob = next_actor_out
            else:
                next_action, next_log_prob = next_actor_out, torch.zeros(
                    rewards.shape, device=self.device
                )

            # compute_actor=False: next_action was already sampled from the
            # (non-target) actor above; the target network's own actor head
            # is never consulted in SAC, so running it here would be pure
            # waste discarded a line later.
            next_q = self.target_model(next_obs, action=next_action, compute_actor=False)["value"]
            if isinstance(next_q, tuple):
                next_q = torch.min(*next_q)
            target_q = rewards + self.cfg.gamma * (1.0 - dones) * (
                next_q - self.alpha.detach() * next_log_prob
            )

        # When encoder should NOT receive gradients from critic, detach obs
        # pixels/features before the critic forward pass.
        if not self._enc_with_critic and self._encoder_param_list:
            obs_for_critic = _detach_visual_obs(obs)
        else:
            obs_for_critic = obs

        # compute_actor=False: this call only needs the critic's Q(s, a);
        # the actor head's output on obs_for_critic is never read below.
        result = self.model(obs_for_critic, action=actions, compute_actor=False)
        q_out = result["value"]
        if isinstance(q_out, tuple):
            q1, q2 = q_out
            critic_loss = sac_q_loss(q1, q2, target_q.detach())
        else:
            critic_loss = F.mse_loss(q_out, target_q.detach())

        # [2b] Backward through critic (+ encoder if enc_with_critic)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # ------------------------------------------------------------------
        # [3] Critic head step (encoder step deferred to [4])
        # ------------------------------------------------------------------
        self.critic_optimizer.step()

        # ------------------------------------------------------------------
        # [4] Aux loss backward (accumulates on top of critic encoder grads)
        # ------------------------------------------------------------------
        aux_loss_val = 0.0
        if self._is_visual and self._aux_loss_type != "none" and self._aux_weight > 0.0:
            aux = self._compute_aux_loss(obs, actions)
            if aux is not None:
                (aux * self._aux_weight).backward()
                aux_loss_val = aux.item()

        # ------------------------------------------------------------------
        # [4b] Encoder step — fires every `freq` critic updates
        # ------------------------------------------------------------------
        self._encoder_update_counter += 1
        if self.encoder_optimizer is not None and (self._encoder_update_counter % freq == 0):
            self.encoder_optimizer.step()

        # ------------------------------------------------------------------
        # [5] Manually zero encoder grads before actor backward
        #     → actor loss NEVER flows into encoder weights
        # ------------------------------------------------------------------
        _zero_param_grads(self._encoder_param_list)

        # ------------------------------------------------------------------
        # [6] Actor forward + backward (encoder grad blocked above)
        # ------------------------------------------------------------------
        # compute_critic=False: only the fresh action/log_prob sample is
        # used from this call (the critic pass below, with that action, is
        # what actually needs the critic head).
        result_actor = self.model(obs, compute_critic=False)
        actor_out = result_actor["actor_out"]
        if isinstance(actor_out, dict):
            new_action = actor_out.get("action")
            log_prob = actor_out.get("log_prob")
        elif isinstance(actor_out, tuple):
            new_action, log_prob = actor_out
        else:
            new_action, log_prob = actor_out, torch.zeros(rewards.shape, device=self.device)

        # compute_actor=False: without it, this call would silently redo the
        # exact actor forward pass above (same obs, same still-unstepped
        # actor/encoder weights) just to discard the result -- new_action is
        # already fixed from result_actor and is what's passed in below.
        q_actor = self.model(obs, action=new_action, compute_actor=False)["value"]
        if isinstance(q_actor, tuple):
            q_actor = torch.min(*q_actor)

        actor_loss = sac_policy_loss(log_prob, q_actor, self.alpha.detach())

        # [7] Actor head step, PLUS any actor-only encoder(s)' step.
        #
        # `actor_loss.backward()` populates gradients for every parameter in
        # its graph -- that's `self.model.actor`'s own params (via
        # `result_actor`/`new_action`), any *shared*/critic-reachable
        # encoder (via `q_actor`'s critic forward, since q_actor's encoder
        # is whatever feeds the critic head), AND any actor-only encoder
        # (via `result_actor`'s own forward, e.g. `actor_state_enc` in a
        # `flows: ["actor_state_enc -> actor", "critic_state_enc ->
        # critic"]` config). Only the actor-only subset should actually be
        # consumed by an optimizer.step() here -- the shared/critic-
        # reachable subset's contribution is exactly the contamination [5]
        # exists to keep out (an actor-only encoder has no such shared
        # state to protect, since nothing else ever trains it: see
        # `_partition_encoder_params`'s docstring in __init__).
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.actor_only_encoder_optimizer is not None and (
            self._encoder_update_counter % freq == 0
        ):
            self.actor_only_encoder_optimizer.step()

        # ------------------------------------------------------------------
        # [8] Zero encoder grads again — clean for next update() call
        #     (critic-reachable encoders only -- actor-only encoders' grad
        #     was just legitimately consumed above, not contamination to
        #     discard).
        # ------------------------------------------------------------------
        _zero_param_grads(self._encoder_param_list)

        # ------------------------------------------------------------------
        # [9] Temperature update
        # ------------------------------------------------------------------
        if self.alpha_optimizer is not None:
            temp_loss = sac_temperature_loss(log_prob.detach(), self.log_alpha, self.target_entropy)
            self.alpha_optimizer.zero_grad()
            temp_loss.backward()
            self.alpha_optimizer.step()
        else:
            temp_loss = torch.zeros(1, device=self.device)

        # ------------------------------------------------------------------
        # [10] Soft update target network (unchanged)
        # ------------------------------------------------------------------
        _soft_update(self.model, self.target_model, self.cfg.tau)
        self._global_step += 1

        metrics: dict[str, float] = {
            "sac/critic_loss": critic_loss.item(),
            "sac/actor_loss": actor_loss.item(),
            "sac/alpha": self.alpha.item(),
            "sac/temp_loss": temp_loss.item(),
            "sac/log_prob_mean": log_prob.mean().item(),
            "sac/q_mean": q_actor.mean().item(),
            "sac/target_q_mean": target_q.mean().item(),
            "sac/encoder_update_freq": float(freq),
        }
        if aux_loss_val:
            metrics["sac/aux_loss"] = aux_loss_val
        return metrics

    # ------------------------------------------------------------------
    # Auxiliary loss dispatcher (Phase A.2)
    # ------------------------------------------------------------------

    def _compute_aux_loss(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute unsupervised/self-supervised encoder auxiliary loss.

        Dispatches to the selected technique based on ``aux_loss_type``.
        Returns a scalar loss tensor (un-weighted) or None if inapplicable.
        """
        mode = self._aux_loss_type

        # Collect pixel tensors (CNN encoder inputs: float CHW in [0,1])
        pixel_obs = {k: v for k, v in obs.items() if v.dim() == 4}
        if not pixel_obs:
            return None

        # Use first visual key as the anchor view
        anchor_key = next(iter(pixel_obs))
        anchor = pixel_obs[anchor_key].float()

        if mode == "ae":
            # Autoencoder: encode then decode, MSE vs original pixels
            z = _encode_obs(self.model, obs, anchor_key)
            recon = _decode_latent(self.model, z, anchor_key)
            if recon is None:
                return None
            return reconstruction_loss(recon, anchor)

        elif mode == "vae":
            z_params = _encode_obs_vae(self.model, obs, anchor_key)
            if z_params is None:
                return None
            mu, log_var = z_params
            z = _reparameterize(mu, log_var)
            recon = _decode_latent(self.model, z, anchor_key)
            if recon is None:
                return None
            return vae_loss(recon, anchor, mu, log_var)

        elif mode == "curl":
            # CURL: InfoNCE between two augmented views via momentum encoder
            aug1 = augment(anchor, mode=self._aug_mode)
            aug2 = augment(anchor, mode=self._aug_mode)
            z_anchor = _project_obs(self.model, obs, anchor_key, aug1)
            z_pos = _project_obs_momentum(self.model, obs, anchor_key, aug2)
            if z_anchor is None or z_pos is None:
                return None
            return info_nce_loss(z_anchor, z_pos)

        elif mode == "byol":
            aug1 = augment(anchor, mode=self._aug_mode)
            aug2 = augment(anchor, mode=self._aug_mode)
            z_online = _project_obs(self.model, obs, anchor_key, aug1)
            z_target = _project_obs_momentum(self.model, obs, anchor_key, aug2)
            if z_online is None or z_target is None:
                return None
            # Update momentum encoder EMA
            _update_momentum_encoder(self.model, anchor_key)
            return byol_loss(z_online, z_target)

        elif mode == "drq":
            # DrQ: augmented Q-consistency (critic loss on two augmented views)
            aug1 = {
                k: augment(v.float(), mode="drq") if v.dim() == 4 else v for k, v in obs.items()
            }
            aug2 = {
                k: augment(v.float(), mode="drq") if v.dim() == 4 else v for k, v in obs.items()
            }
            q1_aug = self.model(aug1, action=actions, compute_actor=False)["value"]
            q2_aug = self.model(aug2, action=actions, compute_actor=False)["value"]
            if isinstance(q1_aug, tuple):
                q1_aug = q1_aug[0]
            if isinstance(q2_aug, tuple):
                q2_aug = q2_aug[0]
            return drq_aug_loss(q1_aug, q2_aug)

        elif mode == "spr":
            z_t = _encode_raw(self.model, obs, anchor_key)
            if z_t is None:
                return None
            return spr_loss(z_t, actions, self.model, anchor_key)

        elif mode == "barlow":
            aug1 = augment(anchor, mode=self._aug_mode)
            aug2 = augment(anchor, mode=self._aug_mode)
            z1 = _project_obs(self.model, obs, anchor_key, aug1)
            z2 = _project_obs(self.model, obs, anchor_key, aug2)
            if z1 is None or z2 is None:
                return None
            return barlow_twins_loss(z1, z2)

        return None

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.checkpoint_payload(), path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_checkpoint_payload(ckpt)

    def checkpoint_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model_state": self.model.state_dict(),
            "target_model_state": self.target_model.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
            "replay_buffer_state": self.buffer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "algo_step": self._global_step,
            "encoder_update_counter": self._encoder_update_counter,
        }
        if self.alpha_optimizer is not None:
            payload["alpha_optimizer_state"] = self.alpha_optimizer.state_dict()
        # encoder_optimizer_state: absent in v0.1.x checkpoints — gracefully skipped on load
        if self.encoder_optimizer is not None:
            payload["encoder_optimizer_state"] = self.encoder_optimizer.state_dict()
        # actor_only_encoder_optimizer_state: absent in checkpoints saved
        # before this optimizer existed — gracefully skipped on load, same
        # as encoder_optimizer_state above.
        if self.actor_only_encoder_optimizer is not None:
            payload["actor_only_encoder_optimizer_state"] = (
                self.actor_only_encoder_optimizer.state_dict()
            )
        return payload

    def load_checkpoint_payload(self, payload: dict[str, object]) -> None:
        model_state = payload.get("model_state", payload.get("model"))
        if model_state is not None:
            self.model.load_state_dict(model_state)
        target_state = payload.get("target_model_state", payload.get("target"))
        if target_state is not None:
            self.target_model.load_state_dict(target_state)
        actor_opt = payload.get("actor_optimizer_state")
        if actor_opt is not None:
            self.actor_optimizer.load_state_dict(actor_opt)
        critic_opt = payload.get("critic_optimizer_state")
        if critic_opt is not None:
            self.critic_optimizer.load_state_dict(critic_opt)
        enc_opt = payload.get("encoder_optimizer_state")
        if self.encoder_optimizer is not None and enc_opt is not None:
            self.encoder_optimizer.load_state_dict(enc_opt)
        actor_only_enc_opt = payload.get("actor_only_encoder_optimizer_state")
        if self.actor_only_encoder_optimizer is not None and actor_only_enc_opt is not None:
            self.actor_only_encoder_optimizer.load_state_dict(actor_only_enc_opt)
        alpha_opt = payload.get("alpha_optimizer_state")
        if self.alpha_optimizer is not None and alpha_opt is not None:
            self.alpha_optimizer.load_state_dict(alpha_opt)
        buf = payload.get("replay_buffer_state")
        if buf is not None:
            self.buffer.load_state_dict(buf)
        log_alpha = payload.get("log_alpha")
        if log_alpha is not None:
            self.log_alpha.data.copy_(log_alpha.to(self.device))
        self._global_step = int(payload.get("algo_step", payload.get("step", 0)))
        self._encoder_update_counter = int(payload.get("encoder_update_counter", 0))


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for sp, tp in zip(src.parameters(), tgt.parameters(), strict=True):
        tp.data.mul_(1.0 - tau).add_(sp.data * tau)


def _partition_encoder_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Module]]:
    """Split a model's encoders into (critic-reachable, actor-only).

    Returns ``(critic_reachable_params, actor_only_params,
    actor_only_modules)`` -- the first two are flat, de-duplicated
    ``nn.Parameter`` lists (matching ``_unique_encoder_params``'s
    de-duplication-by-tensor-id), the third is the actual actor-only
    encoder module objects (kept for anything that needs to walk module
    structure rather than a flat parameter list).

    "Critic-reachable" means named in ``encoder_names_for_head("critic")``
    (which already exists on ``AgentModel`` for exactly this "which
    encoders does this head actually need" question, and conservatively
    returns *every* encoder name when a model has no explicit `flows:`
    routing for its critic -- e.g. one implicit encoder shared by both
    heads -- so single-encoder configs are unaffected: nothing ends up
    actor-only for them). Every encoder not in that set feeds only the
    actor, and is actor-only.

    See ``SAC.__init__``'s comment above this function's call site for why
    this split exists: an actor-only encoder has no valid gradient source
    under the old "train encoders only through critic backward" rule
    (critic_loss's graph never touches it; actor_loss's does, but that
    contribution used to always be discarded) -- it would sit frozen at
    random initialization forever otherwise.
    """
    encoders: dict[str, nn.Module] = dict(getattr(model, "encoders", {}) or {})
    if not encoders:
        return [], [], []

    encoder_names_for_head = getattr(model, "encoder_names_for_head", None)
    critic = getattr(model, "critic", None)
    if encoder_names_for_head is None or critic is None:
        # No critic head (or a model type without the reachability query)
        # -- fall back to the old, conservative "every encoder is
        # critic-reachable" behaviour; nothing is actor-only.
        return _unique_encoder_params(model), [], []

    critic_name = getattr(critic, "name", "critic")
    critic_reachable_names = set(encoder_names_for_head(critic_name))

    critic_reachable_params: list[nn.Parameter] = []
    actor_only_params: list[nn.Parameter] = []
    actor_only_modules: list[nn.Module] = []
    seen: set[int] = set()
    for name, enc in encoders.items():
        is_critic_reachable = name in critic_reachable_names
        target = critic_reachable_params if is_critic_reachable else actor_only_params
        if not is_critic_reachable:
            actor_only_modules.append(enc)
        for p in enc.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                target.append(p)
    return critic_reachable_params, actor_only_params, actor_only_modules


def _unique_encoder_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect all encoder parameters, de-duplicated by tensor id."""
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


def _detach_visual_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return obs dict with pixel tensors (4-D) detached from the graph."""
    return {k: v.detach() if v.dim() == 4 else v for k, v in obs.items()}


# ------------------------------------------------------------------
# Tiny model-introspection helpers for aux loss dispatch
# ------------------------------------------------------------------


def _encode_raw(model: nn.Module, obs: dict, key: str) -> torch.Tensor | None:
    """Run encoder for *key* and return the raw latent (no projection)."""
    encoders = getattr(model, "encoders", {})
    for enc_name, enc in encoders.items():
        if key in enc_name or enc_name in key:
            src = obs.get(key)
            if src is None:
                return None
            online = getattr(enc, "online", enc)
            return online(src.float())
    return None


def _encode_obs(model: nn.Module, obs: dict, key: str) -> torch.Tensor | None:
    return _encode_raw(model, obs, key)


def _encode_obs_vae(
    model: nn.Module,
    obs: dict,
    key: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return (mu, log_var) from a VAEHead if present."""
    from srl.networks.heads.aux_head import VAEHead  # local import to avoid cycles

    for module in model.modules():
        if isinstance(module, VAEHead):
            z = _encode_raw(model, obs, key)
            if z is None:
                return None
            return module(z)
    return None


def _decode_latent(model: nn.Module, z: torch.Tensor, key: str) -> torch.Tensor | None:
    """Run ConvDecoderHead if present."""
    from srl.networks.heads.aux_head import ConvDecoderHead

    for module in model.modules():
        if isinstance(module, ConvDecoderHead):
            return module(z)
    return None


def _project_obs(
    model: nn.Module,
    obs: dict,
    key: str,
    aug_pixels: torch.Tensor,
) -> torch.Tensor | None:
    """Encode augmented pixels, then apply ProjectionHead (online)."""
    from srl.networks.heads.aux_head import ProjectionHead

    z = _encode_raw_pixels(model, obs, key, aug_pixels)
    if z is None:
        return None
    for module in model.modules():
        if isinstance(module, ProjectionHead):
            return module(z)
    return z  # fallback: no projection head, return raw latent


def _project_obs_momentum(
    model: nn.Module,
    obs: dict,
    key: str,
    aug_pixels: torch.Tensor,
) -> torch.Tensor | None:
    """Encode augmented pixels through momentum encoder target, project."""
    from srl.networks.encoders.momentum_encoder import MomentumEncoder
    from srl.networks.heads.aux_head import ProjectionHead

    encoders = getattr(model, "encoders", {})
    for enc_name, enc in encoders.items():
        if not isinstance(enc, MomentumEncoder):
            continue
        if key in enc_name or enc_name in key:
            z = enc(aug_pixels.float(), use_target=True)
            for module in model.modules():
                if isinstance(module, ProjectionHead):
                    return module(z)
            return z
    return None


def _encode_raw_pixels(
    model: nn.Module,
    obs: dict,
    key: str,
    pixels: torch.Tensor,
) -> torch.Tensor | None:
    """Run the encoder matched to *key* on externally-supplied *pixels*."""
    encoders = getattr(model, "encoders", {})
    for enc_name, enc in encoders.items():
        if key in enc_name or enc_name in key:
            online = getattr(enc, "online", enc)
            return online(pixels.float())
    return None


def _update_momentum_encoder(model: nn.Module, key: str) -> None:
    from srl.networks.encoders.momentum_encoder import MomentumEncoder

    encoders = getattr(model, "encoders", {})
    for enc_name, enc in encoders.items():
        if isinstance(enc, MomentumEncoder) and (key in enc_name or enc_name in key):
            enc.update_target()


def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    std = (0.5 * log_var).exp()
    eps = torch.randn_like(std)
    return mu + eps * std
