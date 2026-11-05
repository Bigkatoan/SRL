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
* weight_norm_projection (SACConfig, opt-in): FlashSAC-style (arXiv:2604.04539
  section 4.2) post-optimizer.step() projection of Linear weight rows to unit
  L2 norm and norm-layer affine vectors to L2 norm sqrt(d). See
  `_project_weights_unit_norm`.
* BatchNorm (`norm: batch_norm` in an encoder/actor/critic's layer config) is
  now safe to use with SAC: target_model is permanently eval()-mode with its
  running stats hard-copied from the online net each update()
  (`_sync_target_buffers`), `self.model`'s train/eval mode is set atomically
  with its forward call under `_model_lock` (safe against
  AsyncOffPolicyRunner's concurrent predict()/update() threads), and the
  critic's Bellman-residual forward pass uses FlashSAC's "cross-batch value
  prediction" (`_critic_forward`) so the predicted and target halves share
  BatchNorm statistics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from srl.algorithms._common import (
    _sync_target_buffers,  # noqa: F401 -- re-exported for backwards compat
    _unique_encoder_params,  # noqa: F401 -- re-exported for backwards compat
    _zero_param_grads,
    is_weighted_mse,
    maybe_update_priorities,
)
from srl.core.base_agent import OffPolicyAgent
from srl.core.config import SACConfig, VisualSACConfig
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


class SAC(OffPolicyAgent):
    """Soft Actor-Critic with automatic entropy tuning.

    Target network, model-mode lock, head/encoder optimizers (critic-reachable
    vs actor-only encoders), replay buffer and obs normalizer are built by
    :class:`~srl.core.base_agent.OffPolicyAgent`.

    Parameters
    ----------
    model:
        :class:`~srl.networks.agent_model.AgentModel` with a SquashedGaussian
        actor and TwinQ critic.
    target_model:
        Optional target network (same architecture as *model*), soft-updated
        with ``tau``; deep-copied from *model* when omitted.
        ``SAC(model, config)`` also works.
    config:
        :class:`~srl.core.config.SACConfig` or
        :class:`~srl.core.config.VisualSACConfig`.
    device:
        PyTorch device.
    env:
        Optional training env (raw Gymnasium envs are wrapped).
    """

    algo_name = "sac"
    config_cls = SACConfig

    def __init__(
        self,
        model: nn.Module,
        target_model: nn.Module | None = None,
        config: SACConfig | None = None,
        device: str | torch.device = "cpu",
        *,
        env=None,
    ) -> None:
        # With BatchNorm, `_critic_forward` switches to FlashSAC's
        # "cross-batch value prediction" path and every `self.model(...)`
        # call goes through the base class's `_forward_model` lock (see
        # OffPolicyAgent): train-mode BatchNorm both branches on `.training`
        # AND writes running_mean/running_var in place, so an unguarded race
        # against AsyncOffPolicyRunner's concurrent act() would be a real
        # concurrent read/write hazard.
        super().__init__(model, target_model, config, device, env=env)
        self._weight_norm_projection: bool = bool(
            getattr(self.cfg, "weight_norm_projection", False)
        )

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

    def _build_exploration(self) -> None:
        # "auto" -> -action_dim (recomputed if the env binding changes it).
        action_dim = self.cfg.action_dim
        if self.cfg.target_entropy == "auto":
            self.target_entropy = -float(action_dim) if action_dim else -1.0
        else:
            self.target_entropy = float(self.cfg.target_entropy)

    def _build_buffer(self):
        if getattr(self.cfg, "use_her", False):
            if getattr(self.cfg, "use_per", False):
                raise ValueError("SACConfig: use_per and use_her cannot be combined.")
            if int(getattr(self.cfg, "replay_n_step", 1)) > 1:
                raise ValueError("SACConfig: replay_n_step > 1 is not supported with use_her.")
            return self._build_her_buffer()
        return super()._build_buffer()

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

    def _critic_forward(
        self,
        obs_for_critic: dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        next_action: torch.Tensor,
    ):
        """Compute the online critic's Q(s, a) and the target critic's
        Q(s', a'), returned in whatever form the configured critic head
        produces (a ``(q1, q2)`` tuple for TwinQHead, a single tensor for
        QFunctionHead) -- exactly what the two direct
        ``self.model(...)``/``self.target_model(...)`` calls this method
        replaces would have returned.

        Without BatchNorm this is just those two direct calls (unchanged
        from before this method existed -- same cost, same numerics).

        With BatchNorm ("cross-batch value prediction", FlashSAC section
        4.2 https://arxiv.org/pdf/2604.04539): BatchNorm's train-mode
        forward normalizes using the CURRENT CALL's batch statistics, so
        two separate B-sized calls -- one on (s, a), one on (s', a') --
        hand the predicted half and the target half of the same Bellman
        residual two *different* sets of normalization statistics, purely
        because of how the call happened to be split, not because the
        underlying transitions are actually that different. We concatenate
        (s, a) and (s', a') into one 2B batch and run each network exactly
        once, so the predicted half and the target half are each
        normalized against statistics drawn from the union of current +
        next transitions. Online and target still have different weights
        (that's the point -- target lags via Polyak averaging /
        `_sync_target_buffers`), so their Q-values still differ; only the
        confound of "which half of the batch defined this call's batch
        statistics" is removed.
        """
        if not self._has_batchnorm:
            result = self._forward_model(
                "train", obs_for_critic, action=actions, compute_actor=False
            )
            q_out = result["value"]
            with torch.no_grad():
                next_q = self.target_model(next_obs, action=next_action, compute_actor=False)[
                    "value"
                ]
            return q_out, next_q

        batch_size = actions.shape[0]
        concat_obs = {k: torch.cat([obs_for_critic[k], next_obs[k]], dim=0) for k in obs_for_critic}
        concat_action = torch.cat([actions, next_action], dim=0)

        online_out = self._forward_model(
            "train", concat_obs, action=concat_action, compute_actor=False
        )["value"]
        with torch.no_grad():
            target_out = self.target_model(concat_obs, action=concat_action, compute_actor=False)[
                "value"
            ]

        def _first_half(v):
            return tuple(t[:batch_size] for t in v) if isinstance(v, tuple) else v[:batch_size]

        def _second_half(v):
            return tuple(t[batch_size:] for t in v) if isinstance(v, tuple) else v[batch_size:]

        return _first_half(online_out), _second_half(target_out)

    def _select_action(self, actor_out, deterministic: bool):
        if isinstance(actor_out, dict):
            if deterministic:
                # The deterministic SAC policy is the *squashed* mode,
                # tanh(mean). actor_out["mean"] is the pre-tanh Gaussian
                # mean -- returning it put most eval actions outside
                # [-1, 1] (then clipped by the env), i.e. evaluated a
                # different policy than the one being trained.
                dist = actor_out.get("dist")
                mode = getattr(dist, "mode", None)
                if callable(mode):
                    action = mode()
                elif actor_out.get("mean") is not None:
                    action = torch.tanh(actor_out["mean"])
                else:
                    action = actor_out.get("action")
            else:
                action = actor_out.get("action")
            return action, actor_out.get("log_prob")
        if isinstance(actor_out, tuple):
            return actor_out[0], actor_out[1]
        return actor_out, None

    def update(self) -> dict[str, float]:
        # HERReplayBuffer is episode-indexed (its __len__ counts episodes);
        # `_can_update` uses its can_sample() instead of the length check.
        if not self._can_update():
            return {}

        # Mode is never set once-and-for-all here: every `self.model(...)`
        # call goes through `_forward_model`, which sets train/eval mode
        # atomically with the forward call itself under `_model_lock`.
        batch, obs, next_obs, actions, rewards, dones, discount, is_weights = self._sample_batch()

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
            next_result = self._forward_model("train", next_obs, compute_critic=False)
            next_actor_out = next_result["actor_out"]
            if isinstance(next_actor_out, dict):
                next_action = next_actor_out.get("action")
                next_log_prob = next_actor_out.get("log_prob")
            elif isinstance(next_actor_out, tuple):
                next_action, next_log_prob = next_actor_out
            else:
                next_action, next_log_prob = (
                    next_actor_out,
                    torch.zeros(rewards.shape, device=self.device),
                )

        # When encoder should NOT receive gradients from critic, detach obs
        # pixels/features before the critic forward pass.
        if not self._enc_with_critic and self._encoder_param_list:
            obs_for_critic = _detach_visual_obs(obs)
        else:
            obs_for_critic = obs

        # Predicted Q(s, a) (online, gradient-tracked) and target Q(s', a')
        # (target net, no_grad) -- see `_critic_forward` for why this isn't
        # simply the two direct calls it replaces once the critic has
        # BatchNorm layers.
        q_out, next_q = self._critic_forward(obs_for_critic, actions, next_obs, next_action)

        with torch.no_grad():
            if isinstance(next_q, tuple):
                next_q = torch.min(*next_q)
            # `discount` is gamma**k per transition (k = steps the stored
            # n-step reward spans), or plain gamma for one-step buffers.
            target_q = rewards + discount * (1.0 - dones) * (
                next_q - self.alpha.detach() * next_log_prob
            )

        if isinstance(q_out, tuple):
            q1, q2 = q_out
            if is_weights is None:
                critic_loss = sac_q_loss(q1, q2, target_q.detach())
            else:
                critic_loss = is_weighted_mse(q1, target_q, is_weights) + is_weighted_mse(
                    q2, target_q, is_weights
                )
            td_error = 0.5 * ((q1 - target_q).abs() + (q2 - target_q).abs())
        else:
            critic_loss = is_weighted_mse(q_out, target_q.detach(), is_weights)
            td_error = (q_out - target_q).abs()
        maybe_update_priorities(self.buffer, batch, td_error)

        # [2b] Backward through critic (+ encoder if enc_with_critic)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # ------------------------------------------------------------------
        # [3] Critic head step (encoder step deferred to [4])
        # ------------------------------------------------------------------
        self.critic_optimizer.step()
        if self._weight_norm_projection:
            _project_weights_unit_norm(self.model.critic)

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
            if self._weight_norm_projection:
                _project_weights_unit_norm(self.model.encoders)

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
        result_actor = self._forward_model("train", obs, compute_critic=False)
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
        q_actor = self._forward_model("train", obs, action=new_action, compute_actor=False)["value"]
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
        if self._weight_norm_projection:
            _project_weights_unit_norm(self.model.actor)
        if self.actor_only_encoder_optimizer is not None and (
            self._encoder_update_counter % freq == 0
        ):
            self.actor_only_encoder_optimizer.step()
            if self._weight_norm_projection:
                for enc in self._actor_only_encoder_modules:
                    _project_weights_unit_norm(enc)

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
            # Floor alpha at cfg.min_alpha (permissive 1e-8 default -- see
            # SACConfig.min_alpha's docstring for why this exists: log_alpha
            # is otherwise unclamped, and a real, observed failure mode on a
            # long run is the temperature-loss gradient pushing it down
            # without bound once policy entropy stays above target_entropy
            # for long enough, with nothing to arrest the resulting spiral.
            with torch.no_grad():
                self.log_alpha.clamp_(min=math.log(max(self.cfg.min_alpha, 1e-300)))
        else:
            temp_loss = torch.zeros(1, device=self.device)

        # ------------------------------------------------------------------
        # [10] Soft update target network (unchanged)
        # ------------------------------------------------------------------
        # Polyak-average parameters; hard-copy BatchNorm running stats (see
        # OffPolicyAgent._soft_update_target).
        self._soft_update_target()
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
            q1_aug = self._forward_model("train", aug1, action=actions, compute_actor=False)[
                "value"
            ]
            q2_aug = self._forward_model("train", aug2, action=actions, compute_actor=False)[
                "value"
            ]
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

    def _algo_payload(self) -> dict[str, object]:
        payload = self._base_payload()
        payload["log_alpha"] = self.log_alpha.detach().cpu()
        if self.alpha_optimizer is not None:
            payload["alpha_optimizer_state"] = self.alpha_optimizer.state_dict()
        return payload

    def _load_algo_payload(self, payload: dict[str, object]) -> None:
        self._load_base_payload(payload)
        alpha_opt = payload.get("alpha_optimizer_state")
        if self.alpha_optimizer is not None and alpha_opt is not None:
            self.alpha_optimizer.load_state_dict(alpha_opt)
        log_alpha = payload.get("log_alpha")
        if log_alpha is not None:
            self.log_alpha.data.copy_(log_alpha.to(self.device))


# ──────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────────────────────────


_NORM_AFFINE_TYPES = (
    nn.LayerNorm,
    nn.modules.batchnorm._BatchNorm,
    nn.GroupNorm,
    nn.modules.instancenorm._InstanceNorm,
)


def _project_weights_unit_norm(module: nn.Module, eps: float = 1e-8) -> None:
    """FlashSAC weight normalization (arXiv:2604.04539 section 4.2).

    "After each gradient step, we project each weight vector onto the
    unit-norm sphere and each normalization parameter vector (gamma, beta)
    to norm sqrt(d)." Concretely, in-place and with no extra optimizer
    state:

    * Every ``nn.Linear.weight`` ROW (each output unit's incoming weight
      vector, i.e. dim=1 of the ``(out_features, in_features)`` matrix) is
      rescaled to unit L2 norm.
    * Every normalization layer's affine vector(s) -- LayerNorm/BatchNorm/
      GroupNorm/InstanceNorm's weight (gamma) and bias (beta) if present,
      RMSNorm's weight -- is rescaled to L2 norm ``sqrt(d)`` where ``d`` is
      that vector's length (so its RMS value is ~1 on average).

    Bounds uncontrolled weight growth that would otherwise inflate Q-value
    variance and amplify estimation error under bootstrapping. Meant to be
    called right after an optimizer's ``.step()`` -- i.e. once per actual
    gradient step applied to the module in question, not once per
    ``update()`` call (the encoder optimizer, for instance, only steps
    every ``encoder_update_freq`` critic updates).

    ``nn.Linear.bias`` is deliberately left untouched -- the paper's text
    only names "weight vectors" (Linear rows) and normalization-layer
    affine vectors, not biases.
    """
    from srl.networks.layers.norms import _RMSNorm

    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Linear):
                w = m.weight.data
                row_norm = w.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
                w.div_(row_norm)
            elif isinstance(m, _NORM_AFFINE_TYPES) or isinstance(m, _RMSNorm):
                for attr in ("weight", "bias"):
                    vec = getattr(m, attr, None)
                    if vec is None or vec.dim() != 1:
                        continue
                    d = vec.numel()
                    vec_norm = vec.data.norm(p=2).item()
                    # A vector can only be legitimately at (or extremely
                    # near) exactly zero norm if it has never actually
                    # received a gradient update -- e.g. BatchNorm's bias
                    # (beta) starts at all-zeros by default, and stays
                    # there for as long as nothing trains it. Scaling by
                    # sqrt(d)/vec_norm in that regime is a mathematically
                    # meaningless "direction" (there isn't one) and, worse,
                    # silently masks the underlying problem: a genuinely
                    # zero vector times any finite scale is still zero (no
                    # crash, nothing to notice), while a merely-tiny
                    # (denormal-ish) one could overflow to inf under
                    # `sqrt(d)/vec_norm` for a small `eps`. Skip the rescale
                    # entirely below this threshold rather than either --
                    # this is what surfaced the actor-only-encoder-never-
                    # trains bug in the first place (its BatchNorm bias
                    # was suspiciously stuck at an exact 0.0 vector across
                    # an entire run, instead of silently "succeeding" at
                    # rescaling zero to zero).
                    if vec_norm < eps:
                        continue
                    vec.data.mul_(math.sqrt(d) / vec_norm)


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
