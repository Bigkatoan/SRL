"""Shared helpers for the off-policy actor-critic algorithms (SAC/TD3/DDPG).

Everything here used to be copy-pasted between ``sac.py``, ``td3.py`` and
``ddpg.py`` -- and the copies had drifted (only SAC had the actor-only
encoder fix, only SAC had the model-mode lock). Keeping one implementation
means a fix lands in every algorithm at once.
"""

from __future__ import annotations

import threading

import numpy as np
import torch
import torch.nn as nn


def _soft_update(src: nn.Module, tgt: nn.Module, tau: float) -> None:
    for sp, tp in zip(src.parameters(), tgt.parameters(), strict=True):
        tp.data.mul_(1.0 - tau).add_(sp.data * tau)


def _sync_target_buffers(src: nn.Module, tgt: nn.Module) -> None:
    """Hard-copy every buffer (BatchNorm's running_mean/running_var/
    num_batches_tracked, chiefly) from *src* to *tgt*.

    ``_soft_update`` above only walks ``.parameters()`` -- buffers are
    deliberately excluded from a module's parameter list (they're not
    learned by gradient descent), so BatchNorm's running statistics need
    their own, separate propagation path from the online net to the target
    net. A straight copy, not a further Polyak average: BatchNorm's own
    ``momentum`` already performs an EMA update of these buffers on every
    train-mode forward pass, so re-EMA'ing an already-EMA'd quantity here
    would just add an extra, redundant lag on top of the one BatchNorm
    itself already applies.
    """
    src_buffers = dict(src.named_buffers())
    tgt_buffers = dict(tgt.named_buffers())
    for name, tb in tgt_buffers.items():
        sb = src_buffers.get(name)
        if sb is None or sb.shape != tb.shape:
            continue
        tb.data.copy_(sb.data)


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


def _partition_encoder_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Module]]:
    """Split a model's encoders into (critic-reachable, actor-only).

    Returns ``(critic_reachable_params, actor_only_params,
    actor_only_modules)`` -- the first two are flat, de-duplicated
    ``nn.Parameter`` lists (matching ``_unique_encoder_params``'s
    de-duplication-by-tensor-id), the third is the actual actor-only
    encoder module objects (needed by weight-norm projection, which walks
    module structure to tell a Linear row apart from a norm layer's affine
    vector -- a flat parameter list alone can't distinguish them).

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


def _zero_param_grads(params: list[nn.Parameter]) -> None:
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


# ──────────────────────────────────────────────────────────────────────────────
# Encoder optimizers
# ──────────────────────────────────────────────────────────────────────────────


def build_encoder_optimizers(
    model: nn.Module, lr: float
) -> tuple[
    list[nn.Parameter],
    list[nn.Parameter],
    list[nn.Module],
    torch.optim.Optimizer | None,
    torch.optim.Optimizer | None,
]:
    """Partition encoders (see ``_partition_encoder_params``) and build one
    Adam optimizer per non-empty group.

    Returns ``(critic_reachable_params, actor_only_params, actor_only_modules,
    encoder_optimizer, actor_only_encoder_optimizer)``.
    """
    critic_reachable, actor_only, actor_only_modules = _partition_encoder_params(model)
    enc_opt = torch.optim.Adam(critic_reachable, lr=lr) if critic_reachable else None
    actor_enc_opt = torch.optim.Adam(actor_only, lr=lr) if actor_only else None
    return critic_reachable, actor_only, actor_only_modules, enc_opt, actor_enc_opt


def forward_with_mode(model: nn.Module, lock: threading.Lock, mode: str, *args, **kwargs):
    """Run ``model(...)`` with train/eval mode set atomically under *lock*.

    ``predict()`` (collector thread) and ``update()`` (trainer thread) share
    one model under ``AsyncOffPolicyRunner``; toggling ``.train()`` /
    ``.eval()`` as a separate statement from the forward call races. It also
    guarantees each forward runs in the mode its caller intends -- TD3 used
    to call ``model.eval()`` in ``predict()`` and never switch back, so every
    update after the first action selection silently ran in eval mode.
    """
    with lock:
        model.train(mode == "train")
        return model(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Observation normalization
# ──────────────────────────────────────────────────────────────────────────────


class ObsNormalizer:
    """Running mean/std normalizer for flat vector observations.

    Keyed by encoder name (callers pass the obs dict *after*
    ``AgentModel._remap_obs_dict`` so keys are stable between the buffer's
    raw keys and the CLI's pre-remapped ones). Only floating-point ``(B, D)``
    tensors are normalized -- images and other higher-rank inputs pass
    through untouched. Output is ``clip((x - mean) / std, -clip, clip)`` as
    in OpenAI baselines' HER normalizer.
    """

    def __init__(self, clip: float = 5.0, eps: float = 1e-2) -> None:
        self.clip = float(clip)
        # baselines uses eps=1e-2 on std; prevents blow-up for (nearly)
        # constant features such as a fixed desired goal.
        self.eps = float(eps)
        self._count: dict[str, float] = {}
        self._mean: dict[str, torch.Tensor] = {}
        self._m2: dict[str, torch.Tensor] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _eligible(v) -> bool:
        return isinstance(v, torch.Tensor) and v.is_floating_point() and v.dim() == 2

    def update(self, obs: dict[str, torch.Tensor]) -> None:
        with self._lock:
            for k, v in obs.items():
                if not self._eligible(v) or v.shape[0] == 0:
                    continue
                x = v.detach().to(torch.float64)
                b = float(x.shape[0])
                b_mean = x.mean(dim=0)
                b_m2 = ((x - b_mean) ** 2).sum(dim=0)
                if k not in self._mean:
                    self._count[k] = b
                    self._mean[k] = b_mean
                    self._m2[k] = b_m2
                    continue
                n = self._count[k]
                delta = b_mean - self._mean[k]
                tot = n + b
                self._mean[k] = self._mean[k] + delta * (b / tot)
                self._m2[k] = self._m2[k] + b_m2 + delta**2 * (n * b / tot)
                self._count[k] = tot

    def normalize(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with self._lock:
            out = {}
            for k, v in obs.items():
                if k not in self._mean or not self._eligible(v):
                    out[k] = v
                    continue
                mean = self._mean[k].to(device=v.device, dtype=v.dtype)
                var = (self._m2[k] / max(self._count[k], 1.0)).to(device=v.device, dtype=v.dtype)
                std = torch.sqrt(var).clamp_min(self.eps)
                out[k] = ((v - mean) / std).clamp(-self.clip, self.clip)
            return out

    def state_dict(self) -> dict:
        with self._lock:
            return {
                "clip": self.clip,
                "eps": self.eps,
                "count": dict(self._count),
                "mean": {k: v.cpu() for k, v in self._mean.items()},
                "m2": {k: v.cpu() for k, v in self._m2.items()},
            }

    def load_state_dict(self, state: dict) -> None:
        with self._lock:
            self.clip = float(state.get("clip", self.clip))
            self.eps = float(state.get("eps", self.eps))
            self._count = {k: float(v) for k, v in state.get("count", {}).items()}
            self._mean = {k: v.to(torch.float64) for k, v in state.get("mean", {}).items()}
            self._m2 = {k: v.to(torch.float64) for k, v in state.get("m2", {}).items()}


def maybe_build_obs_normalizer(cfg) -> ObsNormalizer | None:
    if not bool(getattr(cfg, "normalize_obs", False)):
        return None
    return ObsNormalizer(clip=float(getattr(cfg, "obs_norm_clip", 5.0)))


def prepare_obs(
    model: nn.Module,
    normalizer: ObsNormalizer | None,
    obs: dict[str, torch.Tensor],
    *,
    update_stats: bool = False,
) -> dict[str, torch.Tensor]:
    """Remap *obs* onto encoder names and normalize it (no-op without a
    normalizer, so the default path is byte-for-byte unchanged)."""
    if normalizer is None:
        return obs
    remap = getattr(model, "_remap_obs_dict", None)
    if remap is not None:
        obs = remap(obs)
    if update_stats:
        normalizer.update(obs)
    return normalizer.normalize(obs)


# ──────────────────────────────────────────────────────────────────────────────
# Replay helpers
# ──────────────────────────────────────────────────────────────────────────────


def batch_discounts(batch, gamma: float, device) -> torch.Tensor | float:
    """Per-transition bootstrap discount: ``gamma**k`` for a k-step return.

    Buffers that build n-step returns store the discount for each
    transition's *actual* horizon (a window cut short by an episode end is
    shorter than n). Buffers without n-step support return ``None`` and the
    plain one-step ``gamma`` applies.
    """
    d = getattr(batch, "discounts", None)
    if d is None:
        return gamma
    return d.to(device=device, dtype=torch.float32)


def is_weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights) -> torch.Tensor:
    """MSE, optionally weighted per sample by PER importance-sampling weights."""
    if weights is None:
        return torch.nn.functional.mse_loss(pred, target)
    return (weights * (pred - target) ** 2).mean()


def maybe_update_priorities(buffer, batch, td_error: torch.Tensor) -> None:
    update = getattr(buffer, "update_priorities", None)
    if update is None or getattr(batch, "indices", None) is None:
        return
    update(np.asarray(batch.indices), td_error.detach().abs().cpu().numpy())
