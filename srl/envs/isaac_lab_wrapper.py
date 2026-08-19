"""Isaac Lab environment wrapper for SRL."""

from __future__ import annotations

from typing import Any

import numpy as np

from srl.envs._image_obs import maybe_hwc_to_chw


class IsaacLabWrapper:
    """Wrap an Isaac Lab ``ManagerBasedRLEnv`` to match the SRL interface.

    Isaac Lab envs return torch tensors on GPU; this wrapper converts them
    to numpy arrays (CPU) for the SRL buffers.

    Observation dict routing
    ------------------------
    Isaac Lab groups observations by *obs group* names (defined in the env
    config).  Common group names are ``"policy"`` and ``"critic"``.
    The wrapper preserves these group names as dict keys:

    * **Image-only env** (e.g. Cartpole-RGB)::

        obs = {"policy": <(N, 3, H, W) float32>}   # after HWC→CHW

    * **Multi-modal env** (image + privileged state)::

        obs = {
            "policy":  <(N, 3, H, W) float32>,   # camera image
            "critic":  <(N, D) float32>,           # state vector
        }

    To wire them into the model, **name the YAML encoders to match the obs
    group keys** (``policy`` and ``critic`` in the example above).  The
    ``_remap_obs_to_encoders`` function in ``srl/cli/train.py`` will then
    route each obs tensor to the correct encoder.

    ``supports_partial_reset`` / ``info["true_final_obs"]``
    -----------------------------------------------------------
    Isaac Lab/mjlab's own ``env.step()`` defaults to auto-resetting any
    sub-env that just terminated or timed out INSIDE the call, handing back
    the reset (NEXT episode's first) observation with no way to recover the
    true terminal one. That's fine for the "what do I act on next" question
    (the reset observation is exactly right for that), but wrong for value
    bootstrapping: an off-policy Q-target or an on-policy GAE bootstrap at a
    time-limit truncation needs the TRUE final state of the ENDING episode,
    not an unrelated fresh episode's first frame. Confirmed via a real
    training run investigation (see ``srl.cli.train``'s callers of this
    wrapper) that this was silently corrupting exactly that.

    ``_make_cli_env`` sets this flag (and, only when it's True, disables
    the underlying env's own auto-reset via its cfg) after checking whether
    the concrete env CLASS's ``reset()`` accepts an ``env_ids`` kwarg --
    mjlab's ``ManagerBasedRlEnv`` does (verified directly against its
    source); real Isaac Lab is assumed to as well (mjlab's own docs
    describe itself as mirroring Isaac Lab's API "closely enough that
    IsaacLabWrapper works unchanged"), but this repo has no real Isaac Lab
    install to verify that claim against, so the check is a real capability
    probe, not an assumption baked into this class. When False (an env
    whose `reset()` doesn't take `env_ids`), this class behaves EXACTLY as
    it did before this feature existed -- no behavior change, no risk to an
    env type this can't be verified against.

    When True: `step()` disables nothing itself (the caller already set
    `auto_reset=False` on the env's own cfg before constructing it) --
    instead, after every `env.step()`, this class immediately resets
    whichever sub-envs just ended (`env.reset(env_ids=...)`, required by
    mjlab's own `auto_reset=False` contract: it raises if `step()` is
    called again with a pending reset outstanding) and splices the result
    into the observation used for the NEXT action, while additionally
    exposing the TRUE pre-reset observation via `info["true_final_obs"]`
    for any caller that wants correct bootstrapping. `info["true_final_obs"]`
    is always present in the returned `info` (even when nothing ended, in
    which case it's simply identical to the returned observation, and even
    when `supports_partial_reset` is False, in which case it's whatever the
    env already returned -- so callers can uniformly read
    `info.get("true_final_obs", next_obs)` regardless of env type).

    Parameters
    ----------
    env:
        A ``ManagerBasedRLEnv`` or ``DirectRLEnv`` instance from Isaac Lab.
    obs_key:
        Fallback key used when the env returns a bare tensor (not a dict).
    supports_partial_reset:
        Whether ``env.reset(env_ids=...)`` (a partial, per-sub-env reset) is
        available on this env instance -- see the class docstring above.
        Must match whatever the caller actually configured the env's own
        ``auto_reset`` cfg with; this class does not set that cfg itself
        (it doesn't own env construction).
    """

    def __init__(
        self, env: Any, obs_key: str = "state", supports_partial_reset: bool = False
    ) -> None:
        self.env = env
        self.obs_key = obs_key
        self.num_envs: int = getattr(env, "num_envs", 1)
        self.obs_space = getattr(env, "observation_space", None)
        # `action_space` on Isaac Lab/mjlab envs is the BATCHED space (shape
        # (num_envs, action_dim), matching gymnasium's VectorEnv convention),
        # not one env's space -- callers that need to sample a single action
        # per env (e.g. off-policy random-action warmup) want
        # `single_act_space` instead. Falls back to `act_space` itself for
        # envs that don't distinguish the two (already a single-env space).
        self.act_space = getattr(env, "action_space", None)
        self.single_act_space = getattr(env, "single_action_space", None) or self.act_space
        self.supports_partial_reset = supports_partial_reset

    @property
    def device(self):
        if hasattr(self.env, "device"):
            return self.env.device
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "device"):
            return self.env.unwrapped.device
        raise AttributeError("Isaac Lab environment does not expose a device attribute")

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        # Isaac Lab returns (obs_dict, extras) or obs_dict
        if isinstance(out, tuple):
            obs, info = out[0], out[1]
        else:
            obs, info = out, {}
        return self._wrap_obs(obs), info

    def step(self, actions):
        import torch  # local import — optional dep

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)

        out = self.env.step(actions)
        # Returns (obs, reward, terminated, truncated, info) in newer Isaac Lab
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = done, done

        terminated_np = _to_np(terminated).astype(bool)
        truncated_np = _to_np(truncated).astype(bool)
        # With `supports_partial_reset`, `auto_reset=False` on the env's own
        # cfg means `obs` here is the TRUE pre-reset observation for any
        # sub-env that just ended -- exactly what a correct bootstrap needs.
        # Without it (old/unverified env types), `obs` is whatever the env
        # already gives back (its own auto-reset semantics, unchanged from
        # before this feature existed) -- still assigned to
        # `info["true_final_obs"]` below so callers have one uniform place
        # to look, even though for that case it isn't actually "true".
        true_final_obs_wrapped = self._wrap_obs(obs)
        next_obs_wrapped = true_final_obs_wrapped

        done_mask = terminated_np | truncated_np
        if self.supports_partial_reset and done_mask.any():
            done_idx = np.nonzero(done_mask)[0]
            done_idx_t = torch.as_tensor(done_idx, device=self.device, dtype=torch.long)
            # Required by mjlab's own `auto_reset=False` contract: it raises
            # on the NEXT `step()` if any sub-env has a reset outstanding --
            # so this must happen unconditionally, every time any sub-env
            # just ended, not on some later/lazier schedule.
            reset_out = self.env.reset(env_ids=done_idx_t)
            reset_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            reset_obs_wrapped = self._wrap_obs(reset_obs)
            next_obs_wrapped = {k: v.copy() for k, v in true_final_obs_wrapped.items()}
            for k in next_obs_wrapped:
                next_obs_wrapped[k][done_idx] = reset_obs_wrapped[k][done_idx]

        info = dict(info) if isinstance(info, dict) else {}
        info["true_final_obs"] = true_final_obs_wrapped

        return (
            next_obs_wrapped,
            _to_np(reward),
            terminated_np,
            truncated_np,
            info,
        )

    def close(self) -> None:
        self.env.close()

    def _wrap_obs(self, obs: Any) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            return {k: maybe_hwc_to_chw(_to_np(v)) for k, v in obs.items()}
        return {self.obs_key: maybe_hwc_to_chw(_to_np(obs))}


def _to_np(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)
