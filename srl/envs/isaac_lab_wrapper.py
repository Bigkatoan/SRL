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

    Parameters
    ----------
    env:
        A ``ManagerBasedRLEnv`` or ``DirectRLEnv`` instance from Isaac Lab.
    obs_key:
        Fallback key used when the env returns a bare tensor (not a dict).
    """

    def __init__(self, env: Any, obs_key: str = "state") -> None:
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

        return (
            self._wrap_obs(obs),
            _to_np(reward),
            _to_np(terminated).astype(bool),
            _to_np(truncated).astype(bool),
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
