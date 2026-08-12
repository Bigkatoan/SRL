"""Experience collector: rolls out a policy in a (vectorised) environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from srl.core.rollout_buffer import RolloutBuffer


class Collector:
    """Rolls out *agent* in *env* and fills a buffer.

    Parameters
    ----------
    agent:
        An :class:`~srl.core.base_agent.BaseAgent` instance.
    env:
        Any (potentially vectorised) environment with the SRL interface.
    buffer:
        A :class:`~srl.core.rollout_buffer.RolloutBuffer` (on-policy: no
        ``next_obs``, tracks ``value``/``log_prob`` for GAE) or a
        :class:`~srl.core.replay_buffer.ReplayBuffer` /
        :class:`~srl.core.gpu_replay_buffer.GPUReplayBuffer` (off-policy:
        needs ``next_obs`` per transition, no ``value``/``log_prob``
        concept). The two have genuinely different ``add()`` signatures --
        this dispatches on the buffer's actual type rather than assuming one
        shape fits both, which is what used to make this class crash
        (``TypeError``) the moment it was paired with anything but a
        ``RolloutBuffer``, despite claiming to support both.
    device:
        PyTorch device.
    """

    def __init__(self, agent, env, buffer, device: str | torch.device = "cpu") -> None:
        self.agent = agent
        self.env = env
        self.buffer = buffer
        self.device = torch.device(device)
        self._obs: dict[str, np.ndarray] | None = None
        self._hidden: dict[str, Any] = {}

    def reset(self) -> None:
        obs, _ = self.env.reset()
        self._obs = obs

    def collect(self, n_steps: int) -> None:
        """Collect exactly *n_steps* transitions."""
        if self._obs is None:
            self.reset()

        is_rollout = isinstance(self.buffer, RolloutBuffer)

        for _ in range(n_steps):
            obs_t = _np_to_torch(self._obs, self.device)
            with torch.no_grad():
                action, log_prob, value, new_hidden = self.agent.predict(obs_t, self._hidden)
            self._hidden = new_hidden or {}

            action_np = action.cpu().numpy()
            next_obs, reward, done, truncated, info = self.env.step(action_np)

            # `done` must be true termination only, never OR'd with
            # `truncated` -- off-policy bootstrap targets need the
            # distinction (see replay_buffer.py's ReplayBatch docstring).
            if is_rollout:
                self.buffer.add(
                    obs=self._obs,
                    action=action_np,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    log_prob=log_prob.cpu().numpy() if log_prob is not None else None,
                    value=value.cpu().numpy() if value is not None else None,
                )
            else:
                self.buffer.add(
                    obs=self._obs,
                    action=action_np,
                    reward=reward,
                    next_obs=next_obs,
                    done=done,
                    truncated=truncated,
                )
            self._obs = next_obs


def _np_to_torch(obs_dict: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.asarray(v)).float().to(device) for k, v in obs_dict.items()}
