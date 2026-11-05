"""Shared helpers for the on-policy Gaussian agents (PPO / A2C / A3C).

Action bounds
-------------
A ``gaussian`` actor head samples from an *unbounded* Normal. Following
SB3 / CleanRL (``ClipAction``), the sampled action is clipped to the action
bounds only at the env boundary -- the UNCLIPPED sample is what goes into the
rollout buffer, so the PPO ratio / A2C log-prob is evaluated on the exact
action the distribution produced. Deterministic actions (evaluation) are the
distribution's mode, clipped to the same bounds.

The bounds are whatever range the *model's* action space is: the training
loop sets them from the env's (single) action space via
:meth:`OnPolicyActionMixin.set_action_bounds`.
TODO(envlayer): once env-side action rescaling lands (model acts in
[-1, 1], wrapper maps to the env's Box), the env wrapper should expose the
normalized Box as its action space so these bounds become [-1, 1]
automatically.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def box_bounds(space: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Return finite ``(low, high)`` of a Box-like space, else None."""
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return None
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    if low.shape != high.shape or not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        return None
    return low, high


def env_action_bounds(env: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-env (unbatched) action bounds of an SRL env wrapper, if known."""
    space = (
        getattr(env, "single_act_space", None)
        or getattr(env, "act_space", None)
        or getattr(env, "action_space", None)
    )
    return box_bounds(space)


def select_action(actor_out: Any, deterministic: bool) -> tuple[Any, Any]:
    """Pick (action, log_prob) out of an actor head's output.

    For dict outputs, the deterministic action is the distribution's mode
    (``mean`` for a Gaussian, ``tanh(mean)`` for a squashed Gaussian -- the
    raw ``mean`` entry of a squashed head is the PRE-tanh value).
    """
    if isinstance(actor_out, dict):
        if deterministic:
            dist = actor_out.get("dist")
            if dist is not None and hasattr(dist, "mode"):
                action = dist.mode()
            else:
                action = actor_out.get("mean", actor_out.get("action"))
        else:
            action = actor_out.get("action")
        return action, actor_out.get("log_prob")
    if isinstance(actor_out, tuple):
        return actor_out[0], actor_out[1]
    return actor_out, None


class OnPolicyActionMixin:
    """Adds action-bound bookkeeping + clipping to an on-policy agent."""

    _action_low: torch.Tensor | None = None
    _action_high: torch.Tensor | None = None

    def set_action_bounds(self, low: Any, high: Any) -> None:
        """Set the action range that env-bound actions are clipped to."""
        if low is None or high is None:
            self._action_low = self._action_high = None
            return
        dev = self.device  # type: ignore[attr-defined]
        self._action_low = torch.as_tensor(np.asarray(low, dtype=np.float32), device=dev)
        self._action_high = torch.as_tensor(np.asarray(high, dtype=np.float32), device=dev)

    def set_action_bounds_from_env(self, env: Any) -> bool:
        bounds = env_action_bounds(env)
        if bounds is None:
            return False
        self.set_action_bounds(*bounds)
        return True

    @property
    def action_bounds(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self._action_low is None or self._action_high is None:
            return None
        return self._action_low, self._action_high

    def clip_action(self, action: Any) -> Any:
        """Clip an action (tensor or ndarray) to the configured bounds.

        No-op when no bounds are known.
        """
        if self._action_low is None or self._action_high is None or action is None:
            return action
        if isinstance(action, torch.Tensor):
            low = self._action_low.to(action.device, action.dtype)
            high = self._action_high.to(action.device, action.dtype)
            return torch.max(torch.min(action, high), low)
        low = self._action_low.cpu().numpy()
        high = self._action_high.cpu().numpy()
        return np.clip(action, low, high)

    def _action_bounds_payload(self) -> dict[str, Any] | None:
        if self.action_bounds is None:
            return None
        return {
            "low": self._action_low.detach().cpu().numpy(),
            "high": self._action_high.detach().cpu().numpy(),
        }

    def _load_action_bounds_payload(self, payload: dict[str, Any] | None) -> None:
        if payload is not None:
            self.set_action_bounds(payload.get("low"), payload.get("high"))
