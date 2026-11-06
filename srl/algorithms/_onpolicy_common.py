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
:meth:`srl.core.base_agent.OnPolicyAgent.set_action_bounds`
(the bound-tracking / clipping code lives on that class).
TODO(envlayer): once env-side action rescaling lands (model acts in
[-1, 1], wrapper maps to the env's Box), the env wrapper should expose the
normalized Box as its action space so these bounds become [-1, 1]
automatically.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
