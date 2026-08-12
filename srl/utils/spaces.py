"""Shared helpers for action/observation spaces that may not be real
gymnasium.Space objects.

Isaac Lab and mjlab envs (via IsaacLabWrapper) expose their own lightweight
space dataclass (shape/low/high/dtype only -- see mjlab.utils.spaces.Box) in
`act_space`/`obs_space`, not a real `gymnasium.spaces.Box`. Any code that
touches those attributes and assumes the full gymnasium API (`.sample()`,
finite bounds) will break against them. Route such code through here instead
of calling `.sample()` etc. directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_action_space(space: Any) -> np.ndarray:
    """Uniform-random sample from an action space, for off-policy warmup.

    Real gymnasium spaces have `.sample()`; isaaclab/mjlab's lightweight
    space dataclass doesn't, so fall back to a plain numpy uniform draw over
    its bounds.

    isaaclab/mjlab action spaces are typically declared unbounded
    ([-inf, inf] -- action terms do their own internal scale/clip), since the
    raw policy output is what the space describes, not the physical actuator
    range. `np.random.uniform` can't sample a non-finite range (raises
    OverflowError), and a genuinely unbounded warmup action wouldn't be a
    sane "explore near zero" sample anyway -- fall back to [-1, 1], the
    conventional bounded range a Tanh/Gaussian-squashed policy actually
    outputs before that internal scaling, on any non-finite bound.
    """
    if hasattr(space, "sample"):
        return space.sample()

    shape = getattr(space, "shape", ())
    low = np.broadcast_to(np.asarray(getattr(space, "low", -1.0), dtype=np.float32), shape).copy()
    high = np.broadcast_to(np.asarray(getattr(space, "high", 1.0), dtype=np.float32), shape).copy()
    low[~np.isfinite(low)] = -1.0
    high[~np.isfinite(high)] = 1.0
    return np.random.uniform(low, high).astype(np.float32)
