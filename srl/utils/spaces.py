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


def _is_continuous_box(space: Any) -> bool:
    if not (hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape")):
        return False
    dtype = getattr(space, "dtype", np.float32)
    try:
        return bool(np.issubdtype(np.dtype(dtype), np.floating))
    except TypeError:
        return True


def sample_action_space(space: Any) -> np.ndarray:
    """Uniform-random sample from an action space, for off-policy warmup.

    Continuous (Box-like) spaces are sampled UNIFORMLY over their bounds,
    with any non-finite bound replaced by [-1, 1]. This covers both real
    gymnasium Boxes and isaaclab/mjlab's lightweight space dataclass (which
    has no `.sample()`), and keeps warmup actions in the same domain the
    policy acts in: SRL's CLI exposes every finite-bounded env through
    `srl.envs.action_rescale.RescaleActionWrapper` (so the space already is
    [-1, 1]), and a tanh-squashed/clipped policy on an unbounded space also
    only ever outputs [-1, 1]. (gymnasium's own `Box.sample()` would instead
    draw from a normal/exponential distribution on unbounded dims.)

    The draw uses the space's own seeded generator (`space.np_random`,
    seeded via `space.seed(...)`) when it has one, else NumPy's global RNG
    (seeded by `srl.utils.set_random_seed`) -- so warmup is reproducible.
    Discrete/other spaces fall back to `space.sample()`.
    """
    if not _is_continuous_box(space):
        return space.sample()

    shape = tuple(getattr(space, "shape", ()) or ())
    low = np.broadcast_to(np.asarray(space.low, dtype=np.float64), shape).copy()
    high = np.broadcast_to(np.asarray(space.high, dtype=np.float64), shape).copy()
    low[~np.isfinite(low)] = -1.0
    high[~np.isfinite(high)] = 1.0
    rng = getattr(space, "np_random", None) if hasattr(space, "sample") else None
    draw = rng.uniform(low, high) if rng is not None else np.random.uniform(low, high)
    return np.asarray(draw, dtype=np.float32)


def sample_vector_actions(env: Any) -> np.ndarray:
    """Warmup actions for every sub-env of a (possibly vectorised) env.

    Returns shape ``(num_envs, *action_shape)`` for vector envs
    (``num_envs > 1`` or an env exposing a batched ``act_space`` plus
    ``single_act_space``, i.e. isaaclab/mjlab), else one unbatched action.
    """
    num_envs = int(getattr(env, "num_envs", 1) or 1)
    space = env.act_space
    single = getattr(env, "single_act_space", None)
    if single is not None and single is not space:
        return np.stack([sample_action_space(single) for _ in range(num_envs)], axis=0)
    if num_envs > 1:
        shape = tuple(getattr(space, "shape", ()) or ())
        if single is None and len(shape) >= 2 and shape[0] == num_envs:
            # Already-batched space with no per-env space exposed.
            return sample_action_space(space)
        return np.stack([sample_action_space(space) for _ in range(num_envs)], axis=0)
    return sample_action_space(space)
