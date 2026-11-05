"""Normalised action domain: the policy acts in [-1, 1], the env sees [low, high].

SRL's continuous actor heads squash with ``tanh`` (SAC/TD3/DDPG) or are
clipped to ``[-1, 1]`` (PPO/A2C), and off-policy warmup samples uniformly from
the env's ``act_space``. Without a rescaling layer the policy can only reach
the ``[-1, 1]`` slice of the env's real action range (e.g. half of
Pendulum's ``[-2, 2]`` torque) or saturates a narrower one (Humanoid's
``[-0.4, 0.4]``), and warmup actions stored in the replay buffer live in a
different domain than the policy's own actions.

:class:`RescaleActionWrapper` fixes that the SB3 way: the wrapped env's
``act_space``/``action_space`` is ``Box(-1, 1)`` on every dimension with
finite bounds, and ``step()`` maps the (clipped) normalised action affinely
onto ``[low, high]`` just before it reaches the underlying env. Dimensions
with an infinite bound are passed through untouched (and keep their
infinite bounds in the reported space). Buffers, warmup and the policy all
operate in the normalised domain.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


def _is_box(space: Any) -> bool:
    return isinstance(space, gym.spaces.Box) and np.issubdtype(space.dtype, np.floating)


def needs_action_rescale(space: Any) -> bool:
    """Whether ``space`` is a float Box with some finite dim not already [-1, 1]."""
    if not _is_box(space):
        return False
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)
    finite = np.isfinite(low) & np.isfinite(high)
    if not finite.any():
        return False
    return not (np.allclose(low[finite], -1.0) and np.allclose(high[finite], 1.0))


class RescaleActionWrapper(gym.Wrapper):
    """Expose a normalised ``[-1, 1]`` action space over an SRL-interface env.

    Wraps an SRL env (``GymnasiumWrapper``, ``GoalEnvWrapper``,
    ``RacecarWrapper``, ...) or a raw Gymnasium env. Any attribute not defined
    here (``obs_space``, ``num_envs``, ``include_goal``, ...) is forwarded to
    the wrapped env.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        space = env.action_space
        if not _is_box(space):
            raise TypeError(f"RescaleActionWrapper needs a float Box action space, got {space}")
        low = np.asarray(space.low, dtype=np.float64)
        high = np.asarray(space.high, dtype=np.float64)
        self._finite = np.isfinite(low) & np.isfinite(high)
        self._env_low = np.where(self._finite, low, 0.0)
        self._env_high = np.where(self._finite, high, 0.0)
        norm_low = np.where(self._finite, -1.0, low).astype(space.dtype)
        norm_high = np.where(self._finite, 1.0, high).astype(space.dtype)
        self._dtype = space.dtype
        self.action_space = gym.spaces.Box(
            low=norm_low, high=norm_high, shape=space.shape, dtype=space.dtype
        )

    def __getattr__(self, name: str):
        # gymnasium>=1.0 Wrappers no longer forward unknown attributes; SRL
        # callers rely on e.g. `env.obs_space`/`env.num_envs`/`env.include_goal`.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def obs_space(self) -> gym.Space:
        return getattr(self.env, "obs_space", self.env.observation_space)

    @property
    def act_space(self) -> gym.Space:
        return self.action_space

    @property
    def env_action_space(self) -> gym.Space:
        """The wrapped env's own (un-normalised) action space."""
        return self.env.action_space

    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        """Normalised ``[-1, 1]`` action -> env units."""
        a = np.asarray(action, dtype=np.float64).reshape(self.action_space.shape)
        clipped = np.clip(a, -1.0, 1.0)
        scaled = self._env_low + (clipped + 1.0) * 0.5 * (self._env_high - self._env_low)
        return np.where(self._finite, scaled, a).astype(self._dtype)

    def unscale_action(self, env_action: np.ndarray) -> np.ndarray:
        """Env-unit action -> normalised ``[-1, 1]`` action."""
        a = np.asarray(env_action, dtype=np.float64).reshape(self.action_space.shape)
        span = np.where(self._finite, self._env_high - self._env_low, 1.0)
        norm = 2.0 * (a - self._env_low) / span - 1.0
        return np.where(self._finite, norm, a).astype(self._dtype)

    def step(self, action):
        return self.env.step(self.rescale_action(action))


def maybe_rescale_action(env: gym.Env) -> gym.Env:
    """Wrap ``env`` in :class:`RescaleActionWrapper` iff its action space
    needs it (float Box with finite bounds other than [-1, 1])."""
    if needs_action_rescale(getattr(env, "action_space", None)):
        return RescaleActionWrapper(env)
    return env
