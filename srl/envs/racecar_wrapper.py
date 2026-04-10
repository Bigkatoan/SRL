"""Wrapper for racecar_gym dictionary observations/actions.

This adapter makes racecar_gym compatible with SRL by:
1) flattening dict observations into a single state vector
2) converting a flat action vector back into racecar_gym's Dict action format

It is intentionally generic and works for any env whose observation/action
spaces are gymnasium.spaces.Dict of Box spaces.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import gymnasium as gym
import numpy as np


class RacecarWrapper(gym.Wrapper):
    """Flatten dict obs and map flat action vectors for racecar_gym."""

    def __init__(self, env: gym.Env, obs_key: str = "state") -> None:
        super().__init__(env)
        self.obs_key = obs_key
        self.num_envs = 1

        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise TypeError("RacecarWrapper expects Dict observation_space")
        if not isinstance(env.action_space, gym.spaces.Dict):
            raise TypeError("RacecarWrapper expects Dict action_space")

        # Keep deterministic key order for stable flatten/unflatten behavior.
        self._obs_spaces: OrderedDict[str, gym.Space] = OrderedDict(env.observation_space.spaces)
        self._act_spaces: OrderedDict[str, gym.Space] = OrderedDict(env.action_space.spaces)

        self._act_slices: dict[str, tuple[int, int]] = {}
        start = 0
        low_parts: list[np.ndarray] = []
        high_parts: list[np.ndarray] = []

        for key, space in self._act_spaces.items():
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(f"Action space '{key}' must be Box, got {type(space).__name__}")
            size = int(np.prod(space.shape))
            self._act_slices[key] = (start, start + size)
            start += size
            low_parts.append(np.asarray(space.low, dtype=np.float32).ravel())
            high_parts.append(np.asarray(space.high, dtype=np.float32).ravel())

        self.flat_action_dim = start
        self.action_space = gym.spaces.Box(
            low=np.concatenate(low_parts),
            high=np.concatenate(high_parts),
            dtype=np.float32,
        )

        # A conservative unbounded flat observation space.
        obs_dim = sum(int(np.prod(space.shape)) for space in self._obs_spaces.values())
        self.flat_obs_dim = obs_dim
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def obs_space(self) -> gym.Space:
        return self.observation_space

    @property
    def act_space(self) -> gym.Space:
        return self.action_space

    def _flatten_obs(self, obs: dict[str, Any]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key in self._obs_spaces:
            parts.append(np.asarray(obs[key], dtype=np.float32).ravel())
        return np.concatenate(parts)

    def _to_dict_action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != self.flat_action_dim:
            raise ValueError(
                f"Expected flat action dim {self.flat_action_dim}, got {action.shape[0]}"
            )

        out: dict[str, np.ndarray] = {}
        for key, space in self._act_spaces.items():
            s, e = self._act_slices[key]
            out[key] = action[s:e].reshape(space.shape)
        return out

    def reset(self, **kwargs) -> tuple[dict[str, np.ndarray], dict]:
        obs, info = self.env.reset(**kwargs)
        return {self.obs_key: self._flatten_obs(obs)}, info

    def step(self, action: np.ndarray):
        dict_action = self._to_dict_action(action)
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        done = terminated or truncated
        return {self.obs_key: self._flatten_obs(obs)}, float(reward), done, truncated, info
