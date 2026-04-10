"""GoalEnv wrapper for gymnasium-robotics environments (Fetch, AntMaze, etc.).

Flattens the ``{'observation', 'achieved_goal', 'desired_goal'}`` dict into a
single concatenated vector keyed as ``"state"`` for compatibility with SRL's
MLP encoders.  The original goal dict is preserved in ``info`` so HER buffers
can use ``achieved_goal`` / ``desired_goal`` directly.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class GoalEnvWrapper(gym.Wrapper):
    """Wraps a GoalEnv (dictionary-obs) into a flat Box observation.

    The flat observation is ``[observation || achieved_goal || desired_goal]``.

    Parameters
    ----------
    env:
        A ``gymnasium`` GoalEnv whose ``observation_space`` is a ``Dict``
        with keys ``'observation'``, ``'achieved_goal'``, ``'desired_goal'``.
    obs_key:
        Name used as the dict key returned by ``reset`` / ``step``.
        Defaults to ``"state"`` (matching SRL MLP encoder input).
    include_goal:
        Whether to concatenate goal vectors into the flat obs.
        Set to ``False`` to return only the raw observation part.
    """

    def __init__(
        self,
        env: gym.Env,
        obs_key: str = "state",
        include_goal: bool = True,
    ) -> None:
        super().__init__(env)
        self.obs_key = obs_key
        self.include_goal = include_goal
        self.num_envs = 1

        obs_space = env.observation_space
        obs_dim   = int(np.prod(obs_space["observation"].shape))
        ag_dim    = int(np.prod(obs_space["achieved_goal"].shape))
        dg_dim    = int(np.prod(obs_space["desired_goal"].shape))

        if include_goal:
            total_dim = obs_dim + ag_dim + dg_dim
        else:
            total_dim = obs_dim

        self.flat_obs_dim = total_dim
        # Expose a flat observation_space so downstream code can query it
        low  = np.full(total_dim, -np.inf, dtype=np.float32)
        high = np.full(total_dim, +np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------

    def _flatten(self, goal_obs: dict) -> np.ndarray:
        obs = goal_obs["observation"].astype(np.float32)
        if self.include_goal:
            ag = goal_obs["achieved_goal"].astype(np.float32)
            dg = goal_obs["desired_goal"].astype(np.float32)
            return np.concatenate([obs.ravel(), ag.ravel(), dg.ravel()])
        return obs.ravel()

    # ------------------------------------------------------------------

    @property
    def obs_space(self) -> gym.Space:
        return self.observation_space

    @property
    def act_space(self) -> gym.Space:
        return self.env.action_space

    def reset(self, **kwargs) -> tuple[dict[str, np.ndarray], dict]:
        raw_obs, info = self.env.reset(**kwargs)
        info["goal_obs"] = raw_obs          # preserve for HER
        return {self.obs_key: self._flatten(raw_obs)}, info

    def step(self, action: np.ndarray):
        raw_obs, reward, terminated, truncated, info = self.env.step(action)
        info["goal_obs"] = raw_obs          # preserve for HER
        done = terminated or truncated
        flat = self._flatten(raw_obs)
        return {self.obs_key: flat}, float(reward), done, truncated, info
