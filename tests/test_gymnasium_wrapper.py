"""Regression test for the missing HWC->CHW transpose in GymnasiumWrapper.

configs/envs/car_racing_ppo_visual.yaml (a CNN-encoder config against a real
Gymnasium pixel env) crashed on the very first observation with a channel
mismatch: Gymnasium's native pixel envs (CarRacing, etc.) return (H, W, C),
but SRL's CNNEncoder expects (C, H, W) -- IsaacLabWrapper already transposed
for this, GymnasiumWrapper never did.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from srl.envs.gymnasium_wrapper import GymnasiumWrapper


class _FakePixelEnv(gym.Env):
    """Mimics a Gymnasium pixel env's native (H, W, C) observation shape."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(96, 96, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros((96, 96, 3), dtype=np.uint8), {}

    def step(self, action):
        return np.zeros((96, 96, 3), dtype=np.uint8), 0.0, False, False, {}


class _FakeStateEnv(gym.Env):
    """A plain low-dim state env -- must NOT be mistaken for an image and
    transposed (its last dim can coincidentally be small, e.g. 3 or 4)."""

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 0.0, False, False, {}


def test_gymnasium_wrapper_transposes_hwc_pixel_obs_to_chw_on_reset_and_step() -> None:
    env = GymnasiumWrapper(_FakePixelEnv())

    obs, _ = env.reset()
    assert obs["state"].shape == (3, 96, 96)

    next_obs, *_ = env.step(np.zeros(3, dtype=np.float32))
    assert next_obs["state"].shape == (3, 96, 96)


def test_gymnasium_wrapper_leaves_flat_state_obs_untouched() -> None:
    env = GymnasiumWrapper(_FakeStateEnv())

    obs, _ = env.reset()
    assert obs["state"].shape == (4,)
    assert obs["state"].tolist() == [1.0, 2.0, 3.0, 4.0]
