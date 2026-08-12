"""FrameStackEncoder — stacks k consecutive frames on the channel axis."""

from __future__ import annotations

from collections import deque

import numpy as np


class FrameStackPreprocessor:
    """CPU-side preprocessor that keeps a rolling window of k frames.

    Usage (outside the model)::

        fsp = FrameStackPreprocessor(k=4, obs_shape=(3, 84, 84))
        fsp.reset()
        ...
        stacked = fsp.push(new_frame)  # (3*k, 84, 84)

    This is intentionally NOT an nn.Module — it lives in the data pipeline.
    """

    def __init__(self, k: int, obs_shape: tuple[int, int, int]) -> None:
        self.k = k
        self.c, self.h, self.w = obs_shape
        self._buf: deque = deque(maxlen=k)

    def reset(self, first_obs=None) -> None:
        zero = np.zeros((self.c, self.h, self.w), dtype=np.float32)
        for _ in range(self.k):
            self._buf.append(zero)
        if first_obs is not None:
            self._buf.append(first_obs)

    def push(self, obs) -> np.ndarray:
        self._buf.append(obs)
        return np.concatenate(list(self._buf), axis=0)  # (C*k, H, W)

    @property
    def stacked_channels(self) -> int:
        return self.c * self.k
