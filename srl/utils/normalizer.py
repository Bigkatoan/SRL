"""Running mean/variance normaliser (Welford's online algorithm)."""

from __future__ import annotations

import numpy as np


class RunningNormalizer:
    """Normalise observations or rewards online.

    Thread-safe for single-process use. For multi-process sync, call
    ``sync()`` with another normaliser.

    Parameters
    ----------
    shape:
        Shape of the data to normalise (e.g. ``(obs_dim,)``).
    clip:
        Clip normalised values to ``[-clip, clip]``.
    eps:
        Numerical stability constant.
    initial_count:
        Pseudo-count of the initial (mean=0, var=1) prior. SB3's
        ``RunningMeanStd`` uses ``1e-4``, which keeps the very first updates
        from collapsing the variance to exactly zero.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        clip: float = 10.0,
        eps: float = 1e-8,
        initial_count: float = 0.0,
    ) -> None:
        self.shape = shape
        self.clip = clip
        self.eps = eps
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count: float = initial_count

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a batch of samples."""
        if x.ndim == len(self.shape):
            x = x[None]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        new_var = m2 / total

        self.mean = new_mean
        self.var = new_var
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return x * np.sqrt(self.var + self.eps) + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"].copy()
        self.var = d["var"].copy()
        self.count = float(d["count"])
