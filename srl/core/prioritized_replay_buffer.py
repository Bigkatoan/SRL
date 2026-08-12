"""PrioritizedReplayBuffer — PER with SumSegmentTree / MinSegmentTree."""

from __future__ import annotations

import numpy as np

from srl.core.replay_buffer import ReplayBatch, ReplayBuffer

# ──────────────────────────────────────────────────────────────────────────────
# Segment Trees
# ──────────────────────────────────────────────────────────────────────────────


class _SegmentTree:
    """Base segment tree backed by a flat array of size 2*capacity."""

    def __init__(self, capacity: int, operation, neutral: float) -> None:
        self._cap = capacity
        self._op = operation
        self._neutral = neutral
        self._tree = np.full(2 * capacity, neutral, dtype=np.float64)

    def _propagate(self, idx: int) -> None:
        parent = idx >> 1
        while parent >= 1:
            self._tree[parent] = self._op(self._tree[2 * parent], self._tree[2 * parent + 1])
            parent >>= 1

    def update(self, idx: int, value: float) -> None:
        idx += self._cap
        self._tree[idx] = value
        self._propagate(idx)

    def query(self, left: int, right: int) -> float:
        """Query op over [left, right) in leaf-space."""
        result = self._neutral
        left += self._cap
        right += self._cap
        while left < right:
            if left & 1:
                result = self._op(result, self._tree[left])
                left += 1
            if right & 1:
                right -= 1
                result = self._op(result, self._tree[right])
            left >>= 1
            right >>= 1
        return result


class SumTree(_SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, np.add, 0.0)

    def total(self) -> float:
        return self._tree[1]

    def find_prefixsum(self, prefix: float) -> int:
        """Find smallest index whose prefix sum >= *prefix*."""
        idx = 1
        while idx < self._cap:
            if self._tree[2 * idx] >= prefix:
                idx = 2 * idx
            else:
                prefix -= self._tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self._cap


class MinTree(_SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, np.minimum, np.inf)

    def min(self) -> float:
        return self._tree[1]


# ──────────────────────────────────────────────────────────────────────────────
# PrioritizedReplayBuffer
# ──────────────────────────────────────────────────────────────────────────────


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay (Schaul et al., 2015).

    Parameters
    ----------
    alpha:
        Prioritisation exponent.  0 = uniform, 1 = fully prioritised.
    beta_start:
        Initial IS-weight exponent (anneals → 1.0 over training).
    beta_steps:
        Number of steps over which beta anneals from *beta_start* to 1.
    eps:
        Small constant added to |td_error| to avoid zero priority.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape,
        action_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 1_000_000,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(capacity, obs_shape, action_dim, **kwargs)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.eps = eps
        self._step = 0
        self._max_priority = 1.0

        self._sum_tree = SumTree(capacity)
        self._min_tree = MinTree(capacity)

    # ------------------------------------------------------------------
    # Write — assign max priority to new transitions
    # ------------------------------------------------------------------

    def _write(self, obs, action, reward, next_obs, done, truncated=None) -> None:
        idx = self._pos
        super()._write(obs, action, reward, next_obs, done, truncated)
        priority = self._max_priority**self.alpha
        self._sum_tree.update(idx, priority)
        self._min_tree.update(idx, priority)

    # ------------------------------------------------------------------
    # Sample with IS weights
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> ReplayBatch:
        self._step += 1
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self._step / self.beta_steps)

        indices = np.empty(batch_size, dtype=np.int64)
        total = self._sum_tree.total()
        segment = total / batch_size
        for i in range(batch_size):
            mass = np.random.uniform(segment * i, segment * (i + 1))
            indices[i] = self._sum_tree.find_prefixsum(mass)

        # IS weights
        min_prob = self._min_tree.min() / total
        max_weight = (min_prob * self._size) ** (-beta)
        probs = np.array([self._sum_tree._tree[idx + self._sum_tree._cap] for idx in indices])
        probs /= total
        weights = (probs * self._size) ** (-beta) / max_weight
        weights = weights.astype(np.float32)

        batch = self._make_batch(indices, weights)
        return batch

    # ------------------------------------------------------------------
    # Priority update (called by algorithm after computing TD errors)
    # ------------------------------------------------------------------

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for idx, p in zip(indices, priorities, strict=True):
            self._sum_tree.update(int(idx), float(p))
            self._min_tree.update(int(idx), float(p))
        self._max_priority = max(self._max_priority, priorities.max())
