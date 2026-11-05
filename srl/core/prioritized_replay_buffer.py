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

    Proportional variant: ``P(i) = p_i^alpha / sum_j p_j^alpha`` with
    ``p_i = |td_i| + eps``. :meth:`sample` returns importance-sampling
    weights ``w_i = (N * P(i))^-beta / max_j w_j`` in ``batch.weights`` and the
    sampled slots in ``batch.indices``; the algorithm must weight its critic
    loss by ``weights`` and call :meth:`update_priorities` with the new TD
    errors (SAC/TD3/DDPG do this when ``use_per=True``).

    Parameters
    ----------
    alpha:
        Prioritisation exponent.  0 = uniform, 1 = fully prioritised.
    beta_start:
        Initial IS-weight exponent (anneals → 1.0 over training).
    beta_steps:
        Number of :meth:`sample` calls over which beta anneals to 1.
    eps:
        Small constant added to |td_error| to avoid zero priority.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape=None,
        action_dim: int | None = None,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 1_000_000,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(capacity, obs_shape, action_dim, **kwargs)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = max(int(beta_steps), 1)
        self.eps = eps
        self._step = 0
        # Max *raw* priority (|td| + eps), exponentiated by alpha on write.
        self._max_priority = 1.0

        # The top-down prefix-sum descent in SumTree.find_prefixsum is only
        # valid for a complete binary tree, i.e. a power-of-two leaf count.
        tree_cap = 1
        while tree_cap < capacity:
            tree_cap *= 2
        self._sum_tree = SumTree(tree_cap)
        self._min_tree = MinTree(tree_cap)

    # ------------------------------------------------------------------
    # Write — assign max priority to new transitions
    # ------------------------------------------------------------------

    def _write(self, obs, action, reward, next_obs, done, truncated=None, discount=None) -> None:
        idx = self._pos
        super()._write(obs, action, reward, next_obs, done, truncated, discount)
        priority = self._max_priority**self.alpha
        self._sum_tree.update(idx, priority)
        self._min_tree.update(idx, priority)

    # ------------------------------------------------------------------
    # Sample with IS weights
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self._step / self.beta_steps)

    def sample(self, batch_size: int) -> ReplayBatch:
        self._step += 1
        beta = self.beta

        total = self._sum_tree.total()
        segment = total / batch_size
        mass = (np.arange(batch_size) + np.random.uniform(size=batch_size)) * segment
        indices = np.array([self._sum_tree.find_prefixsum(m) for m in mass], dtype=np.int64)
        # Floating-point round-off at the very top of the range can land on
        # an empty (priority 0) leaf past the filled region.
        indices = np.minimum(indices, self._size - 1)

        leaf = self._sum_tree._cap
        probs = self._sum_tree._tree[indices + leaf] / total
        min_prob = self._min_tree.min() / total
        max_weight = (min_prob * self._size) ** (-beta)
        weights = ((probs * self._size) ** (-beta) / max_weight).astype(np.float32)

        return self._make_batch(indices, weights)

    # ------------------------------------------------------------------
    # Priority update (called by algorithm after computing TD errors)
    # ------------------------------------------------------------------

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        raw = np.abs(np.asarray(td_errors, dtype=np.float64)).reshape(-1) + self.eps
        priorities = raw**self.alpha
        for idx, p in zip(np.asarray(indices).reshape(-1), priorities, strict=True):
            self._sum_tree.update(int(idx), float(p))
            self._min_tree.update(int(idx), float(p))
        self._max_priority = max(self._max_priority, float(raw.max()))

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["per"] = {
            "alpha": self.alpha,
            "beta_start": self.beta_start,
            "beta_steps": self.beta_steps,
            "eps": self.eps,
            "step": self._step,
            "max_priority": self._max_priority,
            "sum_tree": self._sum_tree._tree.copy(),
            "min_tree": self._min_tree._tree.copy(),
        }
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        per = state.get("per")
        if per is None:
            # Plain ReplayBuffer checkpoint: give every stored slot max priority.
            for i in range(self._size):
                self._sum_tree.update(i, self._max_priority**self.alpha)
                self._min_tree.update(i, self._max_priority**self.alpha)
            return
        self._step = int(per["step"])
        self._max_priority = float(per["max_priority"])
        if per["sum_tree"].shape == self._sum_tree._tree.shape:
            self._sum_tree._tree[:] = per["sum_tree"]
            self._min_tree._tree[:] = per["min_tree"]


def build_replay_buffer(cfg, **kwargs) -> ReplayBuffer:
    """Uniform or prioritized CPU replay buffer from an algorithm config."""
    common = dict(
        capacity=cfg.buffer_size,
        num_envs=cfg.replay_num_envs,
        n_step=cfg.replay_n_step,
        gamma=cfg.gamma,
        use_fp16=cfg.use_fp16,
    )
    common.update(kwargs)
    if getattr(cfg, "use_per", False):
        return PrioritizedReplayBuffer(
            alpha=float(getattr(cfg, "per_alpha", 0.6)),
            beta_start=float(getattr(cfg, "per_beta_start", 0.4)),
            beta_steps=int(getattr(cfg, "per_beta_steps", 1_000_000)),
            eps=float(getattr(cfg, "per_eps", 1e-6)),
            **common,
        )
    return ReplayBuffer(**common)
