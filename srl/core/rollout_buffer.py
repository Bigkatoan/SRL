"""RolloutBuffer — on-policy storage for PPO, A2C, A3C.

Stores a fixed window of (obs, action, reward, value, log_prob, done) tuples
collected from *n_envs* parallel environments over *n_steps* timesteps.
Supports GAE-λ advantage computation and mini-batch iteration.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    hidden_states: torch.Tensor | None = None
    cell_states: torch.Tensor | None = None


class RolloutBuffer:
    """On-policy buffer with GAE-λ computation.

    Shapes are detected lazily from the first :meth:`add` call so
    ``obs_shape`` and ``action_dim`` are not required at construction.

    Parameters
    ----------
    n_steps / capacity:
        Steps per environment per rollout (either name accepted).
    n_envs / num_envs:
        Number of parallel environments.
    gamma:
        Discount factor.
    lam / gae_lambda:
        GAE-λ parameter.
    device:
        Torch device for returned tensors.
    """

    def __init__(
        self,
        n_steps: int = 2048,
        n_envs: int = 1,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: str | torch.device = "cpu",
        capacity: int | None = None,
        num_envs: int | None = None,
        gae_lambda: float | None = None,
        obs_shape=None,
        action_dim=None,
        use_fp16: bool = False,
    ) -> None:
        self.n_steps = capacity if capacity is not None else n_steps
        self.n_envs = num_envs if num_envs is not None else n_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda if gae_lambda is not None else lam
        self.device = torch.device(device)
        self._dtype = np.float16 if use_fp16 else np.float32

        self._initialized = False
        self._reset_scalars()

    def _reset_scalars(self) -> None:
        S, N = self.n_steps, self.n_envs
        self._obs: dict[str, np.ndarray] = {}
        self._actions: np.ndarray | None = None
        self._rewards = np.zeros((S, N), dtype=np.float32)
        self._dones = np.zeros((S, N), dtype=np.float32)
        self._values = np.zeros((S, N), dtype=np.float32)
        self._log_probs = np.zeros((S, N), dtype=np.float32)
        self.advantages = np.zeros((S, N), dtype=np.float32)
        self.returns = np.zeros((S, N), dtype=np.float32)
        self._hidden: np.ndarray | None = None
        self._cell: np.ndarray | None = None
        self._pos = 0
        self._full = False

    def reset(self) -> None:
        self._initialized = False
        self._reset_scalars()

    def _lazy_init(self, obs: dict, action: np.ndarray) -> None:
        S, N = self.n_steps, self.n_envs
        for k, v in obs.items():
            single = np.asarray(v)
            shape = single.shape[1:] if single.shape[0] == N and single.ndim > 1 else single.shape
            self._obs[k] = np.zeros((S, N, *shape), dtype=np.float32)
        a = np.asarray(action)
        act_shape = (
            a.shape[1:]
            if a.shape[0] == N and a.ndim > 1
            else (a.shape[-1],) if a.ndim > 0 else (1,)
        )
        self._actions = np.zeros((S, N, *act_shape), dtype=np.float32)
        self._initialized = True

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward,
        done,
        value=None,
        log_prob=None,
        truncated=None,
        hidden: np.ndarray | None = None,
        cell: np.ndarray | None = None,
    ) -> None:
        if not self._initialized:
            self._lazy_init(obs, action)

        t = self._pos
        for k, v in obs.items():
            self._obs[k][t] = np.asarray(v, dtype=np.float32)
        self._actions[t] = np.asarray(action, dtype=np.float32)
        self._rewards[t] = np.asarray(reward, dtype=np.float32).reshape(self.n_envs)
        self._dones[t] = np.asarray(done, dtype=np.float32).reshape(self.n_envs)
        if value is not None:
            self._values[t] = np.asarray(value, dtype=np.float32).reshape(self.n_envs)
        if log_prob is not None:
            self._log_probs[t] = np.asarray(log_prob, dtype=np.float32).reshape(self.n_envs)
        if hidden is not None and cell is not None:
            h, c = np.asarray(hidden), np.asarray(cell)
            if self._hidden is None:
                self._hidden = np.zeros((self.n_steps, *h.shape), dtype=np.float32)
                self._cell = np.zeros((self.n_steps, *c.shape), dtype=np.float32)
            self._hidden[t] = h
            self._cell[t] = c

        self._pos = (self._pos + 1) % self.n_steps
        if self._pos == 0:
            self._full = True

    def compute_returns_and_advantages(
        self,
        last_value=0.0,
        last_dones: np.ndarray | None = None,
        # legacy alias
        last_values: np.ndarray | None = None,
    ) -> None:
        """Compute GAE-λ in-place.

        Parameters
        ----------
        last_value:
            Bootstrap value; scalar, ``(n_envs,)`` or ``(n_envs, 1)``.
        last_dones:
            Terminal flags for the step *after* the buffer; defaults to zeros.
        """
        lv_raw = last_values if last_values is not None else last_value
        lv = np.asarray(lv_raw, dtype=np.float32).reshape(self.n_envs)
        ld = (
            np.asarray(last_dones, dtype=np.float32).reshape(self.n_envs)
            if last_dones is not None
            else np.zeros(self.n_envs, dtype=np.float32)
        )

        last_gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - ld
                next_values = lv
            else:
                next_non_terminal = 1.0 - self._dones[t + 1]
                next_values = self._values[t + 1]

            delta = (
                self._rewards[t] + self.gamma * next_values * next_non_terminal - self._values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self._values

    def get_batches(self, batch_size: int) -> Generator[RolloutBatch, None, None]:
        """Yield shuffled mini-batches over the full buffer."""
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        def _flat(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(total, *arr.shape[2:])

        obs_flat = {k: _flat(v) for k, v in self._obs.items()}
        actions_flat = _flat(self._actions)
        lp_flat = _flat(self._log_probs)
        adv_flat = _flat(self.advantages)
        ret_flat = _flat(self.returns)
        val_flat = _flat(self._values)
        h_flat = _flat(self._hidden) if self._hidden is not None else None
        c_flat = _flat(self._cell) if self._cell is not None else None

        dev = self.device

        def _t(x):
            return torch.from_numpy(x.copy()).float().to(dev)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            yield RolloutBatch(
                obs={k: _t(v[idx]) for k, v in obs_flat.items()},
                actions=_t(actions_flat[idx]),
                log_probs=_t(lp_flat[idx]),
                advantages=_t(adv_flat[idx]),
                returns=_t(ret_flat[idx]),
                values=_t(val_flat[idx]),
                hidden_states=_t(h_flat[idx]) if h_flat is not None else None,
                cell_states=_t(c_flat[idx]) if c_flat is not None else None,
            )

    def get_batch(self) -> RolloutBatch:
        """Return the whole buffer as a single batch (no shuffling)."""
        total = self.n_steps * self.n_envs

        def _flat(arr):
            return arr.reshape(total, *arr.shape[2:])

        dev = self.device

        def _t(x):
            return torch.from_numpy(x.copy()).float().to(dev)

        return RolloutBatch(
            obs={k: _t(_flat(v)) for k, v in self._obs.items()},
            actions=_t(_flat(self._actions)),
            log_probs=_t(_flat(self._log_probs)),
            advantages=_t(_flat(self.advantages)),
            returns=_t(_flat(self.returns)),
            values=_t(_flat(self._values)),
        )

    def is_full(self) -> bool:
        return self._full or self._pos == 0

    def __len__(self) -> int:
        return self.n_steps * self.n_envs if self._full else self._pos * self.n_envs
