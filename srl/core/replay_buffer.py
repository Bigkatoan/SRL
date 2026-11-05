"""ReplayBuffer — off-policy circular buffer for SAC and DDPG.

Supports:
* Uniform random sampling
* n-step returns
* Dict observations
* FP16 storage
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReplayBatch:
    observations: torch.Tensor | dict[str, torch.Tensor]
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor | dict[str, torch.Tensor]
    dones: torch.Tensor
    # True termination only (episode actually ended), never OR'd with
    # truncation -- see gymnasium_wrapper.py's step() for why bootstrap
    # targets need this distinction. `truncated` is stored alongside it for
    # callers that do want "episode ended for any reason" (e.g. n-step
    # return accumulation cutting a rollout short).
    truncated: torch.Tensor | None = None
    weights: torch.Tensor | None = None  # IS weights (PER)
    indices: np.ndarray | None = None  # buffer indices (PER update)
    # Per-transition bootstrap discount gamma**k, k = the number of real env
    # steps the stored (possibly n-step) reward spans. Bootstrap targets
    # must use this instead of a bare `gamma`: with n_step > 1 the stored
    # reward already covers k steps, and a window cut short by an episode
    # end has k < n. None = one-step buffer, use `gamma`.
    discounts: torch.Tensor | None = None

    @property
    def obs(self):
        return self.observations

    @property
    def next_obs(self):
        return self.next_observations


class ReplayBuffer:
    """Uniform circular replay buffer with optional n-step returns.

    n-step returns use a per-env *sliding* window (like SB3's
    ``NStepReplayBuffer``): every env step writes exactly one transition
    once the window holds ``n`` steps, and when an episode ends the remaining
    partial windows (k = n-1, ..., 1 steps) are flushed so no data is lost.
    Each stored transition carries ``discount = gamma**k``; ``done`` is the
    true termination of the window's last step (so a time-limit truncation
    still bootstraps, from the final observation, with ``gamma**k``).

    Parameters
    ----------
    capacity:
        Maximum number of transitions.
    obs_shape:
        Shape of a single observation, or ``dict[str, tuple]`` for multi-modal.
    action_dim:
        Dimension of the continuous action vector.
    n_envs:
        Number of environments writing to this buffer in parallel.
    n_step:
        n-step return horizon (1 = standard TD).
    gamma:
        Discount factor (used for n-step returns).
    device:
        Torch device for sampled batches.
    use_fp16:
        Store float32 data as float16 to halve memory.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple | dict[str, tuple] | None = None,
        action_dim: int | None = None,
        n_envs: int = 1,
        n_step: int = 1,
        gamma: float = 0.99,
        device: str | torch.device = "cpu",
        use_fp16: bool = False,
        # aliases
        num_envs: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_envs = num_envs if num_envs is not None else n_envs
        self.n_step = n_step
        self.gamma = gamma
        self.device = torch.device(device)
        self.storage_dtype = np.float16 if use_fp16 else np.float32
        self.tensor_dtype = torch.float16 if use_fp16 else torch.float32

        self._dict_obs = isinstance(obs_shape, dict) if obs_shape is not None else True
        self._initialized = obs_shape is not None and action_dim is not None
        self._pos = 0
        self._size = 0

        if self._initialized:
            self._alloc()
        else:
            self._obs = {}
            self._next_obs = {}
            self._actions = None
            self._rewards = np.zeros((self.capacity,), dtype=np.float32)
            self._dones = np.zeros((self.capacity,), dtype=np.float32)
            self._truncated = np.zeros((self.capacity,), dtype=np.float32)
            self._discounts = np.full((self.capacity,), self.gamma, dtype=np.float32)

        # n-step sliding windows, keyed by env index. A dict (not a list
        # sized by n_envs) so an env index beyond the configured count can
        # never IndexError or silently alias another env's window.
        self._nstep_bufs: dict[int, deque] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _alloc(self) -> None:
        if self._dict_obs:
            self._obs: dict[str, np.ndarray] | np.ndarray = {
                k: np.zeros((self.capacity, *s), dtype=self.storage_dtype)
                for k, s in self.obs_shape.items()
            }
            self._next_obs: dict[str, np.ndarray] | np.ndarray = {
                k: np.zeros((self.capacity, *s), dtype=self.storage_dtype)
                for k, s in self.obs_shape.items()
            }
        else:
            self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=self.storage_dtype)
            self._next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=self.storage_dtype)

        self._actions = np.zeros((self.capacity, self.action_dim), dtype=self.storage_dtype)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._truncated = np.zeros((self.capacity,), dtype=np.float32)
        self._discounts = np.full((self.capacity,), self.gamma, dtype=np.float32)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _lazy_init_from(self, obs: dict, action: np.ndarray, next_obs: dict) -> None:
        self._dict_obs = True
        self._obs = {}
        self._next_obs = {}
        for k, v in obs.items():
            # `add()`/`_write()` store exactly one transition per call (the
            # vectorized CLI loop splits per-env before calling), so the
            # observation handed in here IS the per-transition shape. Dropping
            # a leading axis here used to eat the channel dim of image
            # observations -- a (3, 96, 96) frame allocated (96, 96) slots and
            # every pixel-based off-policy run died on the first add().
            shape = np.asarray(v).shape
            self._obs[k] = np.zeros((self.capacity, *shape), dtype=self.storage_dtype)
            self._next_obs[k] = np.zeros((self.capacity, *shape), dtype=self.storage_dtype)
        a = np.asarray(action)
        act_shape = a.shape[1:] if a.ndim > 1 else a.shape
        self._actions = np.zeros((self.capacity, *act_shape), dtype=self.storage_dtype)
        self._initialized = True

    def add(
        self,
        obs,
        action: np.ndarray | None = None,
        reward=None,
        next_obs=None,
        done=None,
        env_idx: int = 0,
        # keyword-arg aliases
        truncated=None,
    ) -> None:
        """Add a single transition (or pass to n-step buffer if n_step > 1).

        `done` must be TRUE TERMINATION only (episode actually ended), never
        OR'd with `truncated` -- SAC/DDPG/TD3 bootstrap target Q as
        `(1 - dones) * next_q`, gated on true termination. `truncated` is
        stored separately; a time-limit cutoff still ends the n-step
        accumulation (below) but does not zero the one-step bootstrap.
        """
        if not self._initialized and next_obs is not None:
            self._lazy_init_from(obs, action, next_obs)

        effective_done = done
        if effective_done is None:
            effective_done = np.zeros(self.n_envs, dtype=bool)
        effective_trunc = truncated
        if effective_trunc is None:
            effective_trunc = np.zeros(self.n_envs, dtype=bool)

        if self.n_step > 1:
            self._add_nstep(obs, action, reward, next_obs, effective_done, effective_trunc, env_idx)
        else:
            self._write(obs, action, reward, next_obs, effective_done, effective_trunc, self.gamma)

    def _add_nstep(self, obs, action, reward, next_obs, done, trunc, env_idx: int) -> None:
        window = self._nstep_bufs[int(env_idx)]
        r = float(np.asarray(reward, dtype=np.float64).mean())
        d = bool(np.asarray(done).any())
        tr = bool(np.asarray(trunc).any())
        window.append((obs, action, r))
        if d or tr:
            # Episode over: flush every remaining (partial) window, each
            # ending at this step. terminated -> no bootstrap (done=1);
            # truncated -> bootstrap from the final obs with gamma**k.
            while window:
                self._emit_nstep(window, next_obs, d, tr)
                window.popleft()
        elif len(window) >= self.n_step:
            self._emit_nstep(window, next_obs, False, False)
            window.popleft()

    def _emit_nstep(self, window: deque, next_obs, done: bool, trunc: bool) -> None:
        obs0, act0, _ = window[0]
        ret = 0.0
        for i, (_, _, r) in enumerate(window):
            ret += (self.gamma**i) * r
        k = len(window)
        self._write(obs0, act0, ret, next_obs, done, trunc, self.gamma**k)

    def _write(self, obs, action, reward, next_obs, done, truncated=None, discount=None) -> None:
        idx = self._pos
        if self._dict_obs:
            for k in self._obs:
                self._obs[k][idx] = obs[k]
                self._next_obs[k][idx] = next_obs[k]
        else:
            self._obs[idx] = obs
            self._next_obs[idx] = next_obs

        self._actions[idx] = np.asarray(action, dtype=self.storage_dtype)
        self._rewards[idx] = float(np.asarray(reward).mean())
        self._dones[idx] = float(np.asarray(done).any())
        self._truncated[idx] = float(np.asarray(truncated).any()) if truncated is not None else 0.0
        self._discounts[idx] = self.gamma if discount is None else float(discount)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> ReplayBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        return self._make_batch(indices)

    def _make_batch(self, indices: np.ndarray, weights: np.ndarray | None = None) -> ReplayBatch:
        def t(arr):
            return torch.tensor(arr[indices], dtype=self.tensor_dtype, device=self.device)

        obs = {k: t(v) for k, v in self._obs.items()} if self._dict_obs else t(self._obs)
        next_obs = (
            {k: t(v) for k, v in self._next_obs.items()} if self._dict_obs else t(self._next_obs)
        )
        w_tensor = (
            torch.tensor(weights, dtype=torch.float32, device=self.device)
            if weights is not None
            else None
        )
        return ReplayBatch(
            observations=obs,
            actions=t(self._actions),
            rewards=torch.tensor(self._rewards[indices], dtype=torch.float32, device=self.device),
            next_observations=next_obs,
            dones=torch.tensor(self._dones[indices], dtype=torch.float32, device=self.device),
            truncated=torch.tensor(
                self._truncated[indices], dtype=torch.float32, device=self.device
            ),
            weights=w_tensor,
            indices=indices,
            discounts=torch.tensor(
                self._discounts[indices], dtype=torch.float32, device=self.device
            ),
        )

    def state_dict(self) -> dict:
        state = {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_dim": self.action_dim,
            "n_envs": self.n_envs,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "storage_dtype": self.storage_dtype,
            "tensor_dtype": self.tensor_dtype,
            "dict_obs": self._dict_obs,
            "initialized": self._initialized,
            "pos": self._pos,
            "size": self._size,
            "obs": self._obs,
            "next_obs": self._next_obs,
            "actions": self._actions,
            "rewards": self._rewards,
            "dones": self._dones,
            "truncated": self._truncated,
            "discounts": self._discounts,
        }
        if self.n_step > 1:
            state["nstep_bufs"] = {k: list(v) for k, v in self._nstep_bufs.items()}
        return state

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self.obs_shape = state.get("obs_shape")
        self.action_dim = state.get("action_dim")
        self.n_envs = int(state["n_envs"])
        self.n_step = int(state["n_step"])
        self.gamma = float(state["gamma"])
        self.storage_dtype = state.get("storage_dtype", np.float32)
        self.tensor_dtype = state.get("tensor_dtype", torch.float32)
        self._dict_obs = bool(state["dict_obs"])
        self._initialized = bool(state["initialized"])
        self._pos = int(state["pos"])
        self._size = int(state["size"])
        self._obs = state["obs"]
        self._next_obs = state["next_obs"]
        self._actions = state["actions"]
        self._rewards = state["rewards"]
        self._dones = state["dones"]
        # .get() with a fallback: buffers checkpointed before `truncated` was
        # tracked separately have no such key.
        self._truncated = state.get("truncated", np.zeros_like(self._dones))
        # Old checkpoints predate per-transition discounts; their n-step
        # rewards were built with the broken chunked accumulator anyway, so
        # the plain one-step gamma is the only defensible fallback.
        self._discounts = state.get(
            "discounts", np.full_like(self._dones, self.gamma, dtype=np.float32)
        )
        self._nstep_bufs = defaultdict(deque)
        saved = state.get("nstep_bufs")
        if isinstance(saved, dict):
            for k, v in saved.items():
                self._nstep_bufs[int(k)] = deque(v)
        # Legacy list-of-lists windows held a different tuple layout; drop them
        # (at most n-1 transitions per env are lost on resume).

    def __len__(self) -> int:
        return self._size
