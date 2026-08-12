"""ReplayBuffer — off-policy circular buffer for SAC and DDPG.

Supports:
* Uniform random sampling
* n-step returns
* Dict observations
* FP16 storage
"""

from __future__ import annotations

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

    @property
    def obs(self):
        return self.observations

    @property
    def next_obs(self):
        return self.next_observations


class ReplayBuffer:
    """Uniform circular replay buffer with optional n-step returns.

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

        # n-step rolling buffers per env
        if n_step > 1:
            self._nstep_bufs: list[list] = [[] for _ in range(self.n_envs)]

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

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def _lazy_init_from(self, obs: dict, action: np.ndarray, next_obs: dict) -> None:
        self._dict_obs = True
        self._obs = {}
        self._next_obs = {}
        for k, v in obs.items():
            arr = np.asarray(v)
            shape = arr.shape[1:] if arr.ndim > 1 else arr.shape
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
            self._nstep_bufs[env_idx].append(
                (obs, action, reward, next_obs, effective_done, effective_trunc)
            )
            episode_ended = bool(np.asarray(effective_done).any()) or bool(
                np.asarray(effective_trunc).any()
            )
            if len(self._nstep_bufs[env_idx]) == self.n_step or episode_ended:
                obs_n, action_n, reward_n, next_obs_n, done_n, trunc_n = self._compute_nstep(
                    env_idx
                )
                self._write(obs_n, action_n, reward_n, next_obs_n, done_n, trunc_n)
                self._nstep_bufs[env_idx].clear()
        else:
            self._write(obs, action, reward, next_obs, effective_done, effective_trunc)

    def _write(self, obs, action, reward, next_obs, done, truncated=None) -> None:
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

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def _compute_nstep(self, env_idx: int) -> tuple:
        buf = self._nstep_bufs[env_idx]
        obs, action = buf[0][0], buf[0][1]
        next_obs, done, trunc = buf[-1][3], buf[-1][4], buf[-1][5]
        reward = 0.0
        for i, (_, _, r, _, d, tr) in enumerate(buf):
            reward += (self.gamma**i) * r
            if d or tr:
                done, trunc = d, tr
                next_obs = buf[i][3]
                break
        return obs, action, reward, next_obs, done, trunc

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
        }
        if self.n_step > 1:
            state["nstep_bufs"] = self._nstep_bufs
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
        if self.n_step > 1:
            self._nstep_bufs = state.get("nstep_bufs", [[] for _ in range(self.n_envs)])

    def __len__(self) -> int:
        return self._size
