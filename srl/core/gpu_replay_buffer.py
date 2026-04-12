"""GPUReplayBuffer — on-device circular replay buffer for zero-copy training.

Design goals
------------
* All storage pre-allocated as CUDA (or MPS) tensors — no host↔device copies
  during sampling when training on GPU.
* Zero-copy add(): accepts CUDA tensors from Isaac Lab / sim environments and
  writes directly via ``Tensor.copy_(src, non_blocking=True)`` on a dedicated
  CUDA copy stream to overlap with compute.
* Numpy / CPU tensor fallback: ``torch.as_tensor(..., device=self.device)``
  so the buffer is drop-in compatible with environments that return numpy.
* Dict observations: same multi-key interface as :class:`ReplayBuffer`.
* Thread-safe add() via threading.Lock — safe for async runner.
* Compatible output: ``sample()`` returns a :class:`~srl.core.replay_buffer.ReplayBatch`
  whose tensors already live on ``device`` — no ``.to(device)`` needed.
* Checkpoint via ``state_dict()`` / ``load_state_dict()`` — portable across
  devices (saved as CPU tensors, restored on target device).
* ``use_fp16``: store float32 obs/actions as float16 to halve VRAM usage.

Usage::

    buf = GPUReplayBuffer(capacity=100_000, device="cuda:0")
    # in collect loop:
    buf.add(obs, action, reward, done, next_obs)  # obs may be CUDA tensors
    # in train loop:
    batch = buf.sample(256)   # batch.obs already on cuda:0
"""

from __future__ import annotations

import threading
from typing import Union

import torch

from srl.core.replay_buffer import ReplayBatch


class GPUReplayBuffer:
    """Circular replay buffer with all storage on a fixed CUDA/MPS device.

    Parameters
    ----------
    capacity:
        Maximum number of transitions stored.
    device:
        Target device for all tensors (e.g. ``"cuda:0"``).
    n_step:
        n-step return horizon (1 = standard 1-step TD).
    gamma:
        Discount factor for n-step returns.
    use_fp16:
        Store float32 obs/action tensors in float16 to halve VRAM.
    num_envs:
        Number of parallel environments writing to this buffer.
    """

    def __init__(
        self,
        capacity: int,
        device: str | torch.device = "cuda",
        n_step: int = 1,
        gamma: float = 0.99,
        use_fp16: bool = False,
        num_envs: int = 1,
    ) -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.n_step = n_step
        self.gamma = gamma
        self.use_fp16 = use_fp16
        self.num_envs = num_envs
        self._storage_dtype = torch.float16 if use_fp16 else torch.float32

        # Lazily allocated once the first batch of observations arrives
        self._obs_buf: dict[str, torch.Tensor] | torch.Tensor | None = None
        self._next_obs_buf: dict[str, torch.Tensor] | torch.Tensor | None = None
        self._action_buf: torch.Tensor | None = None
        self._reward_buf: torch.Tensor | None = None
        self._done_buf: torch.Tensor | None = None

        self._ptr: int = 0
        self._size: int = 0
        self._lock = threading.Lock()

        # Dedicated CUDA copy stream for non-blocking host→device transfers
        self._copy_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

        # n-step per-env rolling buffers (kept on CPU for simplicity)
        if n_step > 1:
            self._nstep_obs: list[dict[str, list] | list] = [[] for _ in range(num_envs)]
            self._nstep_act: list[list] = [[] for _ in range(num_envs)]
            self._nstep_rew: list[list] = [[] for _ in range(num_envs)]
            self._nstep_done: list[list] = [[] for _ in range(num_envs)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move/cast tensor to buffer device non-blockingly."""
        t = t.to(dtype=self._storage_dtype) if t.is_floating_point() else t
        if t.device == self.device:
            return t
        if self._copy_stream is not None:
            with torch.cuda.stream(self._copy_stream):
                return t.to(device=self.device, non_blocking=True)
        return t.to(device=self.device)

    def _ensure_allocated(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        if self._action_buf is not None:
            return  # already allocated

        act_dim = action.shape[-1] if action.dim() >= 1 else 1
        self._action_buf = torch.zeros(
            (self.capacity, act_dim), dtype=self._storage_dtype, device=self.device
        )
        self._reward_buf = torch.zeros(
            (self.capacity, 1), dtype=torch.float32, device=self.device
        )
        self._done_buf = torch.zeros(
            (self.capacity, 1), dtype=torch.float32, device=self.device
        )

        if isinstance(obs, dict):
            self._obs_buf = {}
            self._next_obs_buf = {}
            for k, v in obs.items():
                shape = v.shape[1:] if v.dim() > 1 else v.shape
                dtype = self._storage_dtype if v.is_floating_point() else v.dtype
                self._obs_buf[k] = torch.zeros((self.capacity, *shape), dtype=dtype, device=self.device)
                self._next_obs_buf[k] = torch.zeros((self.capacity, *shape), dtype=dtype, device=self.device)
        else:
            shape = obs.shape[1:] if obs.dim() > 1 else obs.shape
            dtype = self._storage_dtype if obs.is_floating_point() else obs.dtype
            self._obs_buf = torch.zeros((self.capacity, *shape), dtype=dtype, device=self.device)
            self._next_obs_buf = torch.zeros((self.capacity, *shape), dtype=dtype, device=self.device)

    def _write_obs(
        self,
        buf: dict[str, torch.Tensor] | torch.Tensor,
        idx: int,
        obs: dict[str, torch.Tensor] | torch.Tensor,
    ) -> None:
        if isinstance(buf, dict):
            for k in buf:
                src = self._to_device(obs[k] if isinstance(obs, dict) else obs)
                buf[k][idx].copy_(src.squeeze(0) if src.dim() > buf[k][idx].dim() else src)
        else:
            src = self._to_device(obs if isinstance(obs, torch.Tensor) else next(iter(obs.values())))
            buf[idx].copy_(src.squeeze(0) if src.dim() > buf[idx].dim() else src)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
        reward: float | torch.Tensor,
        done: bool | torch.Tensor,
        next_obs: dict[str, torch.Tensor] | torch.Tensor,
    ) -> None:
        """Add one transition (or a batch from a vectorised env).

        Accepts CUDA tensors directly — no host copy if already on correct device.
        Accepts numpy arrays via implicit ``torch.as_tensor`` conversion.
        Thread-safe.
        """
        # Convert numpy / scalars
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, dtype=torch.float32)
        if not isinstance(done, torch.Tensor):
            done = torch.as_tensor(done, dtype=torch.float32)
        if isinstance(obs, dict):
            obs = {k: torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in obs.items()}
        else:
            obs = torch.as_tensor(obs) if not isinstance(obs, torch.Tensor) else obs
        if isinstance(next_obs, dict):
            next_obs = {k: torch.as_tensor(v) if not isinstance(v, torch.Tensor) else v for k, v in next_obs.items()}
        else:
            next_obs = torch.as_tensor(next_obs) if not isinstance(next_obs, torch.Tensor) else next_obs

        # Handle batched vectorised-env input: iterate per-env row
        batch_size = action.shape[0] if action.dim() > 1 else 1
        if batch_size > 1:
            for i in range(batch_size):
                _obs_i = {k: v[i] for k, v in obs.items()} if isinstance(obs, dict) else obs[i]
                _nobs_i = {k: v[i] for k, v in next_obs.items()} if isinstance(next_obs, dict) else next_obs[i]
                self.add(_obs_i, action[i], reward[i], done[i], _nobs_i)
            return

        with self._lock:
            self._ensure_allocated(obs, action)

            if self.n_step > 1:
                # All envs share one sequential buffer env_idx=0 when called per-sample
                self._nstep_rew[0].append(float(reward))
                self._nstep_done[0].append(float(done))
                self._nstep_act[0].append(action)
                if isinstance(self._nstep_obs[0], list) and len(self._nstep_obs[0]) == 0:
                    self._nstep_obs[0] = []
                self._nstep_obs[0].append((obs, next_obs))

                if len(self._nstep_rew[0]) >= self.n_step:
                    # Compute n-step return
                    ret = 0.0
                    for j in range(self.n_step):
                        ret += (self.gamma ** j) * self._nstep_rew[0][j]
                    _obs0, _ = self._nstep_obs[0][0]
                    _, _nobs_last = self._nstep_obs[0][-1]
                    _act0 = self._nstep_act[0][0]
                    _done_last = self._nstep_done[0][-1]
                    self._nstep_rew[0].pop(0)
                    self._nstep_done[0].pop(0)
                    self._nstep_act[0].pop(0)
                    self._nstep_obs[0].pop(0)
                    self._write_single(_obs0, _act0, ret, _done_last, _nobs_last)
            else:
                self._write_single(obs, action, float(reward), float(done), next_obs)

    def _write_single(self, obs, action, reward, done, next_obs) -> None:
        idx = self._ptr
        self._write_obs(self._obs_buf, idx, obs)
        self._write_obs(self._next_obs_buf, idx, next_obs)
        act = self._to_device(action.float() if action.is_floating_point() else action.to(dtype=self._storage_dtype))
        self._action_buf[idx].copy_(act.view(-1))
        self._reward_buf[idx, 0] = float(reward)
        self._done_buf[idx, 0] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample a random minibatch.  All tensors already on ``self.device``."""
        assert self._size > 0, "Buffer is empty"
        with self._lock:
            indices = torch.randint(0, self._size, (batch_size,), device=self.device)

        if isinstance(self._obs_buf, dict):
            obs_out = {k: v[indices].to(torch.float32) for k, v in self._obs_buf.items()}
            next_obs_out = {k: v[indices].to(torch.float32) for k, v in self._next_obs_buf.items()}
        else:
            obs_out = self._obs_buf[indices].to(torch.float32)
            next_obs_out = self._next_obs_buf[indices].to(torch.float32)

        return ReplayBatch(
            observations=obs_out,
            actions=self._action_buf[indices].to(torch.float32),
            rewards=self._reward_buf[indices],
            next_observations=next_obs_out,
            dones=self._done_buf[indices],
        )

    def __len__(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize buffer to CPU tensors for portable checkpointing."""
        if self._action_buf is None:
            return {"ptr": self._ptr, "size": self._size}

        def _to_cpu(t):
            return t.cpu() if isinstance(t, torch.Tensor) else t

        sd: dict = {
            "ptr": self._ptr,
            "size": self._size,
            "capacity": self.capacity,
            "action_buf": _to_cpu(self._action_buf),
            "reward_buf": _to_cpu(self._reward_buf),
            "done_buf": _to_cpu(self._done_buf),
        }
        if isinstance(self._obs_buf, dict):
            sd["obs_buf"] = {k: _to_cpu(v) for k, v in self._obs_buf.items()}
            sd["next_obs_buf"] = {k: _to_cpu(v) for k, v in self._next_obs_buf.items()}
        else:
            sd["obs_buf"] = _to_cpu(self._obs_buf)
            sd["next_obs_buf"] = _to_cpu(self._next_obs_buf)
        return sd

    def load_state_dict(self, sd: dict) -> None:
        """Restore buffer from a CPU-serialised state dict."""
        self._ptr = int(sd.get("ptr", 0))
        self._size = int(sd.get("size", 0))

        def _to_dev(t):
            return t.to(device=self.device) if isinstance(t, torch.Tensor) else t

        if "action_buf" in sd:
            self._action_buf = _to_dev(sd["action_buf"])
            self._reward_buf = _to_dev(sd["reward_buf"])
            self._done_buf = _to_dev(sd["done_buf"])
            obs_buf = sd.get("obs_buf")
            if isinstance(obs_buf, dict):
                self._obs_buf = {k: _to_dev(v) for k, v in obs_buf.items()}
                self._next_obs_buf = {k: _to_dev(v) for k, v in sd["next_obs_buf"].items()}
            elif obs_buf is not None:
                self._obs_buf = _to_dev(obs_buf)
                self._next_obs_buf = _to_dev(sd.get("next_obs_buf"))
