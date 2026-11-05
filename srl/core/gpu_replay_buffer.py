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

import torch

from srl.core.replay_buffer import ReplayBatch


class _NStepWindow:
    """Vectorised per-env sliding n-step window living on the buffer device.

    Holds the last ``n`` steps for ``E`` envs that are stepped together
    (one :meth:`push` = one ``env.step()`` of all ``E`` envs) in a ring
    indexed by *age* (0 = newest). Every env keeps its own window length,
    reset to 0 when that env's episode ends, so returns never mix rewards
    across envs or across episode boundaries.

    :meth:`push` returns the transitions that became complete this step:

    * env still running with a full window -> one transition, k = n steps;
    * env whose episode ended (terminated or truncated) -> every remaining
      partial window (k = L, L-1, ..., 1), all ending at this step.

    Everything is tensor ops; the only host sync is one ``nonzero`` per push
    to learn how many transitions to write.
    """

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self._obs = None  # dict[str, (n, E, ...)] or (n, E, ...)
        self._act = None  # (n, E, A)
        self._rew = None  # (n, E)
        self._len = None  # (E,) long
        self._head = 0  # ring slot of the newest step

    def _alloc(self, obs, action, reward) -> None:
        n = self.n

        def _ring(t):
            return torch.zeros((n, *t.shape), dtype=t.dtype, device=t.device)

        self._obs = {k: _ring(v) for k, v in obs.items()} if isinstance(obs, dict) else _ring(obs)
        self._act = _ring(action)
        self._rew = torch.zeros((n, reward.shape[0]), dtype=torch.float32, device=reward.device)
        self._len = torch.zeros(reward.shape[0], dtype=torch.long, device=reward.device)
        self._head = n - 1
        self._gammas = self.gamma ** torch.arange(
            1, n + 1, dtype=torch.float32, device=reward.device
        )

    def push(self, obs, action, reward, done, truncated, next_obs):
        """All inputs batched over envs and already on the buffer device."""
        if self._rew is None:
            self._alloc(obs, action, reward)
        n = self.n
        h = (self._head + 1) % n
        self._head = h
        if isinstance(obs, dict):
            for k, v in obs.items():
                self._obs[k][h] = v
        else:
            self._obs[h] = obs
        self._act[h] = action
        self._rew[h] = reward

        ages = torch.arange(n, device=reward.device)
        slots = (h - ages) % n  # ring slot of each age
        L = torch.clamp(self._len + 1, max=n)  # window length incl. this step

        # R[a] = discounted return of the segment starting at age a and
        # ending now: R[a] = r_age(a) + gamma * R[a-1].
        rew_by_age = self._rew[slots]  # (n, E)
        R = torch.empty_like(rew_by_age)
        acc = torch.zeros_like(rew_by_age[0])
        for a in range(n):
            acc = rew_by_age[a] + self.gamma * acc
            R[a] = acc

        ended = (done > 0.5) | (truncated > 0.5)  # (E,)
        flush = ended.unsqueeze(0) & (ages.unsqueeze(1) < L.unsqueeze(0))
        full = (~ended & (L == n)).unsqueeze(0) & (ages == n - 1).unsqueeze(1)
        mask = flush | full  # (n, E)
        self._len = torch.where(ended, torch.zeros_like(L), L)

        a_idx, e_idx = mask.nonzero(as_tuple=True)
        if a_idx.numel() == 0:
            return None
        s_idx = slots[a_idx]
        if isinstance(obs, dict):
            o = {k: v[s_idx, e_idx] for k, v in self._obs.items()}
            no = {k: v[e_idx] for k, v in next_obs.items()}
        else:
            o = self._obs[s_idx, e_idx]
            no = next_obs[e_idx]
        return (
            o,
            self._act[s_idx, e_idx],
            R[a_idx, e_idx],
            done[e_idx],
            no,
            truncated[e_idx],
            self._gammas[a_idx],
        )

    def state_dict(self) -> dict:
        if self._rew is None:
            return {}

        def _cpu(t):
            return {k: v.cpu() for k, v in t.items()} if isinstance(t, dict) else t.cpu()

        return {
            "obs": _cpu(self._obs),
            "act": self._act.cpu(),
            "rew": self._rew.cpu(),
            "len": self._len.cpu(),
            "head": self._head,
        }

    def load_state_dict(self, sd: dict, device) -> None:
        if not sd:
            return

        def _dev(t):
            return {k: v.to(device) for k, v in t.items()} if isinstance(t, dict) else t.to(device)

        self._obs = _dev(sd["obs"])
        self._act = sd["act"].to(device)
        self._rew = sd["rew"].to(device)
        self._len = sd["len"].to(device)
        self._head = int(sd["head"])
        self._gammas = self.gamma ** torch.arange(1, self.n + 1, dtype=torch.float32, device=device)


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
        # True termination only, kept separate from `truncated` -- see
        # replay_buffer.py's ReplayBatch docstring for why SAC/DDPG/TD3's
        # bootstrap target must not treat a time-limit cutoff as termination.
        self._truncated_buf: torch.Tensor | None = None

        self._ptr: int = 0
        self._size: int = 0
        self._lock = threading.Lock()

        # Dedicated CUDA copy stream for non-blocking host→device transfers
        self._copy_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )

        # Per-transition bootstrap discount gamma**k (see ReplayBatch.discounts).
        self._discount_buf: torch.Tensor | None = None

        # n-step sliding windows on-device. Key ("vec", E) for a batched
        # add() of E envs stepped together, ("env", i) for single-transition
        # adds tagged with env_idx=i.
        self._nstep_windows: dict[tuple, _NStepWindow] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        """Move/cast tensor to buffer device non-blockingly."""
        t = t.to(dtype=self._storage_dtype) if t.is_floating_point() else t
        if t.device == self.device:
            return t
        # Issued on the *current* stream: the previous version copied on a
        # side stream and returned without any cross-stream sync, so the
        # very next kernel on the default stream could read the destination
        # before the copy landed.
        return t.to(device=self.device, non_blocking=True)

    def _ensure_allocated(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        if self._action_buf is not None:
            return  # already allocated

        # Inputs here are always batched (add() promotes single transitions
        # to a batch of one), so the per-row shape is simply shape[1:].
        act_dim = action.shape[-1] if action.dim() >= 1 else 1
        self._action_buf = torch.zeros(
            (self.capacity, act_dim), dtype=self._storage_dtype, device=self.device
        )
        # Shape (capacity,), not (capacity, 1) -- must match ReplayBuffer's
        # own convention (srl/core/replay_buffer.py: `self._rewards =
        # np.zeros((self.capacity,), ...)`). SAC/DDPG/TD3's bootstrap target
        # (e.g. `rewards + gamma * (1 - dones) * next_q`) is written against
        # that (B,) contract; a stray trailing singleton dim on `rewards`/
        # `dones` from this buffer broadcasts against a (B,)-shaped `next_q`
        # into an unintended (B, B) target instead of (B,) -- silently wrong
        # bootstrap math, not a crash (confirmed via a real SAC run: `(1 -
        # dones) * (...)` at (B,1) times (B,) broadcasts outer-product-style
        # to (B,B), then `F.mse_loss(q1, target_q)` warns about the size
        # mismatch and trains against mostly cross-sample noise).
        self._reward_buf = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self._done_buf = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self._truncated_buf = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self._discount_buf = torch.full(
            (self.capacity,), float(self.gamma), dtype=torch.float32, device=self.device
        )

        if isinstance(obs, dict):
            self._obs_buf = {}
            self._next_obs_buf = {}
            for k, v in obs.items():
                shape = v.shape[1:]
                dtype = self._storage_dtype if v.is_floating_point() else v.dtype
                self._obs_buf[k] = torch.zeros(
                    (self.capacity, *shape), dtype=dtype, device=self.device
                )
                self._next_obs_buf[k] = torch.zeros(
                    (self.capacity, *shape), dtype=dtype, device=self.device
                )
        else:
            shape = obs.shape[1:]
            dtype = self._storage_dtype if obs.is_floating_point() else obs.dtype
            self._obs_buf = torch.zeros((self.capacity, *shape), dtype=dtype, device=self.device)
            self._next_obs_buf = torch.zeros(
                (self.capacity, *shape), dtype=dtype, device=self.device
            )

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
        truncated: bool | torch.Tensor | None = None,
        env_idx: int | None = None,
    ) -> None:
        """Add one transition or a ``(num_envs, ...)`` batch from a vector env.

        Accepts CUDA tensors directly (no host copy), CPU tensors, numpy
        arrays or scalars. Thread-safe. A batch is written with one
        scatter per tensor and no per-env Python loop or host sync, for
        both 1-step and n-step (the n-step windows are per env, see
        :class:`_NStepWindow`). Pass a whole batch per ``env.step()`` rather
        than splitting it: with ``n_step > 1`` single-transition adds are
        only correct if each carries its own ``env_idx``.

        `done` must be true termination only, not OR'd with `truncated` -- see
        replay_buffer.py's ReplayBatch docstring.
        """
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, dtype=torch.float32)
        if not isinstance(done, torch.Tensor):
            done = torch.as_tensor(done, dtype=torch.float32)
        if truncated is None:
            truncated = torch.zeros_like(done, dtype=torch.float32)
        elif not isinstance(truncated, torch.Tensor):
            truncated = torch.as_tensor(truncated, dtype=torch.float32)

        def _as_t(o):
            if isinstance(o, dict):
                return {
                    k: v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
                    for k, v in o.items()
                }
            return o if isinstance(o, torch.Tensor) else torch.as_tensor(o)

        obs = _as_t(obs)
        next_obs = _as_t(next_obs)

        if action.dim() <= 1:
            # Single transition -> batch of one.
            action = action.reshape(1, -1)
            obs = self._promote_single(obs, self._obs_buf)
            next_obs = self._promote_single(next_obs, self._next_obs_buf)
        batch_size = action.shape[0]

        with self._lock:
            self._ensure_allocated(obs, action)
            obs_d = self._obs_to_device(obs)
            next_obs_d = self._obs_to_device(next_obs)
            act_d = self._to_device(action.float() if action.is_floating_point() else action)
            act_d = act_d.reshape(batch_size, -1)
            rew_d = self._as_flat(reward, batch_size)
            done_d = self._as_flat(done, batch_size)
            trunc_d = self._as_flat(truncated, batch_size)

            if self.n_step <= 1:
                self._write_batch(obs_d, act_d, rew_d, done_d, next_obs_d, trunc_d, None)
                return

            if env_idx is not None and batch_size == 1:
                key = ("env", int(env_idx))
            else:
                key = ("vec", batch_size)
            window = self._nstep_windows.get(key)
            if window is None:
                window = self._nstep_windows[key] = _NStepWindow(self.n_step, self.gamma)
            out = window.push(obs_d, act_d, rew_d, done_d, trunc_d, next_obs_d)
            if out is not None:
                self._write_batch(*out)

    @staticmethod
    def _promote_single(o, buf):
        def _one(v, b):
            if b is not None:
                return v.unsqueeze(0) if v.dim() == b.dim() - 1 else v
            # Not allocated yet: a leading singleton is taken as the batch dim.
            return v if (v.dim() >= 2 and v.shape[0] == 1) else v.unsqueeze(0)

        if isinstance(o, dict):
            return {k: _one(v, buf.get(k) if isinstance(buf, dict) else None) for k, v in o.items()}
        return _one(o, buf if isinstance(buf, torch.Tensor) else None)

    def _obs_to_device(self, o):
        if isinstance(o, dict):
            return {k: self._to_device(v) for k, v in o.items()}
        return self._to_device(o)

    def _write_batch(
        self,
        obs: dict[str, torch.Tensor] | torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: dict[str, torch.Tensor] | torch.Tensor,
        truncated: torch.Tensor,
        discount: torch.Tensor | None = None,
    ) -> None:
        """Vectorised write for a whole batch of parallel-env transitions.

        Writes all `batch_size` rows with a single fancy-indexed scatter per
        tensor (``buf[idx] = src``) instead of `batch_size` individual
        ``.copy_()`` calls. Index wraparound (when `ptr + batch_size` crosses
        `capacity`) is handled automatically by the modulo below -- fancy
        indexing scatters to arbitrary (non-contiguous) positions correctly,
        no need to special-case the wrap.

        Note: if `batch_size > capacity` (replay buffer smaller than the
        number of parallel envs -- not a sane configuration in practice),
        `idx` contains duplicate slots and which write wins is undefined;
        callers are expected to size the buffer well above `num_envs`.
        """
        batch_size = action.shape[0]
        idx = (self._ptr + torch.arange(batch_size, device=self.device)) % self.capacity

        self._write_obs_batch(self._obs_buf, idx, obs)
        self._write_obs_batch(self._next_obs_buf, idx, next_obs)

        act = self._to_device(action)
        self._action_buf[idx] = act.reshape(batch_size, -1).to(self._action_buf.dtype)
        self._reward_buf[idx] = self._as_flat(reward, batch_size)
        self._done_buf[idx] = self._as_flat(done, batch_size)
        self._truncated_buf[idx] = self._as_flat(truncated, batch_size)
        if discount is None:
            self._discount_buf[idx] = float(self.gamma)
        else:
            self._discount_buf[idx] = self._as_flat(discount, batch_size)

        self._ptr = int((self._ptr + batch_size) % self.capacity)
        self._size = min(self._size + batch_size, self.capacity)

    def _write_obs_batch(
        self,
        buf: dict[str, torch.Tensor] | torch.Tensor,
        idx: torch.Tensor,
        obs: dict[str, torch.Tensor] | torch.Tensor,
    ) -> None:
        if isinstance(buf, dict):
            for k in buf:
                src = self._to_device(obs[k] if isinstance(obs, dict) else obs)
                buf[k][idx] = src
        else:
            src = self._to_device(
                obs if isinstance(obs, torch.Tensor) else next(iter(obs.values()))
            )
            buf[idx] = src

    def _as_flat(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Move *x* to the buffer device as float32 and flatten to (N,).

        Reward/done/truncated storage is always float32 (see
        `_ensure_allocated`) regardless of `use_fp16`, so this intentionally
        does not route through `_to_device`'s storage-dtype cast. Shape
        (N,), not (N, 1) -- see `_ensure_allocated`'s comment on why a
        trailing singleton dim here silently corrupts SAC/DDPG/TD3's
        bootstrap target via broadcasting.
        """
        return x.to(device=self.device, dtype=torch.float32).reshape(batch_size)

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
            truncated=self._truncated_buf[indices],
            discounts=self._discount_buf[indices],
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
            "truncated_buf": _to_cpu(self._truncated_buf),
            "discount_buf": _to_cpu(self._discount_buf),
            "nstep_windows": {k: w.state_dict() for k, w in self._nstep_windows.items()},
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
            # .get() fallback: buffers checkpointed before `truncated` was
            # tracked separately have no such key.
            self._truncated_buf = (
                _to_dev(sd["truncated_buf"])
                if "truncated_buf" in sd
                else torch.zeros_like(self._done_buf)
            )
            self._discount_buf = (
                _to_dev(sd["discount_buf"])
                if "discount_buf" in sd
                else torch.full_like(self._done_buf, float(self.gamma))
            )
            self._nstep_windows = {}
            for key, wsd in (sd.get("nstep_windows") or {}).items():
                w = _NStepWindow(self.n_step, self.gamma)
                w.load_state_dict(wsd, self.device)
                self._nstep_windows[tuple(key)] = w
            obs_buf = sd.get("obs_buf")
            if isinstance(obs_buf, dict):
                self._obs_buf = {k: _to_dev(v) for k, v in obs_buf.items()}
                self._next_obs_buf = {k: _to_dev(v) for k, v in sd["next_obs_buf"].items()}
            elif obs_buf is not None:
                self._obs_buf = _to_dev(obs_buf)
                self._next_obs_buf = _to_dev(sd.get("next_obs_buf"))
