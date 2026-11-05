"""Asynchronous vectorised environment using Python multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import Any

import numpy as np

from srl.envs.vector_utils import FINAL_OBS_KEY


class _CloudpickleWrapper:
    """Pickle env factories (lambdas/closures included) with cloudpickle, so
    they survive the ``spawn``/``forkserver`` start methods, which pickle the
    worker's arguments instead of inheriting them through ``fork``."""

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, state) -> None:
        import pickle

        self.fn = pickle.loads(state)

    def __call__(self):
        return self.fn()


def _worker(conn, env_fn):
    env = env_fn()
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                obs, info = env.reset(**(data or {}))
                conn.send((obs, info))
            elif cmd == "step":
                obs, r, done, trunc, info = env.step(data)
                if done or trunc:
                    # Auto-reset, but keep the TRUE terminal observation --
                    # see srl.envs.vector_utils for the contract.
                    info = dict(info) if isinstance(info, dict) else {}
                    info[FINAL_OBS_KEY] = obs
                    obs, _ = env.reset()
                conn.send((obs, r, done, trunc, info))
            elif cmd == "spaces":
                conn.send((env.obs_space, env.act_space))
            elif cmd == "close":
                env.close()
                conn.close()
                break
    except (EOFError, KeyboardInterrupt):
        env.close()


def _default_start_method() -> str:
    # `fork` is unsafe once the parent has initialised CUDA or started threads
    # (e.g. a torch intra-op pool); prefer forkserver, fall back to spawn.
    methods = mp.get_all_start_methods()
    return "forkserver" if "forkserver" in methods else "spawn"


class AsyncVectorEnv:
    """Run *n* copies of an environment in separate worker processes.

    Preferred for computationally heavy environments (physics simulation).
    Auto-resets ended sub-envs and reports their true terminal observation in
    ``infos[i]["final_obs"]`` exactly like :class:`SyncVectorEnv`.

    Parameters
    ----------
    env_fns:
        List of callables that each return a (wrapped) environment. They are
        serialised with cloudpickle, so lambdas/closures are fine.
    context:
        multiprocessing start method. Default: ``forkserver`` where
        available, else ``spawn`` (both CUDA/thread-safe, unlike ``fork``).
    """

    def __init__(self, env_fns: list[Callable], context: str | None = None) -> None:
        self.num_envs = len(env_fns)
        ctx = mp.get_context(context or _default_start_method())
        self._parent_conns: list[mp.connection.Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False

        for fn in env_fns:
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child_conn, _CloudpickleWrapper(fn)), daemon=True)
            p.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._processes.append(p)

        # Query spaces from worker 0 instead of constructing an extra env in
        # the parent process just to read them.
        self._parent_conns[0].send(("spaces", None))
        self.obs_space, self.act_space = self._parent_conns[0].recv()
        self.single_act_space = self.act_space
        self.single_obs_space = self.obs_space

    def reset(self, **kwargs) -> tuple[dict[str, np.ndarray], list[dict]]:
        for index, conn in enumerate(self._parent_conns):
            conn.send(("reset", _reset_kwargs_for_env(kwargs, index)))
        results = [conn.recv() for conn in self._parent_conns]
        obs_list, info_list = zip(*results, strict=True)
        return _stack_obs(obs_list), list(info_list)

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list]:
        for conn, action in zip(self._parent_conns, actions, strict=True):
            conn.send(("step", action))
        results = [conn.recv() for conn in self._parent_conns]
        obs_list, rews, dones, truncs, infos = zip(*results, strict=True)
        return (
            _stack_obs(obs_list),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncs, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def _stack_obs(obs_list) -> dict[str, np.ndarray]:
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}


def _reset_kwargs_for_env(kwargs: dict[str, Any], index: int) -> dict[str, Any]:
    env_kwargs = dict(kwargs)
    seed = env_kwargs.get("seed")
    if isinstance(seed, int):
        env_kwargs["seed"] = seed + index
    return env_kwargs
