"""Asynchronous vectorised environment using Python multiprocessing."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import Any

import numpy as np


def _worker(conn, env_fn):
    env = env_fn()
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            obs, info = env.reset(**(data or {}))
            conn.send((obs, info))
        elif cmd == "step":
            obs, r, done, trunc, info = env.step(data)
            if done or trunc:
                obs, _ = env.reset()
            conn.send((obs, r, done, trunc, info))
        elif cmd == "close":
            env.close()
            conn.close()
            break


class AsyncVectorEnv:
    """Run *n* copies of an environment in separate worker processes.

    Preferred for computationally heavy environments (physics simulation).

    Parameters
    ----------
    env_fns:
        List of callables that each return a (wrapped) environment.
    """

    def __init__(self, env_fns: list[Callable]) -> None:
        self.num_envs = len(env_fns)
        ctx = mp.get_context("fork")
        self._parent_conns: list[mp.connection.Connection] = []
        self._processes: list[mp.Process] = []

        for fn in env_fns:
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child_conn, fn), daemon=True)
            p.start()
            self._parent_conns.append(parent_conn)
            self._processes.append(p)

        # Probe spaces from first env
        _dummy = env_fns[0]()
        self.obs_space = _dummy.obs_space
        self.act_space = _dummy.act_space
        _dummy.close()

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
        for conn in self._parent_conns:
            conn.send(("close", None))
        for p in self._processes:
            p.join(timeout=5)


def _stack_obs(obs_list) -> dict[str, np.ndarray]:
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}


def _reset_kwargs_for_env(kwargs: dict[str, Any], index: int) -> dict[str, Any]:
    env_kwargs = dict(kwargs)
    seed = env_kwargs.get("seed")
    if isinstance(seed, int):
        env_kwargs["seed"] = seed + index
    return env_kwargs
