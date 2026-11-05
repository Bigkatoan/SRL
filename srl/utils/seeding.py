"""Global seeding / determinism helpers."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all CUDA devices).

    With ``deterministic=True`` additionally force bit-reproducible PyTorch
    kernels: ``torch.use_deterministic_algorithms(True)``,
    ``cudnn.deterministic=True``, ``cudnn.benchmark=False``, TF32 disabled,
    and ``CUBLAS_WORKSPACE_CONFIG`` set (required by cuBLAS for
    deterministic matmuls; must be set before the first CUDA matmul). This
    can be noticeably slower on GPU. On CPU, seeding alone is already enough
    for run-to-run reproducibility of SRL's training loops.
    """
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass


def seed_env_spaces(env: Any, seed: int) -> None:
    """Seed an env's action/observation spaces (warmup ``sample()`` draws).

    Works on SRL wrappers (``act_space``/``single_act_space``/``obs_space``)
    and raw Gymnasium envs (``action_space``/``observation_space``); spaces
    without a ``seed()`` method (isaaclab/mjlab dataclasses) are skipped --
    their warmup draws use NumPy's global RNG, seeded by
    :func:`set_random_seed`.
    """
    seen: set[int] = set()
    for attr in ("act_space", "single_act_space", "action_space", "obs_space", "observation_space"):
        try:
            space = getattr(env, attr, None)
        except Exception:
            space = None
        if space is None or id(space) in seen or not callable(getattr(space, "seed", None)):
            continue
        seen.add(id(space))
        try:
            space.seed(int(seed))
        except Exception:
            pass
