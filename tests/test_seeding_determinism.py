"""Regression tests for seeding / reproducibility.

* `_make_env(seed_offset)` must actually seed each sub-env's spaces, and the
  vector env's own action space (warmup) must be seeded -> reproducible warmup.
* `srl.utils.set_random_seed(seed, deterministic=True)` / `--deterministic`
  switch on PyTorch's deterministic mode and turn TF32 off.
* Two identical CPU CLI runs (2k env steps) produce identical metric history.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from test_envlayer_helpers import mlp_config, record_buffer_adds, run_cli

from srl.utils import set_random_seed

_VOLATILE_KEYS = {"time", "train/fps", "train/wall_time_sec", "runner/collect_fps"}


@pytest.fixture
def restore_torch_determinism():
    prev = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    yield
    torch.use_deterministic_algorithms(prev[0])
    torch.backends.cudnn.deterministic = prev[1]
    torch.backends.cudnn.benchmark = prev[2]
    torch.backends.cuda.matmul.allow_tf32 = prev[3]
    torch.backends.cudnn.allow_tf32 = prev[4]
    torch.set_float32_matmul_precision(prev[5])


def test_set_random_seed_reproduces_draws() -> None:
    set_random_seed(123)
    a = (np.random.rand(3), torch.rand(3))
    set_random_seed(123)
    b = (np.random.rand(3), torch.rand(3))
    np.testing.assert_array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_set_random_seed_deterministic_flags(restore_torch_determinism) -> None:
    set_random_seed(0, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_make_env_seed_offset_seeds_each_sub_env(monkeypatch, tmp_path) -> None:
    """Each sub-env's action space is seeded with seed + index (different
    streams per sub-env, identical across runs)."""
    seeds: list[int] = []
    from srl.utils import seeding

    original = seeding.seed_env_spaces

    def _spy(env, seed):
        seeds.append(int(seed))
        return original(env, seed)

    monkeypatch.setattr(seeding, "seed_env_spaces", _spy)
    cfg = mlp_config("sac", total_steps=8)
    run_cli(monkeypatch, tmp_path, cfg, extra=("--seed", "7"))
    # sub-envs 0 and 1 (seed + offset), then the vector env's own space.
    assert seeds[:3] == [7, 8, 7]


def _warmup_actions(monkeypatch, tmp_path, vec_mode: str) -> np.ndarray:
    adds: list[dict] = []
    cfg = mlp_config("sac", total_steps=16, start_steps=16, update_after=1000)
    run_cli(
        monkeypatch,
        tmp_path,
        cfg,
        # "module:id" so spawn/forkserver workers import (and register) it.
        env_id="test_envlayer_helpers:SRLCounterWide-v0",
        extra=("--seed", "3", "--vec-mode", vec_mode),
        on_agent=record_buffer_adds(adds),
    )
    return np.array([np.asarray(k["action"]).reshape(-1) for k in adds])


@pytest.mark.parametrize("vec_mode", ["sync", "async"])
def test_warmup_actions_reproducible(monkeypatch, tmp_path, vec_mode) -> None:
    a = _warmup_actions(monkeypatch, tmp_path / "a", vec_mode)
    b = _warmup_actions(monkeypatch, tmp_path / "b", vec_mode)
    np.testing.assert_array_equal(a, b)


def test_deterministic_cli_flag(monkeypatch, tmp_path, restore_torch_determinism) -> None:
    cfg = mlp_config("sac", total_steps=8)
    run_cli(monkeypatch, tmp_path, cfg, extra=("--deterministic",))
    assert torch.are_deterministic_algorithms_enabled()


def _history(monkeypatch, tmp_path, cfg: dict, env_id: str) -> list[dict]:
    run_cli(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        cfg=cfg,
        env_id=env_id,
        extra=("--seed", "11", "--log-interval", "200"),
    )
    (path,) = list((tmp_path / "runs").rglob("metrics.jsonl"))
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        {k: v for k, v in r.items() if k not in _VOLATILE_KEYS}
        for r in rows
        if r.get("tag") not in _VOLATILE_KEYS
    ]


@pytest.mark.parametrize("algo", ["sac", "ppo"])
def test_two_cpu_runs_same_seed_identical_history(monkeypatch, tmp_path, algo) -> None:
    if algo == "sac":
        cfg = mlp_config(
            "sac",
            obs_dim=3,
            total_steps=2000,
            start_steps=200,
            update_after=200,
            update_every=8,
            batch_size=32,
            buffer_size=5000,
        )
    else:
        cfg = mlp_config("ppo", obs_dim=3, total_steps=2000, n_steps=128, batch_size=64)
    a = _history(monkeypatch, tmp_path / "a", cfg, "Pendulum-v1")
    b = _history(monkeypatch, tmp_path / "b", cfg, "Pendulum-v1")
    assert len(a) > 3
    assert a == b
