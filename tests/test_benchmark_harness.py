"""Unit tests for scripts/benchmark/{run_matrix,report}.py (no training)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "scripts" / "benchmark"


def _load(name: str):
    if str(BENCH_DIR) not in sys.path:
        sys.path.insert(0, str(BENCH_DIR))
    spec = importlib.util.spec_from_file_location(name, BENCH_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


run_matrix = _load("run_matrix")
report = _load("report")


def test_every_matrix_cell_has_a_config() -> None:
    configs = run_matrix.discover_configs(run_matrix.DEFAULT_CONFIG_DIR)
    for env in run_matrix.DEFAULT_ENVS:
        for algo in run_matrix.DEFAULT_ALGOS:
            assert (env, algo) in configs


def test_apply_override_parses_yaml_values() -> None:
    cfg = {"train": {"learning_starts": 10_000}}
    run_matrix._apply_override(cfg, "train.learning_starts=1000")
    run_matrix._apply_override(cfg, "train.new.flag=true")
    assert cfg["train"]["learning_starts"] == 1000
    assert cfg["train"]["new"]["flag"] is True


def test_overrides_can_be_scoped_per_algorithm() -> None:
    ovs = ["train.eval_episodes=5", "sac,td3:train.learning_starts=1000", "ppo:train.n_steps=64"]
    assert run_matrix.overrides_for("sac", ovs) == [
        "train.eval_episodes=5",
        "train.learning_starts=1000",
    ]
    assert run_matrix.overrides_for("PPO", ovs) == ["train.eval_episodes=5", "train.n_steps=64"]
    # A ':' inside the value is not a scope prefix.
    assert run_matrix.overrides_for("sac", ["env_id=mjlab:Foo"]) == ["env_id=mjlab:Foo"]


@pytest.mark.parametrize("algo", run_matrix.DEFAULT_ALGOS)
def test_generated_configs_pass_strict_validation(algo: str, tmp_path: Path) -> None:
    from srl.config import load_experiment

    configs = run_matrix.discover_configs(run_matrix.DEFAULT_CONFIG_DIR)
    job = run_matrix.Job(
        "Hopper-v5", algo, 0, configs[("Hopper-v5", algo)], tmp_path, 5000, 1000, 2
    )
    path = tmp_path / "config.yaml"
    path.write_text(job.build_config_text())
    exp = load_experiment(path)
    assert exp.algo == algo
    assert exp.trainer.total_steps == 5000 and exp.trainer.eval_episodes == 2


def test_report_accepts_runs_dir(tmp_path: Path) -> None:
    out = tmp_path / "bench.md"
    rc = report.main(
        [
            "--runs-dir",
            str(tmp_path / "empty"),
            "--out-md",
            str(out),
            "--fig-dir",
            str(tmp_path / "fig"),
        ]
    )
    assert rc == 0 and "pending" in out.read_text()


def test_spec_hash_tracks_overrides_and_steps(tmp_path: Path) -> None:
    src = run_matrix.DEFAULT_CONFIG_DIR / "hopper_sac.yaml"

    def job(steps: int, overrides: list[str]):
        j = run_matrix.Job("Hopper-v5", "sac", 0, src, tmp_path, steps, 10_000, 10, overrides)
        j.config_text = j.build_config_text()
        return j

    base = job(1_000_000, [])
    assert base.spec_hash == job(1_000_000, []).spec_hash
    assert base.spec_hash != job(5_000, []).spec_hash
    assert base.spec_hash != job(1_000_000, ["train.learning_starts=1000"]).spec_hash
    assert "total_steps: 1000000" in base.config_text


def test_parse_eval_curve_tolerates_torn_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    lines = [
        json.dumps({"tag": "train/score", "value": 1.0, "step": 5}),
        json.dumps({"tag": "eval/score_mean", "value": 10.0, "step": 10_000}),
        json.dumps({"tag": "eval/score_mean", "value": 20.0, "step": 20_000}),
        '{"tag": "eval/score_mean", "val',  # killed mid-write
    ]
    path.write_text("\n".join(lines))
    assert run_matrix.parse_eval_curve(path) == [(10_000, 10.0), (20_000, 20.0)]
    assert run_matrix.last_logged_step(path) == 20_000


def test_job_state_pending_then_done(tmp_path: Path) -> None:
    src = run_matrix.DEFAULT_CONFIG_DIR / "hopper_ppo.yaml"
    j = run_matrix.Job("Hopper-v5", "ppo", 1, src, tmp_path / "seed1", 5000, 1000, 2)
    j.config_text = j.build_config_text()
    assert run_matrix.job_state(j) == "pending"
    j.run_dir.mkdir()
    (j.run_dir / "attempts.json").write_text(json.dumps([{"exit_code": -15, "interrupted": True}]))
    assert run_matrix.job_state(j) == "partial"
    (j.run_dir / "attempts.json").write_text(json.dumps([{"exit_code": 1}]))
    assert run_matrix.job_state(j) == "failed"
    (j.run_dir / "summary.json").write_text(json.dumps({"spec_hash": "other"}))
    assert run_matrix.job_state(j) == "stale"
    (j.run_dir / "summary.json").write_text(json.dumps({"spec_hash": j.spec_hash}))
    assert run_matrix.job_state(j) == "done"


def test_iqm_matches_trimmed_mean() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 5.0, 6.0, 7.0])
    assert report.iqm(x) == pytest.approx(np.mean([3.0, 4.0, 5.0, 6.0]))
    assert report.iqm(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(0)
    data = [rng.normal(100.0, 10.0, size=20)]
    lo, hi = report.bootstrap_ci(data, report.iqm, 500, np.random.default_rng(1))
    assert lo < report.iqm(data[0]) < hi
