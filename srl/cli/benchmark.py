"""CLI utility to benchmark SRL training throughput across vectorization modes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="srl-benchmark",
        description=(
            "Benchmark SRL training throughput across sync / async / internal "
            "vectorization modes."
        ),
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--env", required=True, help="Gymnasium env id or isaaclab:<task>")
    parser.add_argument("--algo", default=None, help="Algorithm override")
    parser.add_argument(
        "--steps", type=int, default=12_000, help="Training steps per benchmark case"
    )
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel environments to request")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument(
        "--modes", default="sync,async", help="Comma-separated modes: single,sync,async,isaac"
    )
    parser.add_argument("--log-interval", type=int, default=4_000, help="Progress logging interval")
    parser.add_argument("--episode-window", type=int, default=25, help="Rolling episode window")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=0,
        help="Evaluation frequency in counted steps (0 disables eval)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=5, help="Evaluation episodes per eval phase"
    )
    parser.add_argument(
        "--target-file",
        default="",
        help="Optional YAML file with config targets for pass/fail judging",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser


def _parse_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    pattern = re.compile(r"^\s{4}([^:]+):\s+([-+0-9.eE]+)$")
    for line in stdout.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        key = match.group(1).strip()
        try:
            metrics[key] = float(match.group(2))
        except ValueError:
            continue
    return metrics


def _case_command(args, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "srl.cli.train",
        "--config",
        args.config,
        "--env",
        args.env,
        "--steps",
        str(args.steps),
        "--device",
        args.device,
        "--log-interval",
        str(args.log_interval),
        "--episode-window",
        str(args.episode_window),
        "--eval-freq",
        str(args.eval_freq),
        "--eval-episodes",
        str(args.eval_episodes),
        "--console-layout",
        "multi_line",
        "--no-plots",
    ]
    if args.algo:
        cmd.extend(["--algo", args.algo])

    if mode == "single":
        cmd.extend(["--n-envs", "1"])
    elif mode in {"sync", "async"}:
        if args.n_envs is not None:
            cmd.extend(["--n-envs", str(args.n_envs)])
        cmd.extend(["--vec-mode", mode])
    elif mode == "isaac":
        if args.n_envs is not None:
            cmd.extend(["--n-envs", str(args.n_envs)])
    else:
        raise ValueError(f"Unsupported benchmark mode: {mode}")
    return cmd


def _run_case(args, mode: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="srl_benchmark_") as tmpdir:
        logdir = Path(tmpdir) / "runs"
        cmd = _case_command(args, mode) + ["--logdir", str(logdir)]
        start_time = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=Path(args.config).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=900,
        )
        elapsed = time.perf_counter() - start_time
        metrics = _parse_metrics(result.stdout)
        metrics.update(_load_summary_metrics(logdir, args.config, args.algo))
        return {
            "mode": mode,
            "returncode": result.returncode,
            "elapsed_sec": elapsed,
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def _load_summary_metrics(logdir: Path, config_path: str, algo: str | None) -> dict[str, float]:
    resolved_algo = algo or _infer_algo_name(config_path)
    run_name = f"{resolved_algo}_{Path(config_path).stem}"
    summary_path = logdir / run_name / "summary.json"
    if not summary_path.exists():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    last_metrics = summary.get("last_metrics", {}) or {}
    return {
        key: float(value) for key, value in last_metrics.items() if isinstance(value, (int, float))
    }


def _infer_algo_name(config_path: str) -> str:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    algo = data.get("algo")
    if algo:
        return str(algo)
    stem = Path(config_path).stem.lower()
    for candidate in ("td3", "sac", "ddpg", "a3c", "a2c", "ppo"):
        if candidate in stem:
            return candidate
    return "ppo"


def _load_targets(target_file: str) -> dict[str, dict[str, float]]:
    if not target_file:
        return {}
    data = yaml.safe_load(Path(target_file).read_text(encoding="utf-8")) or {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _judge_case(
    case: dict[str, object], targets: dict[str, dict[str, float]], config_path: str
) -> dict[str, object]:
    target = targets.get(Path(config_path).stem)
    if not target:
        return {"status": "no_target"}
    metrics = case.get("metrics", {}) if isinstance(case.get("metrics", {}), dict) else {}
    eval_score = metrics.get("eval/score_mean")
    required_score = target.get("eval_score_min")
    if eval_score is None or required_score is None:
        return {"status": "incomplete_target", "target": target}
    passed = float(eval_score) >= float(required_score)
    return {
        "status": "pass" if passed else "fail",
        "target": target,
        "eval_score": float(eval_score),
    }


def _print_summary(cases: list[dict[str, object]]) -> None:
    print("mode      return  elapsed_s  fps      eval_score  critic_loss  utd_ratio  target")
    for case in cases:
        metrics = case.get("metrics", {})
        fps = metrics.get("fps") if isinstance(metrics, dict) else None
        eval_score = metrics.get("eval/score_mean") if isinstance(metrics, dict) else None
        critic_loss = None
        utd_ratio = None
        judge = case.get("judge", {}) if isinstance(case.get("judge", {}), dict) else {}
        if isinstance(metrics, dict):
            critic_loss = metrics.get("sac/critic_loss", metrics.get("td3/critic_loss"))
            utd_ratio = metrics.get("train/utd_ratio")
        print(
            f"{case['mode']:<9} {case['returncode']!s:<7} {case['elapsed_sec']:<10.2f} "
            f"{_fmt(fps):<8} {_fmt(eval_score):<11} {_fmt(critic_loss):<12} "
            f"{_fmt(utd_ratio):<10} {judge.get('status', '-')}"
        )


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if abs(float(value)) < 1000 else f"{value:.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    targets = _load_targets(args.target_file)
    cases = [_run_case(args, mode) for mode in modes]
    for case in cases:
        case["judge"] = _judge_case(case, targets, args.config)
    _print_summary(cases)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    failed = [case for case in cases if int(case["returncode"]) != 0]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
