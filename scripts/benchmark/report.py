#!/usr/bin/env python
"""Aggregate run_matrix.py results into a Markdown report + learning curves.

Reads every ``<runs>/<env>/<algo>/seed<k>/summary.json`` (finished jobs; with
``--include-partial`` also the eval curves of unfinished ones), and writes:

* a Markdown page (default ``docs/source/benchmarks.md``) with, per
  (env, algo): seeds, mean ± std and IQM [95% bootstrap CI] of the final metric
  (mean of the last 5 evaluations), next to the SB3 rl-zoo / CleanRL numbers
  in ``reference.yaml``; plus an aggregate of SB3-zoo-normalised IQM per algo;
* learning-curve PNGs (default ``docs/source/_static/benchmarks/``).

IQM / CIs use ``rliable`` when it is importable, otherwise an equivalent
built-in implementation (IQM = 25% trimmed mean; percentile bootstrap over
runs, stratified per task for the aggregate) -- the page says which.

Usage::

    python scripts/benchmark/report.py --runs-dir benchmarks/runs

    # preview of a live / partial matrix without touching the docs tree
    python scripts/benchmark/report.py --runs-dir benchmarks/runs --include-partial \\
        --out-md /tmp/bench/benchmarks.md --fig-dir /tmp/bench/fig
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_matrix import parse_eval_curve  # noqa: E402

DEFAULT_ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
DEFAULT_ALGOS = ["sac", "td3", "ppo"]

# Validated categorical/neutral tokens (dataviz default palette, light mode).
SRL_COLOR = "#2a78d6"
REF_COLOR = "#52514e"
TEXT = "#0b0b0b"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


# ──────────────────────────────────────────────────────────────────────────────
# statistics
# ──────────────────────────────────────────────────────────────────────────────


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (scipy.stats.trim_mean(x, 0.25), as rliable)."""
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = len(x)
    if n == 0:
        return float("nan")
    cut = int(0.25 * n)
    return float(x[cut : n - cut].mean())


def bootstrap_ci(
    per_task: list[np.ndarray], stat, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap: resample runs with replacement within each task."""
    if not per_task or any(len(t) == 0 for t in per_task):
        return float("nan"), float("nan")
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = [t[rng.integers(0, len(t), len(t))] for t in per_task]
        stats[b] = stat(np.concatenate(sample))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def interval_estimates(
    per_task: list[np.ndarray], n_boot: int, seed: int
) -> tuple[float, float, float, str]:
    """(IQM, CI low, CI high, backend)."""
    try:
        from rliable import library as rly
        from rliable import metrics as rly_metrics

        # rliable wants {name: (num_runs, num_tasks)}; needs equal runs per task.
        n = min(len(t) for t in per_task)
        mat = np.stack([t[:n] for t in per_task], axis=1)
        np.random.seed(seed)
        pt, ci = rly.get_interval_estimates(
            {"x": mat}, lambda s: np.array([rly_metrics.aggregate_iqm(s)]), reps=n_boot
        )
        return float(pt["x"][0]), float(ci["x"][0][0]), float(ci["x"][1][0]), "rliable"
    except ImportError:
        rng = np.random.default_rng(seed)
        lo, hi = bootstrap_ci(per_task, iqm, n_boot, rng)
        return iqm(np.concatenate(per_task)), lo, hi, "built-in"


# ──────────────────────────────────────────────────────────────────────────────
# loading
# ──────────────────────────────────────────────────────────────────────────────


def load_runs(runs_dir: Path, include_partial: bool) -> list[dict[str, Any]]:
    runs = []
    for job_json in sorted(runs_dir.glob("*/*/seed*/job.json")):
        run_dir = job_json.parent
        summary_path = run_dir / "summary.json"
        job = json.loads(job_json.read_text())
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            if s.get("spec_hash") == job.get("spec_hash"):
                s["complete"] = True
                s["run_dir"] = str(run_dir)
                runs.append(s)
                continue
        if include_partial:
            metrics = sorted((run_dir / "train").glob("*/metrics.jsonl"))
            curve = parse_eval_curve(metrics[0]) if metrics else []
            if curve:
                spec = job["spec"]
                runs.append(
                    {
                        "env": spec["env"],
                        "algo": spec["algo"],
                        "seed": spec["seed"],
                        "total_steps": spec["total_steps"],
                        "evals": [list(e) for e in curve],
                        "complete": False,
                        "run_dir": str(run_dir),
                    }
                )
    return runs


# ──────────────────────────────────────────────────────────────────────────────
# plotting
# ──────────────────────────────────────────────────────────────────────────────


def _curves_on_grid(runs: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate each run's eval curve onto a shared step grid."""
    curves = [np.asarray(r["evals"], dtype=float) for r in runs if r.get("evals")]
    if not curves:
        return None
    end = min(c[-1, 0] for c in curves)
    start = max(c[0, 0] for c in curves)
    if end <= start:
        grid = np.array([end])
    else:
        grid = np.linspace(start, end, 100)
    ys = np.stack([np.interp(grid, c[:, 0], c[:, 1]) for c in curves])
    return grid, ys


def plot_env(env: str, algos: list[str], by_key: dict, reference: dict, out_png: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not any(by_key.get((env, a)) for a in algos):
        return False
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": REF_COLOR,
            "ytick.color": REF_COLOR,
            "axes.titlesize": 10,
        }
    )
    fig, axes = plt.subplots(1, len(algos), figsize=(3.6 * len(algos), 3.0), dpi=150)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, algo in zip(axes, algos, strict=True):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        runs = by_key.get((env, algo), [])
        res = _curves_on_grid(runs)
        n = len(runs)
        if res is not None:
            grid, ys = res
            for y in ys:
                ax.plot(grid, y, color=SRL_COLOR, linewidth=0.8, alpha=0.3)
            mean = ys.mean(axis=0)
            if len(ys) > 1:
                ax.fill_between(grid, ys.min(axis=0), ys.max(axis=0), color=SRL_COLOR, alpha=0.12)
            ax.plot(grid, mean, color=SRL_COLOR, linewidth=2.0)
            ax.annotate(
                f"SRL (n={n})",
                xy=(grid[-1], mean[-1]),
                xytext=(-4, 6),
                textcoords="offset points",
                ha="right",
                color=TEXT,
                fontsize=8,
            )
            xmax = grid[-1]
        else:
            ax.text(0.5, 0.5, "no runs yet", transform=ax.transAxes, ha="center", color=REF_COLOR)
            xmax = 1e6
        refs = reference.get(env, {}).get(algo, {})
        for src, style in (("sb3_zoo", "--"), ("cleanrl", ":")):
            ref = refs.get(src)
            if not ref:
                continue
            ax.axhline(ref["mean"], color=REF_COLOR, linestyle=style, linewidth=1.2)
            label = {"sb3_zoo": "SB3 zoo", "cleanrl": "CleanRL"}[src]
            # SB3 label above-left, CleanRL below-right: never collide even
            # when the two reference values are almost equal.
            above = src == "sb3_zoo"
            ax.annotate(
                f"{label} {ref['mean']:.0f}",
                xy=(0.02 if above else 0.98, ref["mean"]),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3 if above else -3),
                textcoords="offset points",
                ha="left" if above else "right",
                va="bottom" if above else "top",
                color=REF_COLOR,
                fontsize=7,
            )
        ax.set_title(f"{env} · {algo.upper()}", color=TEXT)
        ax.set_xlabel("environment steps")
        ax.set_xlim(0, xmax * 1.02)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda v, _p: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"
            )
        )
    axes[0].set_ylabel("eval return (deterministic)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────────────────
# markdown
# ──────────────────────────────────────────────────────────────────────────────


def _steps(s: int) -> str:
    return f"{s / 1e6:g}M" if s >= 1e6 else f"{s / 1e3:g}k"


def _join(values: list, fmt) -> str:
    return "/".join(fmt(v) for v in values)


def _fmt(x: float | None, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:,.{digits}f}"


def build_report(args: argparse.Namespace) -> str:
    runs_dir = Path(args.runs).resolve()
    reference_doc = yaml.safe_load(Path(args.reference).read_text())
    reference = reference_doc["reference"]
    sources = reference_doc["sources"]
    runs = load_runs(runs_dir, args.include_partial)
    envs, algos = args.envs, args.algos

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in runs:
        by_key[(r["env"], r["algo"])].append(r)

    fig_dir = Path(args.fig_dir)
    md_path = Path(args.out_md)
    rel_fig = os.path.relpath(fig_dir.resolve(), md_path.resolve().parent)

    backend = None
    table = [
        "| Env | Algo | Seeds | Steps | SRL mean ± std | SRL IQM [95% CI] "
        "| SB3 zoo | CleanRL | SRL / zoo |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    normalised: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for env in envs:
        for algo in algos:
            complete = [r for r in by_key.get((env, algo), []) if r.get("complete")]
            finals = np.array([r["final_metric"]["value"] for r in complete], dtype=float)
            refs = reference.get(env, {}).get(algo, {})
            zoo = refs.get("sb3_zoo")
            clean = refs.get("cleanrl")
            zoo_s = f"{_fmt(zoo['mean'])} ± {_fmt(zoo['std'])}" if zoo else "–"
            clean_s = f"{_fmt(clean['mean'])} ± {_fmt(clean['std'])}" if clean else "–"
            for ref, cell in ((zoo, "zoo"), (clean, "clean")):
                if ref and not ref.get("verified", False):
                    if cell == "zoo":
                        zoo_s += " (unverified)"
                    else:
                        clean_s += " (unverified)"
            if len(finals):
                std = finals.std(ddof=1) if len(finals) > 1 else float("nan")
                point, lo, hi, backend = interval_estimates([finals], args.n_boot, args.seed)
                steps = sorted({r["total_steps"] for r in complete})
                steps_s = _join(steps, _steps)
                mean_s = f"{_fmt(finals.mean())} ± {_fmt(std)}"
                iqm_s = f"{_fmt(point)} [{_fmt(lo)}, {_fmt(hi)}]"
                ratio = f"{finals.mean() / zoo['mean']:.2f}" if zoo else "–"
                if zoo:
                    normalised[algo][env] = finals / zoo["mean"]
                table.append(
                    f"| {env} | {algo.upper()} | {len(finals)} | {steps_s} | {mean_s} | {iqm_s} "
                    f"| {zoo_s} | {clean_s} | {ratio} |"
                )
            else:
                table.append(
                    f"| {env} | {algo.upper()} | 0 | – | pending | pending "
                    f"| {zoo_s} | {clean_s} | – |"
                )

    agg = [
        "| Algo | Envs | Runs | IQM of SRL / SB3-zoo [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for algo in algos:
        per_task = [normalised[algo][e] for e in envs if e in normalised[algo]]
        if per_task:
            point, lo, hi, backend = interval_estimates(per_task, args.n_boot, args.seed)
            n_runs = sum(len(t) for t in per_task)
            agg.append(
                f"| {algo.upper()} | {len(per_task)} | {n_runs} "
                f"| {point:.2f} [{lo:.2f}, {hi:.2f}] |"
            )
        else:
            agg.append(f"| {algo.upper()} | 0 | 0 | pending |")

    figures = []
    for env in envs:
        png = fig_dir / f"{env.split('-')[0].lower()}.png"
        if plot_env(env, algos, by_key, reference, png):
            figures.append((env, f"{rel_fig}/{png.name}"))

    complete_runs = [r for r in runs if r.get("complete")]
    shas = sorted({(r.get("git") or {}).get("sha") or "?" for r in complete_runs})
    dirty = any((r.get("git") or {}).get("dirty") for r in complete_runs)
    versions = complete_runs[0].get("versions", {}) if complete_runs else {}
    devices = sorted({r.get("device", "?") for r in complete_runs})
    overrides = sorted({o for r in complete_runs for o in r.get("overrides", [])})
    hours = sum(r.get("wall_clock_sec", 0) for r in complete_runs) / 3600
    partial = [r for r in runs if not r.get("complete")]

    # Protocol as actually run (falls back to the Phase 1 protocol when empty).
    steps_set = sorted({r["total_steps"] for r in complete_runs}) or [1_000_000]
    freq_set = sorted({r["eval_freq"] for r in complete_runs}) or [10_000]
    eps_set = sorted({r["eval_episodes"] for r in complete_runs}) or [10]

    lines = [f"# {args.title}", ""]
    if args.note:
        lines += ["```{note}", args.note, "```", ""]
    if not complete_runs:
        lines += [
            "```{note}",
            "No finished benchmark runs yet: SRL columns read *pending*. Launch the",
            "matrix with `scripts/benchmark/launch_phase1.sh`, then regenerate this",
            "page with `python scripts/benchmark/report.py`.",
            "```",
            "",
        ]
    if overrides:
        lines += [
            "```{warning}",
            "These runs used config overrides, so they are NOT the reference protocol: "
            + ", ".join(f"`{o}`" for o in overrides),
            "```",
            "",
        ]
    lines += [
        "SRL's SAC, TD3 and PPO on three Gymnasium MuJoCo tasks, trained with the",
        "[rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) hyperparameters",
        "(configs in `configs/benchmarks/mujoco/`) and compared with the published",
        "Stable-Baselines3 zoo and CleanRL numbers.",
        "",
        "## Protocol",
        "",
        f"- Environments: {', '.join(f'`{e}`' for e in envs)}, a single",
        f"  environment per run (`n_envs: 1`), {_join(steps_set, _steps)} environment steps.",
        f"- Every {_join(freq_set, _steps)} steps: {_join(eps_set, str)} deterministic episodes on",
        "  a separate evaluation environment.",
        "- **Final metric** of a run: mean of its last 5 evaluations.",
        "- Across seeds: mean ± sample std, and the interquartile mean (IQM) with a",
        f"  95% percentile-bootstrap CI ({args.n_boot} resamples, "
        f"{backend or 'rliable if installed, else built-in'} implementation). With",
        "  3 seeds the IQM equals the mean and the CI is wide: read it as a range.",
        "- *SRL / zoo*: SRL mean divided by the SB3 zoo number; the aggregate table",
        "  pools these ratios over environments (stratified bootstrap).",
        "",
        "## Results",
        "",
        *table,
        "",
        "### Aggregate (normalised by SB3 zoo)",
        "",
        *agg,
        "",
    ]
    if figures:
        lines += ["## Learning curves", ""]
        lines += [
            "Thick line: mean over seeds; thin lines: individual seeds; band: min–max.",
            "Horizontal lines: SB3 zoo (dashed) and CleanRL (dotted) final returns.",
            "",
        ]
        for env, path in figures:
            lines += [f"![{env} learning curves]({path})", ""]
    if partial:
        lines += [
            "## Unfinished runs",
            "",
            "Shown in the curves above but excluded from the tables:",
            "",
        ]
        for r in partial:
            last = r["evals"][-1][0] if r.get("evals") else 0
            lines.append(f"- {r['env']} {r['algo'].upper()} seed {r['seed']}: step {last:,}")
        lines.append("")
    lines += [
        "## Reference numbers",
        "",
        "The reference protocols differ from ours, so treat them as context and not as",
        "exact targets:",
        "",
    ]
    for src in sources.values():
        urls = src.get("urls") or {"": src.get("url")}
        links = ", ".join(f"[{k or 'source'}]({u})" for k, u in urls.items())
        lines.append(f"- **{src['label']}** ({links}): {src['protocol'].strip()}")
    lines += [
        "",
        "CleanRL's PPO uses its own default hyperparameters, not the tuned rl-zoo",
        "values our PPO configs use. All values live in",
        "`scripts/benchmark/reference.yaml`, with the date they were checked.",
        "",
        "## Reproducing",
        "",
        "```bash",
        "scripts/benchmark/launch_phase1.sh            # start, detached",
        "python scripts/benchmark/run_matrix.py status # progress",
        "python scripts/benchmark/report.py            # regenerate this page",
        "```",
        "",
    ]
    if complete_runs:
        lines += [
            "Provenance of the runs in this report: git "
            + ", ".join(f"`{s[:10]}`" for s in shas)
            + (" (uncommitted changes)" if dirty else "")
            + f"; device {', '.join(devices)}; torch {versions.get('torch')}, gymnasium "
            f"{versions.get('gymnasium')}, mujoco {versions.get('mujoco')}; "
            f"{len(complete_runs)} runs, {hours:.1f} job-hours of wall clock. "
            f"Generated {_dt.date.today().isoformat()}.",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--runs-dir",
        "--runs",
        dest="runs",
        default=str(REPO_ROOT / "benchmarks" / "runs"),
        help="run_matrix.py --out directory to read (only read, never written)",
    )
    p.add_argument("--reference", default=str(Path(__file__).resolve().parent / "reference.yaml"))
    p.add_argument("--out-md", default=str(REPO_ROOT / "docs" / "source" / "benchmarks.md"))
    p.add_argument(
        "--fig-dir", default=str(REPO_ROOT / "docs" / "source" / "_static" / "benchmarks")
    )
    p.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    p.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    p.add_argument("--title", default="Benchmarks")
    p.add_argument("--note", default="", help="admonition text placed under the title")
    p.add_argument("--include-partial", action="store_true", help="plot unfinished runs too")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    text = build_report(args)
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
