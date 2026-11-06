#!/usr/bin/env python
"""Resumable benchmark-matrix runner for SRL (Phase 1: MuJoCo vs SB3 / CleanRL).

Runs every (env, algo, seed) job of a matrix as its own ``srl-train``
subprocess and lays results out as::

    <out>/<env>/<algo>/seed<k>/
        config.yaml      # exact config trained (benchmark config + --set overrides)
        job.json         # job spec + the command line used
        stdout.log       # stdout/stderr of every attempt (appended, with markers)
        attempts.json    # one record per attempt (start/end, exit code)
        train/           # srl-train --logdir  (metrics.jsonl, TensorBoard, history.csv)
        ckpt/            # srl-train --ckptdir (v1 checkpoint directories)
        summary.json     # written ONLY when the job finished successfully

``summary.json`` is the completion marker: a job with a summary whose spec
hash matches the requested spec is skipped, so re-running the same command
continues where a killed run stopped.

Resume policy: incomplete jobs are restarted from step 0. ``srl-train
--resume`` restores weights/optimizers/RNG but NOT the replay buffer (nor
episode bookkeeping), so resuming SAC/TD3 mid-run would train on an empty
buffer -- not comparable with the uninterrupted SB3 reference runs. A clean
restart is the only way to get a credible number.

Robustness:

* each job dir is guarded by an ``flock`` held by the runner AND inherited by
  the child, so two runners (or an orphaned child of a killed runner) can
  never train the same job concurrently;
* children get ``PR_SET_PDEATHSIG=SIGTERM`` (Linux), so ``kill -9`` of the
  runner does not leave orphans training;
* SIGTERM/SIGINT to the runner terminates the children cleanly and exits.

Usage::

    # full Phase 1 matrix
    python scripts/benchmark/run_matrix.py run \\
        --envs HalfCheetah-v5 Hopper-v5 Walker2d-v5 --algos sac td3 ppo \\
        --seeds 0 1 2 --total-steps 1000000 --max-parallel 6 --device cuda

    # quick smoke test; strictly validated configs mean an override that only
    # exists for some algorithms has to be scoped to them
    python scripts/benchmark/run_matrix.py run --envs Hopper-v5 --seeds 0 \\
        --total-steps 3000 --eval-freq 1000 --eval-episodes 1 --device cpu \\
        --set sac,td3:train.learning_starts=500 --out /tmp/bench-smoke

    # progress
    python scripts/benchmark/run_matrix.py status --out benchmarks/runs

Every generated config is validated with ``srl.config.load_experiment`` (in
the child interpreter) before any job starts.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "benchmarks" / "mujoco"
DEFAULT_OUT = REPO_ROOT / "benchmarks" / "runs"
DEFAULT_ENVS = ["HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
DEFAULT_ALGOS = ["sac", "td3", "ppo"]
LAST_N_EVALS = 5
ALGO_COST_RANK = {"sac": 0, "td3": 1, "ddpg": 1, "ppo": 2, "a2c": 2}
LOCK_NAME = ".lock"


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    """Atomic JSON write (tmp + rename) so a kill never leaves half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _git_info() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _child_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    # Run the srl in THIS checkout, never whatever `pip install -e` points at.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(threads)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _runtime_versions(python: str, threads: int) -> dict[str, Any]:
    """Versions as seen by the CHILD interpreter/PYTHONPATH (what actually trains)."""
    code = (
        "import json, sys, platform\n"
        "out = {'python': platform.python_version()}\n"
        "import srl; out['srl_file'] = srl.__file__\n"
        "out['srl'] = getattr(srl, '__version__', None)\n"
        "for name in ('torch', 'gymnasium', 'mujoco', 'numpy'):\n"
        "    try:\n"
        "        m = __import__(name); out[name] = getattr(m, '__version__', None)\n"
        "    except Exception as e:\n"
        "        out[name] = None\n"
        "import torch\n"
        "out['cuda_available'] = torch.cuda.is_available()\n"
        "out['torch_cuda'] = torch.version.cuda\n"
        "out['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
        "print(json.dumps(out))\n"
    )
    res = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, env=_child_env(threads), check=True
    )
    info = json.loads(res.stdout.strip().splitlines()[-1])
    info["platform"] = platform.platform()
    info["cpu_count"] = os.cpu_count()
    return info


def overrides_for(algo: str, overrides: list[str]) -> list[str]:
    """The ``--set`` overrides that apply to *algo*, with any scope prefix removed.

    ``--set train.x=1`` applies to every algorithm; ``--set sac,td3:train.x=1``
    only to the listed ones. Configs are strictly validated, so e.g.
    ``train.learning_starts`` must be scoped away from PPO (it is not a PPO
    hyperparameter and srl-train would refuse the config).
    """
    selected = []
    for ov in overrides:
        head, sep, rest = ov.partition(":")
        if sep and "=" not in head:
            algos = {a.strip().lower() for a in head.split(",") if a.strip()}
            if algo.lower() in algos:
                selected.append(rest)
        else:
            selected.append(ov)
    return selected


def validate_job_configs(python: str, threads: int, jobs: list[Job]) -> None:
    """Strict-validate every job's generated config in the CHILD interpreter.

    srl-train rejects unknown/misplaced keys (e.g. a ``--set`` that does not
    belong to the algorithm); catching that here fails the whole launch in
    seconds instead of burning a retry per job.
    """
    if not jobs:
        return
    code = (
        "import json, sys, warnings\n"
        "from srl.config import load_experiment\n"
        "out = {}\n"
        "for path, algo in json.loads(sys.argv[1]):\n"
        "    try:\n"
        "        with warnings.catch_warnings():\n"
        "            warnings.simplefilter('ignore')\n"
        "            exp = load_experiment(path)\n"
        "        if exp.algo != algo:\n"
        "            out[path] = f'algo: {exp.algo!r}, expected {algo!r}'\n"
        "    except Exception as e:\n"
        "        out[path] = f'{type(e).__name__}: {e}'\n"
        "print(json.dumps(out))\n"
    )
    with tempfile.TemporaryDirectory(prefix="srl_bench_cfg_") as tmp:
        items, names = [], {}
        for i, job in enumerate(jobs):
            path = Path(tmp) / f"{i}.yaml"
            path.write_text(job.config_text, encoding="utf-8")
            items.append([str(path), job.algo])
            names[str(path)] = job.name
        res = subprocess.run(
            [python, "-c", code, json.dumps(items)],
            capture_output=True,
            text=True,
            env=_child_env(threads),
        )
        if res.returncode != 0:
            raise SystemExit(f"config validation failed to run:\n{res.stderr}")
        errors = json.loads(res.stdout.strip().splitlines()[-1])
    if errors:
        lines = "\n".join(f"  {names[p]}: {msg}" for p, msg in sorted(errors.items()))
        raise SystemExit(f"invalid generated config(s):\n{lines}")


def _apply_override(cfg: dict, dotted: str) -> None:
    if "=" not in dotted:
        raise SystemExit(f"--set expects key.path=value, got {dotted!r}")
    key, raw = dotted.split("=", 1)
    value = yaml.safe_load(raw)
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise SystemExit(f"--set {key}: {part!r} is not a mapping")
    node[parts[-1]] = value


def discover_configs(config_dir: Path) -> dict[tuple[str, str], Path]:
    """Map (env_id, algo) -> config path by reading each YAML's env_id/algo."""
    found: dict[tuple[str, str], Path] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        env_id, algo = raw.get("env_id"), raw.get("algo")
        if env_id and algo:
            key = (str(env_id), str(algo).lower())
            if key in found:
                raise SystemExit(f"duplicate config for {key}: {found[key]} and {path}")
            found[key] = path
    return found


def parse_eval_curve(metrics_jsonl: Path) -> list[tuple[int, float]]:
    """(step, eval/score_mean) pairs from an srl-train metrics.jsonl."""
    curve: list[tuple[int, float]] = []
    if not metrics_jsonl.exists():
        return curve
    with metrics_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except ValueError:
                continue  # a torn last line from a killed writer
            if ev.get("tag") == "eval/score_mean":
                curve.append((int(ev["step"]), float(ev["value"])))
    # A step can only be evaluated once per attempt; keep the last if repeated.
    dedup = dict(curve)
    return sorted(dedup.items())


def last_logged_step(metrics_jsonl: Path) -> int:
    """Largest step in metrics.jsonl (cheap: reads only the tail)."""
    if not metrics_jsonl.exists():
        return 0
    with metrics_jsonl.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(size - 8192, 0))
        tail = fh.read().decode("utf-8", errors="replace").splitlines()
    best = 0
    for line in tail:
        with contextlib.suppress(ValueError, KeyError, TypeError):
            best = max(best, int(json.loads(line)["step"]))
    return best


# ──────────────────────────────────────────────────────────────────────────────
# jobs
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Job:
    env: str
    algo: str
    seed: int
    source_config: Path
    run_dir: Path
    total_steps: int
    eval_freq: int
    eval_episodes: int
    overrides: list[str] = field(default_factory=list)

    # filled in at prepare time
    config_text: str = ""

    def build_config_text(self) -> str:
        cfg = yaml.safe_load(self.source_config.read_text(encoding="utf-8"))
        for ov in self.overrides:
            _apply_override(cfg, ov)
        cfg.setdefault("train", {})["total_steps"] = self.total_steps
        cfg["train"]["seed"] = self.seed
        cfg["train"]["eval_freq"] = self.eval_freq
        cfg["train"]["eval_episodes"] = self.eval_episodes
        header = (
            f"# Generated by scripts/benchmark/run_matrix.py from {_rel(self.source_config)}\n"
            f"# overrides: {self.overrides or 'none'}\n"
        )
        return header + yaml.safe_dump(cfg, sort_keys=False)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.config_text.encode("utf-8")).hexdigest()

    @property
    def spec(self) -> dict[str, Any]:
        """Everything that determines the result (device deliberately excluded)."""
        return {
            "env": self.env,
            "algo": self.algo,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "eval_freq": self.eval_freq,
            "eval_episodes": self.eval_episodes,
            "config_hash": self.config_hash,
        }

    @property
    def spec_hash(self) -> str:
        blob = json.dumps(self.spec, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def name(self) -> str:
        return f"{self.env}/{self.algo}/seed{self.seed}"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def metrics_path(self) -> Path:
        return find_metrics_file(self.run_dir)

    def command(self, python: str, device: str) -> list[str]:
        return [
            python,
            "-m",
            "srl.cli.train",
            "--config",
            str(self.run_dir / "config.yaml"),
            "--seed",
            str(self.seed),
            "--steps",
            str(self.total_steps),
            "--n-envs",
            "1",
            "--device",
            device,
            "--eval-freq",
            str(self.eval_freq),
            "--eval-episodes",
            str(self.eval_episodes),
            "--logdir",
            str(self.run_dir / "train"),
            "--ckptdir",
            str(self.run_dir / "ckpt"),
            "--log-interval",
            "10000",
            "--console-layout",
            "single_line",
            "--no-plots",
        ]


def find_metrics_file(run_dir: Path) -> Path:
    hits = sorted((run_dir / "train").glob("*/metrics.jsonl"))
    return hits[0] if hits else run_dir / "train" / "_" / "metrics.jsonl"


def job_state(job: Job) -> str:
    """done | stale (done, different spec) | running | partial | failed | pending."""
    summary = _read_json(job.summary_path)
    if summary is not None:
        return "done" if summary.get("spec_hash") == job.spec_hash else "stale"
    if is_locked(job.run_dir):
        return "running"
    attempts = _read_json(job.run_dir / "attempts.json") or []
    if attempts and _attempt_failed(attempts[-1]):
        return "failed"
    if job.metrics_path.exists() or attempts:
        return "partial"
    return "pending"


def _attempt_failed(attempt: dict[str, Any]) -> bool:
    """Non-zero exit that we did NOT cause by stopping the runner."""
    return attempt.get("exit_code") not in (None, 0) and not attempt.get("interrupted")


def is_locked(run_dir: Path) -> bool:
    lock = run_dir / LOCK_NAME
    if not lock.exists():
        return False
    try:
        fd = os.open(lock, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def build_jobs(args: argparse.Namespace) -> list[Job]:
    config_dir = Path(args.config_dir).resolve()
    configs = discover_configs(config_dir)
    out = Path(args.out).resolve()
    jobs: list[Job] = []
    missing = []
    for env in args.envs:
        for algo in args.algos:
            key = (env, algo.lower())
            if key not in configs:
                missing.append(key)
                continue
            for seed in args.seeds:
                job = Job(
                    env=env,
                    algo=algo.lower(),
                    seed=int(seed),
                    source_config=configs[key],
                    run_dir=out / env / algo.lower() / f"seed{seed}",
                    total_steps=int(args.total_steps),
                    eval_freq=int(args.eval_freq),
                    eval_episodes=int(args.eval_episodes),
                    overrides=overrides_for(algo, list(args.set or [])),
                )
                job.config_text = job.build_config_text()
                jobs.append(job)
    if missing:
        raise SystemExit(f"no config in {config_dir} for: {missing}")
    # Seed-major order: every (env, algo) cell gets its first seed before any
    # cell gets its second, so a partially finished matrix is already a
    # complete (if 1-seed) benchmark, and algorithms with different costs and
    # devices are interleaved instead of queued in blocks. Within a seed the
    # slowest algorithms go first (measured: SAC ~1.6x TD3 ~6x PPO per step),
    # which shortens the tail of the run.
    jobs.sort(key=lambda j: (args.seeds.index(j.seed), ALGO_COST_RANK.get(j.algo, 1)))
    return jobs


# ──────────────────────────────────────────────────────────────────────────────
# running
# ──────────────────────────────────────────────────────────────────────────────


def _set_pdeathsig() -> None:  # runs in the child between fork and exec
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


@dataclass
class Running:
    job: Job
    proc: subprocess.Popen
    lock_fd: int
    log_fh: Any
    started: float
    attempt: dict[str, Any]


class Runner:
    def __init__(self, args: argparse.Namespace, jobs: list[Job]) -> None:
        self.args = args
        self.jobs = jobs
        self.python = args.python or sys.executable
        self.threads = args.threads_per_job or max(
            1, (os.cpu_count() or 1) // max(args.max_parallel, 1)
        )
        self.running: list[Running] = []
        self.stopping = False
        self.failures: dict[str, int] = {}
        self.git = _git_info()
        self.versions: dict[str, Any] = {}
        self.devices = {j.algo: args.device for j in jobs}
        for item in args.device_for or []:
            algo, _, dev = item.partition("=")
            if dev not in ("cuda", "cpu"):
                raise SystemExit(f"--device-for expects ALGO=cuda|cpu, got {item!r}")
            self.devices[algo.lower()] = dev

    def device_for(self, job: Job) -> str:
        return self.devices.get(job.algo, self.args.device)

    # -- lifecycle ---------------------------------------------------------

    def _on_signal(self, signum, _frame) -> None:
        if not self.stopping:
            print(
                f"[run_matrix] {_now()} got signal {signum}; stopping "
                f"{len(self.running)} running job(s)...",
                flush=True,
            )
        self.stopping = True

    def run(self) -> int:
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGHUP, self._on_signal)

        self.versions = _runtime_versions(self.python, self.threads)
        srl_file = Path(self.versions["srl_file"]).resolve()
        if not srl_file.is_relative_to(REPO_ROOT):
            raise SystemExit(
                f"child imports srl from {srl_file}, not from {REPO_ROOT}; refusing to run"
            )
        if "cuda" in self.devices.values() and not self.versions.get("cuda_available"):
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
        print(
            f"[run_matrix] {_now()} repo={REPO_ROOT} git={self.git['sha']}"
            f"{' (dirty)' if self.git['dirty'] else ''} srl={srl_file}",
            flush=True,
        )
        print(
            f"[run_matrix] devices={self.devices} max_parallel={self.args.max_parallel} "
            f"threads/job={self.threads} torch={self.versions.get('torch')} "
            f"gymnasium={self.versions.get('gymnasium')} mujoco={self.versions.get('mujoco')}",
            flush=True,
        )

        queue: list[Job] = []
        for job in self.jobs:
            state = job_state(job)
            if state == "done":
                continue
            if state == "running":
                print(f"[run_matrix] skip {job.name}: locked by another process", flush=True)
                continue
            if state == "stale" and not self.args.rerun_stale:
                print(
                    f"[run_matrix] skip {job.name}: finished with a DIFFERENT spec "
                    f"(pass --rerun-stale to redo it)",
                    flush=True,
                )
                continue
            queue.append(job)
        done = sum(job_state(j) == "done" for j in self.jobs)
        print(f"[run_matrix] {len(self.jobs)} jobs: {done} done, {len(queue)} to run", flush=True)
        validate_job_configs(self.python, self.threads, queue)
        if self.args.dry_run:
            for job in queue:
                print("  " + " ".join(job.command(self.python, self.device_for(job))))
            return 0

        while (queue or self.running) and not self.stopping:
            while queue and len(self.running) < self.args.max_parallel and not self.stopping:
                job = queue.pop(0)
                started = self._start(job)
                if started is None:
                    continue
                self.running.append(started)
            self._reap(queue)
            time.sleep(self.args.poll_interval)

        if self.stopping:
            self._terminate_all()
            print(f"[run_matrix] {_now()} stopped; re-run the same command to continue", flush=True)
            return 130
        n_failed = sum(job_state(j) == "failed" for j in self.jobs)
        print(f"[run_matrix] {_now()} finished; {n_failed} failed job(s)", flush=True)
        return 1 if n_failed else 0

    # -- per job -----------------------------------------------------------

    def _start(self, job: Job) -> Running | None:
        job.run_dir.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(job.run_dir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(lock_fd)
            print(f"[run_matrix] skip {job.name}: locked by another process", flush=True)
            return None

        # Fresh start: drop any previous partial attempt / stale result.
        for sub in ("train", "ckpt"):
            shutil.rmtree(job.run_dir / sub, ignore_errors=True)
        for stale in ("summary.json", "evals.csv"):
            with contextlib.suppress(FileNotFoundError):
                (job.run_dir / stale).unlink()
        (job.run_dir / "config.yaml").write_text(job.config_text, encoding="utf-8")
        cmd = job.command(self.python, self.device_for(job))
        _write_json(
            job.run_dir / "job.json",
            {
                "spec": job.spec,
                "spec_hash": job.spec_hash,
                "source_config": str(job.source_config),
                "overrides": job.overrides,
                "device": self.device_for(job),
                "threads": self.threads,
                "command": cmd,
            },
        )
        attempts = _read_json(job.run_dir / "attempts.json") or []
        attempt = {"n": len(attempts) + 1, "start": _now(), "end": None, "exit_code": None}
        attempts.append(attempt)
        _write_json(job.run_dir / "attempts.json", attempts)

        log_fh = (job.run_dir / "stdout.log").open("a", encoding="utf-8")
        log_fh.write(f"\n===== attempt {attempt['n']} {attempt['start']} =====\n")
        log_fh.write("$ " + " ".join(cmd) + "\n")
        log_fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(job.run_dir),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=_child_env(self.threads),
            pass_fds=(lock_fd,),  # child keeps the job lock even if we die
            preexec_fn=_set_pdeathsig,
        )
        print(f"[run_matrix] {_now()} start {job.name} pid={proc.pid}", flush=True)
        return Running(job, proc, lock_fd, log_fh, time.time(), attempt)

    def _finish_attempt(self, r: Running, code: int, interrupted: bool = False) -> None:
        attempts = _read_json(r.job.run_dir / "attempts.json") or []
        if attempts:
            attempts[-1]["end"] = _now()
            attempts[-1]["exit_code"] = code
            attempts[-1]["interrupted"] = interrupted
            attempts[-1]["wall_sec"] = round(time.time() - r.started, 1)
        _write_json(r.job.run_dir / "attempts.json", attempts)
        r.log_fh.write(f"===== attempt {r.attempt['n']} exit={code} {_now()} =====\n")
        r.log_fh.close()

    def _reap(self, queue: list[Job]) -> None:
        still: list[Running] = []
        for r in self.running:
            code = r.proc.poll()
            if code is None:
                still.append(r)
                continue
            interrupted = code != 0 and self.stopping
            self._finish_attempt(r, code, interrupted=interrupted)
            if interrupted:
                os.close(r.lock_fd)
                continue
            if code == 0:
                try:
                    self._write_summary(r)
                    print(
                        f"[run_matrix] {_now()} done  {r.job.name} "
                        f"({time.time() - r.started:.0f}s)",
                        flush=True,
                    )
                except Exception as exc:  # e.g. no eval recorded
                    print(f"[run_matrix] {r.job.name}: summary failed: {exc}", flush=True)
            else:
                n = self.failures.get(r.job.name, 0) + 1
                self.failures[r.job.name] = n
                print(
                    f"[run_matrix] {_now()} FAIL  {r.job.name} exit={code} "
                    f"(see {r.job.run_dir / 'stdout.log'})",
                    flush=True,
                )
                if n <= self.args.max_retries:
                    queue.append(r.job)
            os.close(r.lock_fd)
        self.running = still

    def _terminate_all(self) -> None:
        for r in self.running:
            with contextlib.suppress(ProcessLookupError):
                r.proc.terminate()
        deadline = time.time() + 30
        for r in self.running:
            try:
                r.proc.wait(timeout=max(deadline - time.time(), 0.1))
            except subprocess.TimeoutExpired:
                r.proc.kill()
                r.proc.wait()
            self._finish_attempt(r, r.proc.returncode, interrupted=True)
            os.close(r.lock_fd)
        self.running = []

    def _write_summary(self, r: Running) -> None:
        job = r.job
        curve = parse_eval_curve(job.metrics_path)
        if not curve:
            raise RuntimeError("no eval/score_mean in metrics.jsonl")
        last = [v for _, v in curve[-LAST_N_EVALS:]]
        wall = time.time() - r.started
        with (job.run_dir / "evals.csv").open("w", encoding="utf-8") as fh:
            fh.write("step,eval_score_mean\n")
            for step, val in curve:
                fh.write(f"{step},{val}\n")
        summary = {
            "env": job.env,
            "algo": job.algo,
            "seed": job.seed,
            "total_steps": job.total_steps,
            "eval_freq": job.eval_freq,
            "eval_episodes": job.eval_episodes,
            "config_hash": job.config_hash,
            "spec_hash": job.spec_hash,
            "source_config": _rel(job.source_config),
            "overrides": job.overrides,
            "device": self.device_for(job),
            "threads_per_job": self.threads,
            "concurrent_jobs_max": self.args.max_parallel,
            "git": self.git,
            "versions": self.versions,
            "started": r.attempt["start"],
            "finished": _now(),
            "wall_clock_sec": round(wall, 1),
            "steps_per_sec": round(job.total_steps / wall, 2) if wall > 0 else None,
            "attempts": r.attempt["n"],
            "final_metric": {
                "name": f"mean of last {len(last)} evals of eval/score_mean "
                f"({job.eval_episodes} deterministic episodes each)",
                "value": sum(last) / len(last),
            },
            "best_eval": max(v for _, v in curve),
            "evals": [[s, v] for s, v in curve],
        }
        _write_json(job.summary_path, summary)


# ──────────────────────────────────────────────────────────────────────────────
# status
# ──────────────────────────────────────────────────────────────────────────────


def discover_run_dirs(out: Path) -> list[Path]:
    return sorted(p.parent for p in out.glob("*/*/seed*/job.json"))


def status(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    rows = []
    counts: dict[str, int] = {}
    total_steps_done = 0
    total_steps_all = 0
    for run_dir in discover_run_dirs(out):
        jobinfo = _read_json(run_dir / "job.json") or {}
        spec = jobinfo.get("spec", {})
        summary = _read_json(run_dir / "summary.json")
        attempts = _read_json(run_dir / "attempts.json") or []
        steps = int(spec.get("total_steps", 0))
        if summary is not None and summary.get("spec_hash") == jobinfo.get("spec_hash"):
            state, done_steps = "done", steps
            hours = summary["wall_clock_sec"] / 3600
            extra = f"final={summary['final_metric']['value']:.1f}  {hours:.2f}h"
        else:
            done_steps = last_logged_step(find_metrics_file(run_dir))
            if is_locked(run_dir):
                state = "running"
            elif attempts and _attempt_failed(attempts[-1]):
                state = "failed"
            else:
                state = "partial" if done_steps else "pending"
            curve = parse_eval_curve(find_metrics_file(run_dir)) if state != "pending" else []
            extra = f"last_eval={curve[-1][1]:.1f}" if curve else ""
            if state == "running" and attempts:
                started = _dt.datetime.fromisoformat(attempts[-1]["start"])
                el = (_dt.datetime.now().astimezone() - started).total_seconds()
                if done_steps > 0 and el > 0:
                    rate = done_steps / el
                    eta = (steps - done_steps) / rate / 3600 if rate else float("nan")
                    extra += f"  {rate:.0f} sps  eta {eta:.1f}h"
        counts[state] = counts.get(state, 0) + 1
        total_steps_done += min(done_steps, steps)
        total_steps_all += steps
        rel = run_dir.relative_to(out)
        pct = 100.0 * done_steps / steps if steps else 0.0
        rows.append(f"  {str(rel):<34} {state:<8} {pct:6.1f}%  tries={len(attempts)}  {extra}")
    if not rows:
        print(f"no jobs under {out}")
        return 0
    print(f"jobs under {out}:")
    print("\n".join(rows))
    overall = 100.0 * total_steps_done / total_steps_all if total_steps_all else 0.0
    print(
        "summary: "
        + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        + f"  | overall {overall:.1f}% of {total_steps_all:,} env steps"
    )
    pid_file = out / "runner.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        alive = Path(f"/proc/{pid}").exists() if pid.isdigit() else False
        print(f"runner pid {pid}: {'alive' if alive else 'not running'}")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run (or continue) a matrix")
    r.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    r.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--total-steps", type=int, default=1_000_000)
    r.add_argument("--eval-freq", type=int, default=10_000)
    r.add_argument("--eval-episodes", type=int, default=10)
    r.add_argument("--max-parallel", type=int, default=3)
    r.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    r.add_argument(
        "--device-for",
        action="append",
        metavar="ALGO=DEVICE",
        help="per-algorithm device override, e.g. --device-for ppo=cpu (repeatable)",
    )
    r.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help="OMP/MKL threads per job (default: cpu_count // max_parallel)",
    )
    r.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument(
        "--set",
        action="append",
        metavar="KEY.PATH=VALUE",
        help="override a config value, e.g. --set train.learning_starts=1000; prefix with "
        "ALGO[,ALGO]: to scope it, e.g. --set sac,td3:train.learning_starts=1000 "
        "(repeatable; part of the spec hash)",
    )
    r.add_argument("--max-retries", type=int, default=1, help="retries per failed job per launch")
    r.add_argument(
        "--rerun-stale", action="store_true", help="redo jobs finished with another spec"
    )
    r.add_argument("--python", default=None, help="interpreter for jobs (default: this one)")
    r.add_argument("--poll-interval", type=float, default=2.0)
    r.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("status", help="print progress of a matrix")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "status":
        return status(args)
    jobs = build_jobs(args)
    return Runner(args, jobs).run()


if __name__ == "__main__":
    sys.exit(main())
