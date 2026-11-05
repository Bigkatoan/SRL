#!/usr/bin/env bash
# Launch the Phase 1 MuJoCo benchmark matrix detached from the terminal:
#   3 envs (HalfCheetah/Hopper/Walker2d-v5) x 3 algos (SAC/TD3/PPO) x 3 seeds x 1M steps.
#
# Safe to re-run: finished jobs are skipped and unfinished ones restart (see
# run_matrix.py). Refuses to start while a previous launch is still alive.
#
#   scripts/benchmark/launch_phase1.sh            # start
#   python scripts/benchmark/run_matrix.py status # progress (or: tail -f benchmarks/runs/runner.log)
#   kill -TERM -- -"$(cat benchmarks/runs/runner.pid)"   # stop (whole process group)
#
# Tunables (environment variables):
#   MAX_PARALLEL  concurrent jobs                    (default 6)
#   DEVICE        cuda | cpu                         (default cuda)
#   THREADS       torch/OMP threads per job          (default 2)
#   NICE          CPU nice level of runner + jobs    (default 10; 19 = lowest priority)
#   OUT           output dir                         (default <repo>/benchmarks/runs)
#   PYTHON        interpreter                        (default /home/bigkatoan/SRL/.venv/bin/python)
#   SEEDS         seeds                              (default "0 1 2")
#   STEPS         env steps per run                  (default 1000000)
#   EXTRA_ARGS    extra run_matrix.py arguments
#
# NICE only lowers CPU priority. NVIDIA consumer GPUs have no per-process
# scheduling priority, so on a shared GPU lower MAX_PARALLEL (or use
# DEVICE=cpu) if the owner's GPU job needs headroom; benchmark jobs need only
# ~300-400 MB of GPU memory each.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-2}"
NICE="${NICE:-10}"
OUT="${OUT:-$REPO/benchmarks/runs}"
PYTHON="${PYTHON:-/home/bigkatoan/SRL/.venv/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-1000000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT"
PID_FILE="$OUT/runner.pid"
LOG="$OUT/runner.log"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "runner already running (pid $(cat "$PID_FILE")); status:" >&2
    "$PYTHON" "$REPO/scripts/benchmark/run_matrix.py" status --out "$OUT" >&2 || true
    exit 1
fi

# Fail fast (in the foreground) if the interpreter would import the wrong srl.
SRL_FILE="$(PYTHONPATH="$REPO" "$PYTHON" -c 'import srl; print(srl.__file__)')"
case "$SRL_FILE" in
    "$REPO"/*) ;;
    *) echo "error: $PYTHON imports srl from $SRL_FILE, not $REPO" >&2; exit 1 ;;
esac

# shellcheck disable=SC2086  # SEEDS / EXTRA_ARGS are intentionally word-split
setsid nohup nice -n "$NICE" env PYTHONPATH="$REPO" "$PYTHON" \
    "$REPO/scripts/benchmark/run_matrix.py" run \
    --envs HalfCheetah-v5 Hopper-v5 Walker2d-v5 \
    --algos sac td3 ppo \
    --seeds $SEEDS \
    --total-steps "$STEPS" \
    --eval-freq 10000 --eval-episodes 10 \
    --max-parallel "$MAX_PARALLEL" \
    --threads-per-job "$THREADS" \
    --device "$DEVICE" \
    --out "$OUT" \
    $EXTRA_ARGS \
    >>"$LOG" 2>&1 </dev/null &
PID=$!
echo "$PID" >"$PID_FILE"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
    echo "runner exited immediately; see $LOG" >&2
    tail -n 20 "$LOG" >&2
    exit 1
fi
echo "started runner pid $PID (process group $PID), nice $NICE, device $DEVICE, max-parallel $MAX_PARALLEL"
echo "  log:    $LOG"
echo "  status: $PYTHON $REPO/scripts/benchmark/run_matrix.py status --out $OUT"
echo "  stop:   kill -TERM -- -$PID"
