#!/usr/bin/env bash

set -uo pipefail

# Every path below can be overridden from the environment (or the matching
# CLI flag) so the script works outside the original dev machine:
#   SRL_ROOT     repo root (default: parent of this script's directory)
#   PYTHON_BIN   interpreter (default: $VIRTUAL_ENV, then <root>/.venv,
#                <root>/venv, then python3 on PATH)  -- or --python
#   OUTPUT_ROOT  output directory (default: <root>/matrix_runs/<label>)
#                                                    -- or --output-dir
ROOT_DIR="${SRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}" \
    "$ROOT_DIR/.venv/bin/python" \
    "$ROOT_DIR/venv/bin/python"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DEVICE="auto"
SEED="0"
STEPS_OVERRIDE=""
N_ENVS_OVERRIDE=""
LOG_INTERVAL="2000"
EVAL_FREQ="5000"
EVAL_EPISODES="10"
LABEL=""
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
TARGET_FILE=""
BUDGET_MODE="convergence"
CONVERGENCE_STEPS_DEFAULT="20000"
SKIP_INSTALL=0
ENABLE_PLOTS=0
CONFIG_PATTERN="*.yaml"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_full_matrix_benchmark.sh [options]

Run every YAML config under configs/envs, preserve per-case artifacts, and
write a master log plus aggregate JSONL summary.

Options:
  --python PATH           Python executable to use. Default: $PYTHON_BIN, else
                          $VIRTUAL_ENV, <repo>/.venv, <repo>/venv, python3.
  --device DEVICE         Device passed to srl-train. Default: auto.
  --seed SEED             Shared seed for all cases. Default: 0.
  --steps N               Override train.total_steps for every case.
  --n-envs N              Override train.n_envs for every case.
  --log-interval N        Training log interval passed to srl-train. Default: 2000.
  --eval-freq N           Evaluation frequency. Default: 5000.
  --eval-episodes N       Evaluation episodes. Default: 10.
  --budget-mode MODE      convergence|full. Default: convergence.
  --convergence-steps N   Default step budget when budget-mode=convergence. Default: 20000.
  --label NAME            Optional run label. Default: timestamped full_matrix_<ts>.
  --output-dir PATH       Explicit output root. Default: matrix_runs/<label>.
  --target-file PATH      YAML threshold file for benchmark judging.
  --config-pattern GLOB   Limit enumeration to matching YAML basenames. Default: *.yaml.
  --skip-install          Skip pip install -e . before the sweep.
  --with-plots            Keep plot export enabled.
  --help                  Show this message.

Exit status is non-zero only when one or more runnable cases fail.
Cases blocked by missing optional runtimes are recorded as blocked, not failed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --steps)
      STEPS_OVERRIDE="$2"
      shift 2
      ;;
    --n-envs)
      N_ENVS_OVERRIDE="$2"
      shift 2
      ;;
    --log-interval)
      LOG_INTERVAL="$2"
      shift 2
      ;;
    --eval-freq)
      EVAL_FREQ="$2"
      shift 2
      ;;
    --eval-episodes)
      EVAL_EPISODES="$2"
      shift 2
      ;;
    --budget-mode)
      BUDGET_MODE="$2"
      shift 2
      ;;
    --convergence-steps)
      CONVERGENCE_STEPS_DEFAULT="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --target-file)
      TARGET_FILE="$2"
      shift 2
      ;;
    --config-pattern)
      CONFIG_PATTERN="$2"
      shift 2
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --with-plots)
      ENABLE_PLOTS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$BUDGET_MODE" != "convergence" && "$BUDGET_MODE" != "full" ]]; then
  echo "Unsupported --budget-mode: $BUDGET_MODE" >&2
  exit 2
fi

RUN_LABEL="${LABEL:-full_matrix_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/matrix_runs/$RUN_LABEL}"
MASTER_LOG="$RUN_ROOT/matrix.log"
SUMMARY_JSONL="$RUN_ROOT/cases.jsonl"
CASES_DIR="$RUN_ROOT/cases"
REPORT_JSON="$RUN_ROOT/report.json"
REPORT_MD="$RUN_ROOT/summary.md"

if [[ -z "$TARGET_FILE" ]]; then
  if [[ "$BUDGET_MODE" == "convergence" ]]; then
    TARGET_FILE="$ROOT_DIR/configs/benchmarks/convergence_targets.yaml"
  else
    TARGET_FILE="$ROOT_DIR/configs/benchmarks/core_targets.yaml"
  fi
fi

mkdir -p "$CASES_DIR"
printf '' > "$SUMMARY_JSONL"

log() {
  printf '%s\n' "$1" | tee -a "$MASTER_LOG"
}

append_case_record() {
  CASE_ROOT="$1" \
  CASE_CONFIG="$2" \
  CASE_ENV_ID="$3" \
  CASE_ENV_TYPE="$4" \
  CASE_ALGO="$5" \
  CASE_STATUS="$6" \
  CASE_REASON="$7" \
  CASE_EXIT_CODE="$8" \
  CASE_ELAPSED_SEC="$9" \
  CASE_COMMAND="${10}" \
  CASE_SUMMARY_PATH="${11}" \
  CASE_TARGET_FILE="$TARGET_FILE" \
  CASE_SEED="$SEED" \
  "$PYTHON_BIN" - "$SUMMARY_JSONL" <<'PY'
import json
import os
import pathlib
import sys

import yaml


def judge_record(record: dict, target_file: str) -> dict:
    if record["status"] == "blocked":
        return {"status": "blocked"}
    if not target_file:
        return {"status": "no_target_file"}

    target_path = pathlib.Path(target_file)
    if not target_path.exists():
        return {"status": "missing_target_file", "target_file": target_file}

    target_data = yaml.safe_load(target_path.read_text(encoding="utf-8")) or {}
    config_key = pathlib.Path(record["config"]).stem
    target = target_data.get(config_key)
    if not isinstance(target, dict):
        return {"status": "no_target"}

    metrics = record.get("last_metrics", {}) or {}
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

summary_path = pathlib.Path(os.environ["CASE_SUMMARY_PATH"])
record = {
    "config": os.environ["CASE_CONFIG"],
    "env_id": os.environ["CASE_ENV_ID"],
    "env_type": os.environ["CASE_ENV_TYPE"],
    "algo": os.environ["CASE_ALGO"],
    "status": os.environ["CASE_STATUS"],
    "reason": os.environ["CASE_REASON"],
    "exit_code": int(os.environ["CASE_EXIT_CODE"]),
    "elapsed_sec": float(os.environ["CASE_ELAPSED_SEC"]),
    "command": os.environ["CASE_COMMAND"],
    "case_root": os.environ["CASE_ROOT"],
    "summary_path": os.environ["CASE_SUMMARY_PATH"],
    "seed": int(os.environ["CASE_SEED"]),
}
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record["last_metrics"] = summary.get("last_metrics", {}) or {}
else:
    record["last_metrics"] = {}
record["judge"] = judge_record(record, os.environ.get("CASE_TARGET_FILE", ""))
with open(sys.argv[1], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
PY
}

probe_optional_runtime() {
  local env_type="$1"
  local output
  case "$env_type" in
    goal)
      output="$($PYTHON_BIN -c 'import gymnasium_robotics; gymnasium_robotics.register_robotics_envs()' 2>&1)" || {
        printf '%s' "$output"
        return 1
      }
      ;;
    racecar)
      output="$($PYTHON_BIN -c 'import racecar_gym' 2>&1)" || {
        printf '%s' "$output"
        return 1
      }
      ;;
    isaaclab)
      output="$($PYTHON_BIN -c 'import isaaclab_tasks; from isaaclab.envs import ManagerBasedRLEnv' 2>&1)" || {
        printf '%s' "$output"
        return 1
      }
      ;;
  esac
  return 0
}

if [[ $SKIP_INSTALL -eq 0 ]]; then
  log "[setup] Installing current repo into $PYTHON_BIN"
  if ! "$PYTHON_BIN" -m pip install -e "$ROOT_DIR" >> "$MASTER_LOG" 2>&1; then
    log "[setup] pip install -e . failed"
    exit 1
  fi
fi

log "[matrix] root=$RUN_ROOT"
log "[matrix] python=$PYTHON_BIN device=$DEVICE seed=$SEED"
log "[matrix] budget_mode=$BUDGET_MODE convergence_steps=$CONVERGENCE_STEPS_DEFAULT"
log "[matrix] config_pattern=$CONFIG_PATTERN"
log "[matrix] target_file=$TARGET_FILE"

passed=0
failed=0
blocked=0
total=0

while IFS=$'\t' read -r config_rel env_id env_type algo cfg_steps; do
  [[ -n "$config_rel" ]] || continue
  total=$((total + 1))

  config_path="$ROOT_DIR/$config_rel"
  config_stem="$(basename "$config_rel" .yaml)"
  case_root="$CASES_DIR/$config_stem"
  stdout_file="$case_root/stdout.log"
  stderr_file="$case_root/stderr.log"
  run_logdir="$case_root/runs"
  ckpt_dir="$case_root/checkpoints"
  summary_path="$run_logdir/${algo}_${config_stem}/summary.json"
  case_steps="$cfg_steps"
  if [[ -n "$STEPS_OVERRIDE" ]]; then
    case_steps="$STEPS_OVERRIDE"
  elif [[ "$BUDGET_MODE" == "convergence" ]]; then
    case_steps="$CONVERGENCE_STEPS_DEFAULT"
  fi

  mkdir -p "$case_root"

  log "[case:$config_stem] env=$env_id type=$env_type algo=$algo steps=${case_steps:-$cfg_steps}"

  if ! block_reason="$(probe_optional_runtime "$env_type")"; then
    blocked=$((blocked + 1))
    log "[case:$config_stem] blocked: $block_reason"
    append_case_record "$case_root" "$config_rel" "$env_id" "$env_type" "$algo" "blocked" "$block_reason" "0" "0" "blocked-before-run" "$summary_path"
    continue
  fi

  cmd=(
    "$PYTHON_BIN" -m srl.cli.train
    --config "$config_path"
    --device "$DEVICE"
    --seed "$SEED"
    --logdir "$run_logdir"
    --ckptdir "$ckpt_dir"
    --log-interval "$LOG_INTERVAL"
    --eval-freq "$EVAL_FREQ"
    --eval-episodes "$EVAL_EPISODES"
    --console-layout multi_line
  )
  if [[ -n "$case_steps" ]]; then
    cmd+=(--steps "$case_steps")
  fi
  if [[ -n "$N_ENVS_OVERRIDE" ]]; then
    cmd+=(--n-envs "$N_ENVS_OVERRIDE")
  fi
  if [[ $ENABLE_PLOTS -eq 0 ]]; then
    cmd+=(--no-plots)
  fi

  printf '' > "$stdout_file"
  printf '' > "$stderr_file"

  command_string="$(printf '%q ' "${cmd[@]}")"
  start_epoch="$(date +%s)"
  if "${cmd[@]}" \
    > >(tee "$stdout_file" | sed "s/^/[case:$config_stem] /" | tee -a "$MASTER_LOG") \
    2> >(tee "$stderr_file" | sed "s/^/[case:$config_stem][err] /" | tee -a "$MASTER_LOG" >&2); then
    exit_code=0
    status="passed"
    reason=""
    passed=$((passed + 1))
  else
    exit_code=$?
    status="failed"
    reason="train command exited with status $exit_code"
    failed=$((failed + 1))
  fi
  end_epoch="$(date +%s)"
  elapsed_sec="$((end_epoch - start_epoch))"

  {
    printf '===== CASE %s =====\n' "$config_stem"
    printf 'config: %s\n' "$config_rel"
    printf 'env: %s\n' "$env_id"
    printf 'env_type: %s\n' "$env_type"
    printf 'algo: %s\n' "$algo"
    printf 'configured_steps: %s\n' "$cfg_steps"
    printf 'status: %s\n' "$status"
    printf 'exit_code: %s\n' "$exit_code"
    printf 'elapsed_sec: %s\n' "$elapsed_sec"
    printf 'command: %s\n' "$command_string"
    printf '%s\n' '--- stdout ---'
    cat "$stdout_file"
    printf '%s\n' '--- stderr ---'
    cat "$stderr_file"
    printf '\n'
  } >> "$MASTER_LOG"

  append_case_record "$case_root" "$config_rel" "$env_id" "$env_type" "$algo" "$status" "$reason" "$exit_code" "$elapsed_sec" "$command_string" "$summary_path"
done < <(
  SRL_ROOT="$ROOT_DIR" SRL_CONFIG_PATTERN="$CONFIG_PATTERN" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
import yaml

root = Path(os.environ['SRL_ROOT'])
pattern = os.environ['SRL_CONFIG_PATTERN']
configs_dir = root / 'configs' / 'envs'
for config_path in sorted(configs_dir.glob(pattern)):
    data = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    train_cfg = data.get('train') or {}
    env_id = data.get('env_id') or train_cfg.get('env_id') or ''
    env_type = str(data.get('env_type') or train_cfg.get('env_type') or 'flat').strip().lower()
    algo = str(data.get('algo') or '').strip().lower()
    if not algo:
        stem = config_path.stem.lower()
        for candidate in ('td3', 'sac', 'ddpg', 'a3c', 'a2c', 'ppo'):
            if candidate in stem:
                algo = candidate
                break
        if not algo:
            algo = 'ppo'
    total_steps = train_cfg.get('total_steps', '')
    print(f"{config_path.relative_to(root)}\t{env_id}\t{env_type}\t{algo}\t{total_steps}")
PY
)

log "[matrix] completed total=$total passed=$passed failed=$failed blocked=$blocked"
log "[matrix] master_log=$MASTER_LOG"
log "[matrix] summary_jsonl=$SUMMARY_JSONL"
report_counts="$($PYTHON_BIN - "$SUMMARY_JSONL" "$REPORT_JSON" "$REPORT_MD" <<'PY'
import json
from pathlib import Path
import sys

summary_path = Path(sys.argv[1])
report_json_path = Path(sys.argv[2])
report_md_path = Path(sys.argv[3])

records = []
if summary_path.exists():
  with summary_path.open('r', encoding='utf-8') as handle:
    for line in handle:
      line = line.strip()
      if line:
        records.append(json.loads(line))

status_counts = {}
judge_counts = {}
for record in records:
  status = record.get('status', 'unknown')
  status_counts[status] = status_counts.get(status, 0) + 1
  judge_status = (record.get('judge') or {}).get('status', 'unknown')
  judge_counts[judge_status] = judge_counts.get(judge_status, 0) + 1

report = {
  'total': len(records),
  'status_counts': status_counts,
  'judge_counts': judge_counts,
  'records': records,
}
report_json_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

lines = [
  '# Matrix Benchmark Summary',
  '',
  f"Total cases: {len(records)}",
  '',
  '## Run Status',
]
for key in sorted(status_counts):
  lines.append(f"- {key}: {status_counts[key]}")
lines.extend(['', '## Benchmark Gate'])
for key in sorted(judge_counts):
  lines.append(f"- {key}: {judge_counts[key]}")
lines.extend(['', '## Cases', '', '| config | status | judge | eval_score | reason |', '|---|---|---|---:|---|'])
for record in records:
  judge = record.get('judge') or {}
  eval_score = judge.get('eval_score')
  eval_score_str = '-' if eval_score is None else f"{float(eval_score):.4f}"
  reason = (record.get('reason') or '').replace('|', '/').replace('\n', ' ')
  lines.append(
    f"| {record.get('config', '-')} | {record.get('status', '-')} | {judge.get('status', '-')} | {eval_score_str} | {reason} |"
  )
report_md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

judge_failures = judge_counts.get('fail', 0) + judge_counts.get('incomplete_target', 0) + judge_counts.get('missing_target_file', 0)
print(f"{judge_failures}\t{judge_counts.get('pass', 0)}")
PY
)"
IFS=$'\t' read -r gate_failures gate_passes <<< "$report_counts"
log "[matrix] report_json=$REPORT_JSON"
log "[matrix] report_md=$REPORT_MD"
log "[matrix] gate_failures=$gate_failures gate_passes=$gate_passes"

if [[ $failed -gt 0 || ${gate_failures:-0} -gt 0 ]]; then
  exit 1
fi
exit 0
