#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_VENV="${TEST_VENV:-$ROOT_DIR/tests/venv}"
PYTHON_BIN="${PYTHON_BIN:-$TEST_VENV/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Test Python executable not found at $PYTHON_BIN"
  echo "Set TEST_VENV or PYTHON_BIN before running this script."
  exit 1
fi

cd "$ROOT_DIR"

"$PYTHON_BIN" -m pip install -e . >/dev/null
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest tests/test_deep_env_algorithms.py -v