#!/usr/bin/env bash
set -euo pipefail

# Ensure the ByteTrack assignment dependency is available in the *same*
# Python runtime that launches the GPU API.  This script is deliberately
# idempotent: a healthy server only performs the import check and does not
# download or alter an already-installed package.
PYTHON_BIN="${GOOD_BADMINTON_PYTHON_BIN:-python3}"
WHEELHOUSE="${GOOD_BADMINTON_WHEELHOUSE:-}"
PIP_INDEX_URL="${GOOD_BADMINTON_PIP_INDEX_URL:-}"

[[ -x "$PYTHON_BIN" || "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]] || {
  echo "ERROR: Python runtime not found: $PYTHON_BIN" >&2
  exit 1
}

if "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("lap") is None:
    raise SystemExit(1)

import lap
print(f"ByteTrack dependency ready: lap={getattr(lap, '__version__', 'installed')}, python={sys.executable}")
PY
then
  exit 0
fi

echo "Installing ByteTrack dependency lap into: $PYTHON_BIN"
PIP_ARGS=()
if [[ -n "$WHEELHOUSE" ]]; then
  [[ -d "$WHEELHOUSE" ]] || {
    echo "ERROR: GOOD_BADMINTON_WHEELHOUSE does not exist: $WHEELHOUSE" >&2
    exit 1
  }
  PIP_ARGS=(--no-index --find-links "$WHEELHOUSE")
elif [[ -n "$PIP_INDEX_URL" ]]; then
  PIP_ARGS=(--index-url "$PIP_INDEX_URL")
fi

"$PYTHON_BIN" -m pip install "${PIP_ARGS[@]}" 'lap>=0.5.12'

"$PYTHON_BIN" - <<'PY'
import lap
print(f"ByteTrack dependency installed: lap={getattr(lap, '__version__', 'installed')}")
PY
