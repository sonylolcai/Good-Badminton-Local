#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${GOOD_BADMINTON_ENV_FILE:-$APP_DIR/.gpu-api.env}"
PYTHON_BIN="${GOOD_BADMINTON_PYTHON_BIN:-python3}"
PID_FILE="$APP_DIR/.gpu-api.pid"
LOG_FILE="$APP_DIR/gpu-api.log"

[[ -f "$ENV_FILE" ]] || { echo "Missing API environment file: $ENV_FILE" >&2; exit 1; }
[[ -f "$APP_DIR/api/app.py" ]] || { echo "Missing shared GPU API: $APP_DIR/api/app.py" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PORT="${PORT:-8080}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null; then
    echo "Shared GPU API is already healthy on port ${PORT}."
    exit 0
  fi
  echo "Stopping stale GPU API process $(cat "$PID_FILE")."
  kill "$(cat "$PID_FILE")" || true
  sleep 1
fi

cd "$APP_DIR"
nohup bash -c '
  child_pid=""
  stop_child() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
      kill "$child_pid" 2>/dev/null || true
      wait "$child_pid" 2>/dev/null || true
    fi
    exit 0
  }
  trap stop_child INT TERM
  while true; do
    "$1" -m uvicorn api.app:app --host 0.0.0.0 --port "$2" &
    child_pid=$!
    wait "$child_pid"
    exit_code=$?
    child_pid=""
    if [[ "$exit_code" -eq 75 ]]; then
      echo "Stream watchdog requested GPU API restart; resuming durable sessions."
      sleep 1
      continue
    fi
    exit "$exit_code"
  done
' _ "$PYTHON_BIN" "$PORT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

for _ in {1..10}; do
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null; then
    echo "Shared GPU API started (PID $(cat "$PID_FILE"), port ${PORT})."
    exit 0
  fi
  sleep 1
done

echo "GPU API did not become healthy; see $LOG_FILE" >&2
exit 1
