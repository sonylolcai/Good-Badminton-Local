#!/usr/bin/env bash
set -euo pipefail

# Container-safe API launcher. Configure this exact command as the cloud
# instance's startup command so a reboot restores the GPU API without systemd.
APP_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${GOOD_BADMINTON_ENV_FILE:-$APP_DIR/.gpu-api.env}"
PYTHON_BIN="${GOOD_BADMINTON_PYTHON_BIN:-python3}"
PID_FILE="$APP_DIR/.gpu-api.pid"
LOG_FILE="$APP_DIR/gpu-api.log"

[[ -f "$ENV_FILE" ]] || { echo "Missing API environment file: $ENV_FILE" >&2; exit 1; }
[[ -f "$APP_DIR/api/app.py" ]] || { echo "Not a Good-Badminton API directory: $APP_DIR" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PORT="${PORT:-8080}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null; then
    echo "Good-Badminton GPU API is already healthy on port ${PORT}."
    exit 0
  fi
  echo "Stopping stale GPU API process $(cat "$PID_FILE")."
  kill "$(cat "$PID_FILE")" || true
  sleep 1
fi

cd "$APP_DIR"
nohup "$PYTHON_BIN" -m uvicorn api.app:app --host 0.0.0.0 --port "$PORT" \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

for _ in {1..10}; do
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null; then
    echo "Good-Badminton GPU API started (PID $(cat "$PID_FILE"), port ${PORT})."
    exit 0
  fi
  sleep 1
done

echo "GPU API did not become healthy; see $LOG_FILE" >&2
exit 1
