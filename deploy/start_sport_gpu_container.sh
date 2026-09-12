#!/usr/bin/env bash
set -euo pipefail

# Shared container launcher for the two fixed-sport pure GPU services.  The
# wrapper scripts below pass a literal, allow-listed entry point; this script
# never accepts api.app or an arbitrary Python module as a production target.
APP_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENTRYPOINT="${2:?missing fixed GPU entry point}"
SPORT_ID="${3:?missing sport identity}"
ENV_FILE="${GOOD_BADMINTON_ENV_FILE:-$APP_DIR/.gpu-api.env}"
PYTHON_BIN="${GOOD_BADMINTON_PYTHON_BIN:-python3}"
PID_FILE="$APP_DIR/.gpu-api.pid"
LOG_FILE="$APP_DIR/gpu-api.log"

case "${SPORT_ID}:${ENTRYPOINT}" in
  badminton:apps.badminton_gpu.app:app|tennis:apps.tennis_gpu.app:app) ;;
  *)
    echo "Refusing unsupported GPU entry point: ${SPORT_ID}:${ENTRYPOINT}" >&2
    exit 64
    ;;
esac

[[ -f "$ENV_FILE" ]] || { echo "Missing API environment file: $ENV_FILE" >&2; exit 1; }
[[ -f "$APP_DIR/api/gpu_stream_app.py" ]] || {
  echo "Not a pure GPU stream package: $APP_DIR" >&2
  exit 1
}
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
PORT="${PORT:-8080}"

# The profile is selected by the immutable Python entry point, not by this
# variable.  Fail fast on a stale env file so server operators do not believe
# a tennis deployment is using badminton configuration (or vice versa).
if [[ -n "${GOOD_SPORT_VISION_PROFILE:-}" && "${GOOD_SPORT_VISION_PROFILE}" != "$SPORT_ID" ]]; then
  echo "GOOD_SPORT_VISION_PROFILE=${GOOD_SPORT_VISION_PROFILE} conflicts with ${SPORT_ID} launcher" >&2
  exit 64
fi

# Tennis may run in pose-only mode before any ball checkpoint is available.
# A session that explicitly enables YOLO performs its own model-path and label
# check. This lets WebUI operators validate player tracking first, while still
# rejecting an unavailable ball model at the request boundary.

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null; then
    echo "${SPORT_ID} GPU stream API is already healthy on port ${PORT}."
    exit 0
  fi
  echo "Stopping stale GPU stream API process $(cat "$PID_FILE")."
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
    "$1" -m uvicorn "$2" --host 0.0.0.0 --port "$3" &
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
' _ "$PYTHON_BIN" "$ENTRYPOINT" "$PORT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

for _ in {1..10}; do
  if curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/api/v1/health" >/dev/null; then
    echo "${SPORT_ID} GPU stream API started (PID $(cat "$PID_FILE"), port ${PORT})."
    exit 0
  fi
  sleep 1
done

echo "GPU stream API did not become healthy; see $LOG_FILE" >&2
exit 1
