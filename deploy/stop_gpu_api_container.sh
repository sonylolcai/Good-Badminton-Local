#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PID_FILE="$APP_DIR/.gpu-api.pid"

[[ -f "$PID_FILE" ]] || { echo "No GPU API PID file at $PID_FILE"; exit 0; }
pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped Good-Badminton GPU API (PID $pid)."
else
  echo "Removing stale GPU API PID file ($pid)."
fi
rm -f "$PID_FILE"
