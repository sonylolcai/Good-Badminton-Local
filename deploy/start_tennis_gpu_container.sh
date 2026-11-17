#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
exec "$SCRIPT_DIR/start_sport_gpu_container.sh" "$APP_DIR" "apps.tennis_gpu.app:app" "tennis"
