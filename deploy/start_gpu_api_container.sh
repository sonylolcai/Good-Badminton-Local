#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible command name for existing badminton deployments.  New
# deployments should use start_badminton_gpu_container.sh or the tennis peer.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_badminton_gpu_container.sh" "$@"
