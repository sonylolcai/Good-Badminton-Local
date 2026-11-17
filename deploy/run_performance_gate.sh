#!/usr/bin/env bash
# Evaluate one completed GPU job after a relevant model/pipeline deployment.
# Usage:
#   bash deploy/run_performance_gate.sh /path/to/performance_trace.json [baseline_trace.json] [stream_replay.json]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_PATH="${1:?usage: run_performance_gate.sh PERFORMANCE_TRACE [BASELINE_TRACE] [STREAM_REPLAY]}"
BASELINE_PATH="${2:-}"
STREAM_REPLAY_PATH="${3:-}"
PROFILE_PATH="$REPO_ROOT/evaluation/performance/rtx_3090_production_v1.json"
OUTPUT_PATH="$(dirname "$TRACE_PATH")/performance_gate.json"

COMMAND=(python3 "$REPO_ROOT/evaluation/performance/performance_gate.py" --trace "$TRACE_PATH" --profile "$PROFILE_PATH" --output "$OUTPUT_PATH")
if [[ -n "$BASELINE_PATH" ]]; then
  COMMAND+=(--baseline "$BASELINE_PATH")
fi
if [[ -n "$STREAM_REPLAY_PATH" ]]; then
  COMMAND+=(--stream-replay "$STREAM_REPLAY_PATH")
fi

set +e
"${COMMAND[@]}"
GATE_EXIT_CODE=$?
set -e
echo "Performance gate report: $OUTPUT_PATH"
exit "$GATE_EXIT_CODE"
