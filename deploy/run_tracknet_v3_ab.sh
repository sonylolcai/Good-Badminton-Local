#!/usr/bin/env bash
set -euo pipefail

# Run TrackNetV3 against the same original video as a completed YOLO analysis.
# It writes only a new A/B output directory.  It never replaces the source
# detections.jsonl, never changes the live GPU API and never promotes inferred
# InpaintNet points to measurements.
#
# Required positional arguments:
#   1. original video retained by the GPU job
#   2. matching, immutable YOLO detections.jsonl
# Optional third argument:
#   completed human annotation directory containing annotations.jsonl and
#   metadata.json; with it, the script also writes the A/B benchmark report.

VIDEO_PATH="${1:?Usage: run_tracknet_v3_ab.sh <original_video> <yolo_detections.jsonl> [completed_annotation_dir]}"
BASELINE_DETECTIONS="${2:?Usage: run_tracknet_v3_ab.sh <original_video> <yolo_detections.jsonl> [completed_annotation_dir]}"
ANNOTATION_DIR="${3:-}"
APP_DIR="${GOOD_BADMINTON_APP_DIR:-/root/good-badminton-gpu-api}"
STATE_DIR="${GOOD_BADMINTON_STATE_DIR:-/root/good-badminton-gpu-api-state}"
APP_PYTHON="${GOOD_BADMINTON_PYTHON_BIN:-python3}"
TRACKNET_PYTHON="${TRACKNET_PYTHON_BIN:-$APP_PYTHON}"
TRACKNET_ROOT="$STATE_DIR/models/tracknetv3/source"
TRACKNET_CHECKPOINT="$STATE_DIR/models/tracknetv3/ckpts/TrackNet_best.pt"
INPAINT_CHECKPOINT="$STATE_DIR/models/tracknetv3/ckpts/InpaintNet_best.pt"
RUN_ROOT="$STATE_DIR/tracknet_ab"
# The adapter itself decodes/preprocesses bounded chunks.  This ceiling is
# independent of video duration and avoids the container OOM that the old
# full-video frame list could trigger.
TRACKNET_BATCH_SIZE="${TRACKNET_BATCH_SIZE:-16}"
TRACKNET_BACKGROUND_SAMPLE_COUNT="${TRACKNET_BACKGROUND_SAMPLE_COUNT:-120}"
TRACKNET_CHUNK_FRAMES="${TRACKNET_CHUNK_FRAMES:-96}"
TRACKNET_RECTIFICATION="${TRACKNET_RECTIFICATION:-0}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$APP_DIR" == /root/good-badminton-gpu-api ]] || fail "For safety APP_DIR must be /root/good-badminton-gpu-api."
[[ "$STATE_DIR" == /root/good-badminton-gpu-api-state ]] || fail "For safety STATE_DIR must be /root/good-badminton-gpu-api-state."
[[ -f "$VIDEO_PATH" ]] || fail "Original video not found: $VIDEO_PATH"
[[ -f "$BASELINE_DETECTIONS" ]] || fail "YOLO detections.jsonl not found: $BASELINE_DETECTIONS"
[[ -f "$APP_DIR/evaluation/shuttle_tracknet_ab/run_tracknet_v3.py" ]] || \
  fail "The shared GPU package does not include TrackNetV3 A/B tools. Use an explicitly approved separate tool package."
[[ -f "$TRACKNET_ROOT/predict.py" ]] || fail "TrackNetV3 source is not installed. Run setup_tracknet_v3_ab.sh first."
[[ -f "$TRACKNET_CHECKPOINT" ]] || fail "TrackNetV3 checkpoint is not installed. Run setup_tracknet_v3_ab.sh first."
[[ "$TRACKNET_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "TRACKNET_BATCH_SIZE must be a positive integer."
[[ "$TRACKNET_BACKGROUND_SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]] || \
  fail "TRACKNET_BACKGROUND_SAMPLE_COUNT must be a positive integer."
[[ "$TRACKNET_CHUNK_FRAMES" =~ ^[1-9][0-9]*$ ]] || \
  fail "TRACKNET_CHUNK_FRAMES must be a positive integer."
[[ "$TRACKNET_RECTIFICATION" == "0" || "$TRACKNET_RECTIFICATION" == "1" ]] || \
  fail "TRACKNET_RECTIFICATION must be 0 (default) or 1."

mkdir -p "$RUN_ROOT"
video_stem="$(basename "${VIDEO_PATH%.*}")"
run_id="${video_stem}_$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$RUN_ROOT/$run_id"

command=(
  "$APP_PYTHON" "$APP_DIR/evaluation/shuttle_tracknet_ab/run_tracknet_v3.py"
  --video "$VIDEO_PATH"
  --tracknet-root "$TRACKNET_ROOT"
  --tracknet-python "$TRACKNET_PYTHON"
  --tracknet-checkpoint "$TRACKNET_CHECKPOINT"
  --fast-predictor "$APP_DIR/evaluation/shuttle_tracknet_ab/fast_predict_tracknet_v3.py"
  --output-dir "$output_dir"
  --batch-size "$TRACKNET_BATCH_SIZE"
  --background-sample-count "$TRACKNET_BACKGROUND_SAMPLE_COUNT"
  --chunk-frames "$TRACKNET_CHUNK_FRAMES"
)
if [[ "$TRACKNET_RECTIFICATION" == "1" && -f "$INPAINT_CHECKPOINT" ]]; then
  command+=(--run-rectified)
  command+=(--inpaint-checkpoint "$INPAINT_CHECKPOINT")
fi

echo "Running TrackNetV3 inference for: $VIDEO_PATH"
"${command[@]}"

raw_csv="$output_dir/tracknet_raw/${video_stem}_ball.csv"
[[ -f "$raw_csv" ]] || fail "TrackNetV3 completed but raw CSV is missing: $raw_csv"
echo "TrackNet raw measurement CSV: $raw_csv"

rectified_csv="$output_dir/tracknet_rectified/${video_stem}_ball.csv"
if [[ -z "$ANNOTATION_DIR" ]]; then
  echo "No completed human annotation directory provided."
  echo "Raw TrackNet output is ready for visual review; no A/B quality claim has been made."
  echo "When annotations are complete, rerun with its directory as the third argument."
  exit 0
fi

annotations="$ANNOTATION_DIR/annotations.jsonl"
annotation_metadata="$ANNOTATION_DIR/metadata.json"
[[ -f "$annotations" && -f "$annotation_metadata" ]] || \
  fail "Annotation directory must contain completed annotations.jsonl and metadata.json: $ANNOTATION_DIR"

benchmark_dir="$output_dir/benchmark"
benchmark_command=(
  "$APP_PYTHON" "$APP_DIR/evaluation/shuttle_tracknet_ab/run_ab_benchmark.py"
  --video "$VIDEO_PATH"
  --baseline-detections "$BASELINE_DETECTIONS"
  --tracknet-raw-csv "$raw_csv"
  --annotations "$annotations"
  --metadata "$annotation_metadata"
  --output-dir "$benchmark_dir"
)
if [[ -f "$rectified_csv" ]]; then
  benchmark_command+=(--tracknet-rectified-csv "$rectified_csv")
fi

"${benchmark_command[@]}"
echo "A/B report: $benchmark_dir/report.json"
echo "Read raw B metrics for replacement; B* is visual-review-only."
