#!/usr/bin/env bash
set -euo pipefail

# One-time, offline-safe TrackNetV3 setup for a Good-Badminton A/B evaluation.
#
# This script intentionally does NOT modify or restart the GPU API.  TrackNet
# stays an isolated candidate model until its raw detections beat the current
# YOLO path on completed human-reviewed labels.
#
# Upload these two official files from a Windows machine before running it:
#   /root/good-badminton-tracknet-upload/TrackNetV3-source.zip
#   /root/good-badminton-tracknet-upload/TrackNetV3_ckpts.zip
#
# Source: https://github.com/qaz812345/TrackNetV3
# Checkpoints: the TrackNetV3_ckpts.zip linked from that repository README.

UPLOAD_DIR="${1:-/root/good-badminton-tracknet-upload}"
STATE_DIR="${GOOD_BADMINTON_STATE_DIR:-/root/good-badminton-gpu-api-state}"
TRACKNET_PYTHON="${TRACKNET_PYTHON_BIN:-python3}"
SOURCE_ARCHIVE="$UPLOAD_DIR/TrackNetV3-source.zip"
CHECKPOINT_ARCHIVE="$UPLOAD_DIR/TrackNetV3_ckpts.zip"
MODEL_ROOT="$STATE_DIR/models/tracknetv3"
SOURCE_DIR="$MODEL_ROOT/source"
CHECKPOINT_DIR="$MODEL_ROOT/ckpts"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ "$UPLOAD_DIR" == /root/good-badminton-tracknet-upload ]] || \
  fail "For safety UPLOAD_DIR must be /root/good-badminton-tracknet-upload."
[[ "$STATE_DIR" == /root/good-badminton-gpu-api-state ]] || \
  fail "For safety STATE_DIR must be /root/good-badminton-gpu-api-state."
command -v unzip >/dev/null || fail "unzip is required"
command -v sha256sum >/dev/null || fail "sha256sum is required"
[[ -s "$SOURCE_ARCHIVE" ]] || fail "Missing source archive: $SOURCE_ARCHIVE"
[[ -s "$CHECKPOINT_ARCHIVE" ]] || fail "Missing checkpoint archive: $CHECKPOINT_ARCHIVE"

# Do not extract archives which can write outside their staging directories.
validate_archive() {
  local archive="$1"
  local member
  while IFS= read -r member; do
    [[ -n "$member" ]] || continue
    [[ "$member" != /* && "$member" != *'..'* ]] || \
      fail "Unsafe path in archive $archive: $member"
  done < <(unzip -Z -1 "$archive")
}

validate_archive "$SOURCE_ARCHIVE"
validate_archive "$CHECKPOINT_ARCHIVE"

stage="$(mktemp -d /tmp/good-badminton-tracknet-setup.XXXXXX)"
cleanup() { rm -rf "$stage"; }
trap cleanup EXIT

echo "[1/4] Validating TrackNetV3 archives..."
unzip -q "$SOURCE_ARCHIVE" -d "$stage/source"
unzip -q "$CHECKPOINT_ARCHIVE" -d "$stage/checkpoints"

candidate_source="$(find "$stage/source" -type f -name predict.py -printf '%h\n' | head -n 1)"
[[ -n "$candidate_source" && -f "$candidate_source/model.py" ]] || \
  fail "TrackNetV3-source.zip does not contain predict.py and model.py"
candidate_tracknet="$(find "$stage/checkpoints" -type f -name TrackNet_best.pt -print -quit)"
candidate_inpaint="$(find "$stage/checkpoints" -type f -name InpaintNet_best.pt -print -quit)"
[[ -n "$candidate_tracknet" ]] || \
  fail "TrackNetV3_ckpts.zip does not contain TrackNet_best.pt"

echo "[2/4] Checking selected Python and CUDA..."
"$TRACKNET_PYTHON" - <<'PY'
import importlib.util
import sys
required = (
    "cv2",
    "numpy",
    "pandas",
    "PIL",
    "torch",
    # The official inference module imports these indirectly through test.py
    # and dataset.py, even though it does not run COCO evaluation.
    "parse",
    "pycocotools",
    "tqdm",
)
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing Python packages: " + ", ".join(missing))
import torch
if not torch.cuda.is_available():
    raise SystemExit("TrackNetV3 needs CUDA but torch.cuda.is_available() is false")
print(
    "TrackNet candidate runtime: "
    f"python={sys.executable}, torch={torch.__version__}, cuda={torch.version.cuda}"
)
PY

echo "[3/4] Installing source and checkpoint files into persistent model storage..."
mkdir -p "$MODEL_ROOT"
rm -rf "$SOURCE_DIR" "$CHECKPOINT_DIR"
mkdir -p "$SOURCE_DIR" "$CHECKPOINT_DIR"
cp -a "$candidate_source/." "$SOURCE_DIR/"
cp -p "$candidate_tracknet" "$CHECKPOINT_DIR/TrackNet_best.pt"
if [[ -n "$candidate_inpaint" ]]; then
  cp -p "$candidate_inpaint" "$CHECKPOINT_DIR/InpaintNet_best.pt"
fi
chmod -R u=rwX,go= "$MODEL_ROOT"

echo "[4/4] Importing official TrackNetV3 inference entry point..."
(
  cd "$SOURCE_DIR"
  "$TRACKNET_PYTHON" predict.py --help >/dev/null
)

manifest="$MODEL_ROOT/manifest.txt"
{
  echo "source_archive_sha256=$(sha256sum "$SOURCE_ARCHIVE" | awk '{print $1}')"
  echo "tracknet_checkpoint_sha256=$(sha256sum "$CHECKPOINT_DIR/TrackNet_best.pt" | awk '{print $1}')"
  if [[ -f "$CHECKPOINT_DIR/InpaintNet_best.pt" ]]; then
    echo "inpaint_checkpoint_sha256=$(sha256sum "$CHECKPOINT_DIR/InpaintNet_best.pt" | awk '{print $1}')"
  fi
  echo "tracknet_python=$TRACKNET_PYTHON"
  echo "installed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$manifest"
chmod 600 "$manifest"

echo "TrackNetV3 setup complete."
echo "Source: $SOURCE_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Manifest: $manifest"
echo "Next: run deploy/run_tracknet_v3_ab.sh with the original video path and its YOLO detections.jsonl path."
