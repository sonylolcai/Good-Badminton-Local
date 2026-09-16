#!/usr/bin/env bash
set -euo pipefail

# Refresh an already-provisioned Good-Badminton GPU API from one uploaded ZIP.
#
# The ZIP is validated and staged before downtime. The most recent application
# directory is retained beside persistent state and restored automatically if
# the candidate cannot start or advertise the shared multi-sport health shape.
#
# The package must be produced by deploy/package_gpu_api.ps1. It contains the
# shared allow-listed multi-sport API in one top-level directory, no virtual environment,
# no model files, no API data and no secrets.  The script also accepts a flat
# archive containing api/app.py for recovery purposes.

DEPLOY_NAME="good-badminton-gpu-api"
DEFAULT_ARCHIVE_PATH="/root/${DEPLOY_NAME}-upload.zip"
DEFAULT_APP_DIR="/root/${DEPLOY_NAME}"

ARCHIVE_PATH="${1:-$DEFAULT_ARCHIVE_PATH}"
APP_DIR="${2:-$DEFAULT_APP_DIR}"
STATE_DIR="${GOOD_BADMINTON_STATE_DIR:-${APP_DIR}-state}"
ENV_FILE="${STATE_DIR}/.gpu-api.env"
DATA_DIR="${STATE_DIR}/api_data"
WEIGHTS_DIR="${STATE_DIR}/weights"
PREVIOUS_APP_DIR="$STATE_DIR/previous-app"
CANDIDATE_APP_DIR="$STATE_DIR/.candidate-app"
FAILED_APP_DIR="$STATE_DIR/failed-app"
LEGACY_APP_DIR="${GOOD_BADMINTON_LEGACY_APP_DIR:-}"
REMOVE_LEGACY_APP="${GOOD_BADMINTON_REMOVE_LEGACY_APP:-0}"

if [[ -n "${GOOD_BADMINTON_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$GOOD_BADMINTON_PYTHON_BIN"
elif [[ -x "$STATE_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$STATE_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

# Do not ever turn a typo or an unset environment variable into a broad rm.
[[ "$APP_DIR" == "$DEFAULT_APP_DIR" ]] || \
  fail "For safety APP_DIR must be exactly $DEFAULT_APP_DIR (got: $APP_DIR)."
[[ "$STATE_DIR" == "${DEFAULT_APP_DIR}-state" ]] || \
  fail "For safety STATE_DIR must be exactly ${DEFAULT_APP_DIR}-state (got: $STATE_DIR)."
[[ "$ARCHIVE_PATH" == "$DEFAULT_ARCHIVE_PATH" ]] || \
  fail "For safety ARCHIVE_PATH must be exactly $DEFAULT_ARCHIVE_PATH (got: $ARCHIVE_PATH)."
if [[ -n "$LEGACY_APP_DIR" ]]; then
  [[ "$LEGACY_APP_DIR" == /root/* && "$LEGACY_APP_DIR" != /root && \
     "$LEGACY_APP_DIR" != "$APP_DIR" && -d "$LEGACY_APP_DIR" ]] || \
    fail "GOOD_BADMINTON_LEGACY_APP_DIR must be a different existing /root subdirectory."
  [[ "$REMOVE_LEGACY_APP" == "0" || "$REMOVE_LEGACY_APP" == "1" ]] || \
    fail "GOOD_BADMINTON_REMOVE_LEGACY_APP must be 0 or 1."
fi

command -v unzip >/dev/null || fail "unzip is required. Install it once in the GPU image."
command -v curl >/dev/null || fail "curl is required for the local health check."
command -v "$PYTHON_BIN" >/dev/null || fail "Python runtime is not executable: $PYTHON_BIN"
[[ -f "$ARCHIVE_PATH" ]] || fail "Upload package first: $ARCHIVE_PATH"
[[ -s "$ARCHIVE_PATH" ]] || fail "Uploaded package is empty: $ARCHIVE_PATH"

# Reject path traversal and absolute paths before extraction.  unzip normally
# protects against these too, but deployment should fail closed rather than
# trust a package uploaded through a browser UI.
while IFS= read -r member; do
  [[ -n "$member" ]] || continue
  [[ "$member" != /* ]] || fail "Archive contains an absolute path: $member"
  [[ "$member" != *'..'* ]] || fail "Archive contains an unsafe path: $member"
done < <(unzip -Z -1 "$ARCHIVE_PATH")

STAGING_DIR="$(mktemp -d /tmp/good-badminton-gpu-refresh.XXXXXX)"
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

echo "[1/7] Validating uploaded package..."
unzip -q "$ARCHIVE_PATH" -d "$STAGING_DIR"

SOURCE_DIR=""
if [[ -f "$STAGING_DIR/$DEPLOY_NAME/api/app.py" ]]; then
  SOURCE_DIR="$STAGING_DIR/$DEPLOY_NAME"
elif [[ -f "$STAGING_DIR/api/app.py" ]]; then
  SOURCE_DIR="$STAGING_DIR"
else
  fail "Package does not contain $DEPLOY_NAME/api/app.py"
fi

[[ -f "$SOURCE_DIR/deploy/start_gpu_api_container.sh" ]] || \
  fail "Package is missing deploy/start_gpu_api_container.sh"
[[ -f "$SOURCE_DIR/deploy/stop_gpu_api_container.sh" ]] || \
  fail "Package is missing deploy/stop_gpu_api_container.sh"
[[ -f "$SOURCE_DIR/api/vision_profiles.py" ]] || \
  fail "Package is missing shared sport profiles"
[[ -f "$SOURCE_DIR/deploy/install_lap.sh" ]] || \
  fail "Package is missing deploy/install_lap.sh"

echo "[2/7] Checking the existing Python/GPU runtime..."
"$PYTHON_BIN" - <<'PY'
import importlib.util
required = ("cv2", "fastapi", "multipart", "torch", "ultralytics", "uvicorn")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing runtime packages: " + ", ".join(missing))
import torch
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA in the selected Python runtime")
print(f"Python GPU runtime ready: torch={torch.__version__}, cuda={torch.version.cuda}")
PY

"$PYTHON_BIN" -m compileall -q "$SOURCE_DIR/api" "$SOURCE_DIR/badminton_analysis"
PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import tempfile
from api.app import create_app

with tempfile.TemporaryDirectory() as data_dir:
    app = create_app(data_dir=data_dir, start_worker=False)
    assert app.title == "Good-Badminton multi-sport GPU API"
PY

echo "[3/7] Ensuring the ByteTrack lap dependency..."
GOOD_BADMINTON_PYTHON_BIN="$PYTHON_BIN" \
GOOD_BADMINTON_WHEELHOUSE="${GOOD_BADMINTON_WHEELHOUSE:-}" \
GOOD_BADMINTON_PIP_INDEX_URL="${GOOD_BADMINTON_PIP_INDEX_URL:-}" \
  bash "$SOURCE_DIR/deploy/install_lap.sh"

echo "[4/7] Preparing persistent state..."
mkdir -p "$STATE_DIR" "$DATA_DIR" "$WEIGHTS_DIR"
chmod 700 "$STATE_DIR" "$DATA_DIR" "$WEIGHTS_DIR"

# The first transition from a legacy application keeps its secret, jobs and
# weights.  Later runs find symlinks here and therefore leave STATE_DIR alone.
migrate_persistent_state() {
  local source_dir="$1"
  [[ -d "$source_dir" ]] || return 0
  if [[ ! -f "$ENV_FILE" && -f "$source_dir/.gpu-api.env" && ! -L "$source_dir/.gpu-api.env" ]]; then
    cp -p "$source_dir/.gpu-api.env" "$ENV_FILE"
    echo "Migrated API environment file from $source_dir."
  fi
  if [[ ! -L "$source_dir/api_data" && -d "$source_dir/api_data" ]] && \
     [[ -z "$(find "$DATA_DIR" -mindepth 1 -print -quit)" ]]; then
    cp -a "$source_dir/api_data/." "$DATA_DIR/"
    echo "Migrated API job data from $source_dir."
  fi
  # The service may have run from one source directory while its API data was
  # configured in another.  This happened in the first cloud deployment, so
  # honour an explicit, safe absolute data path before it is normalised below.
  local configured_data_dir=""
  if [[ -f "$source_dir/.gpu-api.env" ]]; then
    configured_data_dir="$(sed -n 's/^GOOD_BADMINTON_API_DATA_DIR=//p' "$source_dir/.gpu-api.env" | tail -n 1 | tr -d '\r')"
  fi
  if [[ "$configured_data_dir" == /root/* && -d "$configured_data_dir" && \
        "$configured_data_dir" != "$source_dir/api_data" ]] && \
     [[ -z "$(find "$DATA_DIR" -mindepth 1 -print -quit)" ]]; then
    cp -a "$configured_data_dir/." "$DATA_DIR/"
    echo "Migrated API job data from configured path $configured_data_dir."
  fi
  if [[ ! -L "$source_dir/weights" && -d "$source_dir/weights" ]] && \
     [[ -z "$(find "$WEIGHTS_DIR" -mindepth 1 -print -quit)" ]]; then
    cp -a "$source_dir/weights/." "$WEIGHTS_DIR/"
    echo "Migrated model weights from $source_dir."
  fi
}
migrate_persistent_state "$APP_DIR"
if [[ -n "$LEGACY_APP_DIR" ]]; then
  migrate_persistent_state "$LEGACY_APP_DIR"
fi

# A first deployment can create a key, but never print it.  Existing secrets
# remain untouched.  Data and weights are deliberately external to APP_DIR.
if [[ ! -f "$ENV_FILE" ]]; then
  command -v openssl >/dev/null || fail "openssl is required to create the initial API key"
  umask 077
  api_key="$(openssl rand -hex 32)"
  printf 'GOOD_BADMINTON_API_KEY=%s\nGOOD_BADMINTON_API_DATA_DIR=%s\nPORT=8080\n' \
    "$api_key" "$DATA_DIR" > "$ENV_FILE"
  unset api_key
  echo "Created persistent API environment file: $ENV_FILE"
  echo "Add its API key to the business/WebUI server secret configuration before submitting jobs."
fi
chmod 600 "$ENV_FILE"

# Older deployments sometimes kept GOOD_BADMINTON_API_DATA_DIR pointing at a
# versioned source folder.  Keep its API key and PORT, but force the data path
# to the stable state directory before the old source directory is removed.
env_tmp="$(mktemp "${STATE_DIR}/.gpu-api.env.XXXXXX")"
grep -v '^GOOD_BADMINTON_API_DATA_DIR=' "$ENV_FILE" > "$env_tmp" || true
printf 'GOOD_BADMINTON_API_DATA_DIR=%s\n' "$DATA_DIR" >> "$env_tmp"
chmod 600 "$env_tmp"
mv "$env_tmp" "$ENV_FILE"

checkpoint_from_env() {
  local key="$1"
  local fallback="$2"
  local configured
  configured="$(sed -n "s/^${key}=//p" "$ENV_FILE" | tail -n 1 | tr -d '\r')"
  printf '%s' "${configured:-$fallback}"
}

pose_checkpoint="$(checkpoint_from_env GOOD_BADMINTON_STREAM_POSE_MODEL "$WEIGHTS_DIR/yolo11n-pose.pt")"
ball_checkpoint="$(checkpoint_from_env GOOD_BADMINTON_STREAM_BALL_MODEL "$WEIGHTS_DIR/yolo11s-ball.pt")"
[[ -f "$pose_checkpoint" ]] || fail \
  "Missing pose checkpoint: $pose_checkpoint. Upload yolo11n-pose.pt to $WEIGHTS_DIR or correct GOOD_BADMINTON_STREAM_POSE_MODEL."
[[ -f "$ball_checkpoint" ]] || fail \
  "Missing badminton ball checkpoint: $ball_checkpoint. Upload yolo11s-ball.pt to $WEIGHTS_DIR or correct GOOD_BADMINTON_STREAM_BALL_MODEL."
for tennis_key in GOOD_TENNIS_STREAM_BALL_MODEL GOOD_TENNIS_EXPERIMENTAL_BALL_MODEL; do
  tennis_checkpoint="$(sed -n "s/^${tennis_key}=//p" "$ENV_FILE" | tail -n 1 | tr -d '\r')"
  if [[ -n "$tennis_checkpoint" && ! -f "$tennis_checkpoint" ]]; then
    fail "Tennis YOLO checkpoint not found for ${tennis_key}: $tennis_checkpoint"
  fi
done

stop_existing_api() {
  local source_dir="$1"
  [[ -d "$source_dir" ]] || return 0
  if [[ -x "$source_dir/deploy/stop_gpu_api_container.sh" ]]; then
    "$source_dir/deploy/stop_gpu_api_container.sh" "$source_dir" || true
  elif [[ -f "$source_dir/.gpu-api.pid" ]]; then
    local old_pid
    old_pid="$(cat "$source_dir/.gpu-api.pid" 2>/dev/null || true)"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      kill "$old_pid"
    fi
  fi
}
link_persistent_state() {
  local target_dir="$1"
  rm -rf "$target_dir/.gpu-api.env" "$target_dir/api_data" "$target_dir/weights"
  ln -s "$ENV_FILE" "$target_dir/.gpu-api.env"
  ln -s "$DATA_DIR" "$target_dir/api_data"
  ln -s "$WEIGHTS_DIR" "$target_dir/weights"
}

echo "[5/7] Staging the candidate application..."
rm -rf "$CANDIDATE_APP_DIR"
mkdir -p "$CANDIDATE_APP_DIR"
cp -a "$SOURCE_DIR/." "$CANDIDATE_APP_DIR/"
chmod +x "$CANDIDATE_APP_DIR/deploy/start_gpu_api_container.sh" \
  "$CANDIDATE_APP_DIR/deploy/stop_gpu_api_container.sh" \
  "$CANDIDATE_APP_DIR/deploy/install_lap.sh" \
  "$CANDIDATE_APP_DIR/deploy/refresh_gpu_api_from_zip.sh"
link_persistent_state "$CANDIDATE_APP_DIR"

echo "[6/7] Switching application code with rollback retained..."
stop_existing_api "$APP_DIR"
if [[ -n "$LEGACY_APP_DIR" ]]; then
  stop_existing_api "$LEGACY_APP_DIR"
fi
rm -rf "$PREVIOUS_APP_DIR"
if [[ -d "$APP_DIR" ]]; then
  mv "$APP_DIR" "$PREVIOUS_APP_DIR"
fi
mv "$CANDIDATE_APP_DIR" "$APP_DIR"

verify_shared_health() {
  local health_url="$1"
  local payload
  payload="$(curl -fsS --max-time 5 "$health_url")" || return 1
  "$PYTHON_BIN" -c '
import json
import sys
payload = json.load(sys.stdin)
missing = {"badminton", "tennis"}.difference(payload.get("supported_sport_ids") or [])
if payload.get("status") != "ok" or missing:
    raise SystemExit("shared GPU health is missing: " + ", ".join(sorted(missing)))
' <<<"$payload"
}

restore_previous_api() {
  local reason="$1"
  stop_existing_api "$APP_DIR"
  [[ -d "$PREVIOUS_APP_DIR" ]] || \
    fail "$reason; this was a first deployment, so no previous application exists. Candidate retained at $APP_DIR."
  rm -rf "$FAILED_APP_DIR"
  mv "$APP_DIR" "$FAILED_APP_DIR"
  mv "$PREVIOUS_APP_DIR" "$APP_DIR"
  if ! GOOD_BADMINTON_ENV_FILE="$ENV_FILE" GOOD_BADMINTON_PYTHON_BIN="$PYTHON_BIN" \
    "$APP_DIR/deploy/start_gpu_api_container.sh" "$APP_DIR"; then
    fail "$reason; automatic rollback could not restart $APP_DIR. Failed candidate: $FAILED_APP_DIR"
  fi
  fail "$reason; restored the previous API. Failed candidate: $FAILED_APP_DIR"
}

echo "[7/7] Starting and verifying the refreshed API..."
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
health_url="http://127.0.0.1:${PORT:-8080}/api/v1/health"
if ! GOOD_BADMINTON_ENV_FILE="$ENV_FILE" GOOD_BADMINTON_PYTHON_BIN="$PYTHON_BIN" \
  "$APP_DIR/deploy/start_gpu_api_container.sh" "$APP_DIR"; then
  restore_previous_api "Candidate API did not start"
fi
for _ in {1..10}; do
  if verify_shared_health "$health_url"; then
    break
  fi
  sleep 1
done
verify_shared_health "$health_url" || \
  restore_previous_api "Candidate API did not remain shared-multi-sport healthy; see $APP_DIR/gpu-api.log"
if [[ -n "$LEGACY_APP_DIR" && "$REMOVE_LEGACY_APP" == "1" ]]; then
  # Explicit opt-in only: the fresh process has passed its health check, so
  # removing this one legacy code directory cannot remove persistent state.
  rm -rf "$LEGACY_APP_DIR"
  echo "Removed legacy application directory: $LEGACY_APP_DIR"
fi
echo "Deployment complete. Code: $APP_DIR"
echo "Persistent state (preserved on next update): $STATE_DIR"
echo "Cloud startup command: GOOD_BADMINTON_ENV_FILE=$ENV_FILE bash $APP_DIR/deploy/start_gpu_api_container.sh $APP_DIR"
