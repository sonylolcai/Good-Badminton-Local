#!/usr/bin/env bash
set -euo pipefail

# Refresh an already-provisioned Good-Badminton GPU API from one uploaded ZIP.
#
# This is intentionally a *replacement* deployment: after the ZIP has passed
# validation and the currently running API has stopped, the old application
# directory is removed and recreated.  Persistent state lives next to, not in,
# the application directory, so this replacement never removes API keys,
# queued/completed job records, or model weights.
#
# The explicit third argument selects one of two fixed deployment identities;
# it cannot be supplied by a WebUI request or changed after the server starts.
# Defaults preserve the existing badminton command.
#
# The package must be produced by deploy/package_gpu_api.ps1.  It contains a
# single fixed-sport top-level directory, no virtual environment,
# no model files, no API data and no secrets.  The script also accepts a flat
# archive containing api/app.py for recovery purposes.

SPORT_ID="${3:-badminton}"
case "$SPORT_ID" in
  badminton|tennis) ;;
  *) echo "ERROR: sport must be badminton or tennis (got: $SPORT_ID)" >&2; exit 64 ;;
esac
DEPLOY_NAME="good-${SPORT_ID}-gpu-api"
DEFAULT_ARCHIVE_PATH="/root/${DEPLOY_NAME}-upload.zip"
DEFAULT_APP_DIR="/root/${DEPLOY_NAME}"

ARCHIVE_PATH="${1:-$DEFAULT_ARCHIVE_PATH}"
APP_DIR="${2:-$DEFAULT_APP_DIR}"
STATE_DIR="${GOOD_BADMINTON_STATE_DIR:-${APP_DIR}-state}"
PYTHON_BIN="${GOOD_BADMINTON_PYTHON_BIN:-python3}"
ENV_FILE="${STATE_DIR}/.gpu-api.env"
DATA_DIR="${STATE_DIR}/api_data"
WEIGHTS_DIR="${STATE_DIR}/weights"
LEGACY_APP_DIR="${GOOD_BADMINTON_LEGACY_APP_DIR:-}"
REMOVE_LEGACY_APP="${GOOD_BADMINTON_REMOVE_LEGACY_APP:-0}"

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

echo "[1/6] Validating uploaded package..."
unzip -q "$ARCHIVE_PATH" -d "$STAGING_DIR"

SOURCE_DIR=""
if [[ -f "$STAGING_DIR/$DEPLOY_NAME/api/gpu_stream_app.py" ]]; then
  SOURCE_DIR="$STAGING_DIR/$DEPLOY_NAME"
elif [[ -f "$STAGING_DIR/api/app.py" ]]; then
  SOURCE_DIR="$STAGING_DIR"
else
  fail "Package does not contain $DEPLOY_NAME/api/gpu_stream_app.py"
fi

[[ -f "$SOURCE_DIR/deploy/start_gpu_api_container.sh" ]] || \
  fail "Package is missing deploy/start_gpu_api_container.sh"
[[ -f "$SOURCE_DIR/deploy/start_badminton_gpu_container.sh" ]] || \
  fail "Package is missing deploy/start_badminton_gpu_container.sh"
[[ -f "$SOURCE_DIR/deploy/start_sport_gpu_container.sh" ]] || \
  fail "Package is missing deploy/start_sport_gpu_container.sh"
[[ -f "$SOURCE_DIR/apps/${SPORT_ID}_gpu/app.py" ]] || \
  fail "Package is missing fixed ${SPORT_ID} GPU entry point"
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

if [[ "$SPORT_ID" == "tennis" && -f "$SOURCE_DIR/weights/yolo11s-ball.pt" ]]; then
  # The optional package payload is the existing badminton checkpoint used
  # only by the explicit tennis experimental mode. Move it into persistent
  # state before replacing APP_DIR, just like all other server weights.
  cp -p "$SOURCE_DIR/weights/yolo11s-ball.pt" "$WEIGHTS_DIR/yolo11s-ball.pt"
  echo "Installed experimental tennis YOLO-ball checkpoint into persistent weights."
fi

SPORT_ENV_PREFIX="GOOD_${SPORT_ID^^}_STREAM"
POSE_MODEL_VAR="${SPORT_ENV_PREFIX}_POSE_MODEL"
POSE_MODEL_PATH="$(sed -n "s/^${POSE_MODEL_VAR}=//p" "$ENV_FILE" | tail -n 1 | tr -d '\r')"
if [[ -z "$POSE_MODEL_PATH" ]]; then
  fail "Missing ${POSE_MODEL_VAR} in $ENV_FILE. Upload this sport's pose checkpoint into $WEIGHTS_DIR and configure its absolute path before refresh."
fi
[[ -f "$POSE_MODEL_PATH" ]] || fail "Configured ${POSE_MODEL_VAR} does not exist: $POSE_MODEL_PATH"

if [[ "$SPORT_ID" == "badminton" ]]; then
  [[ -f "$WEIGHTS_DIR/yolo11s-ball.pt" ]] || fail \
    "Missing $WEIGHTS_DIR/yolo11s-ball.pt. Upload the checked ball-model weight there once; it is preserved on later code upgrades."
else
  tennis_checkpoint="$(sed -n 's/^GOOD_TENNIS_STREAM_BALL_MODEL=//p' "$ENV_FILE" | tail -n 1 | tr -d '\r')"
  if [[ -n "$tennis_checkpoint" && ! -f "$tennis_checkpoint" ]]; then
    fail "Tennis YOLO checkpoint not found: $tennis_checkpoint"
  fi
  if [[ -z "$tennis_checkpoint" && ! -f "$WEIGHTS_DIR/yolo11s-ball.pt" ]]; then
    echo "No tennis ball checkpoint installed; tennis service will start in pose-only mode."
  fi
fi

echo "[5/7] Stopping the old API, if present..."
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
stop_existing_api "$APP_DIR"
if [[ -n "$LEGACY_APP_DIR" ]]; then
  stop_existing_api "$LEGACY_APP_DIR"
fi

echo "[6/7] Replacing only application code..."
# APP_DIR is validated above.  STATE_DIR is a sibling and survives this rm.
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR"
cp -a "$SOURCE_DIR/." "$APP_DIR/"
chmod +x "$APP_DIR/deploy/start_gpu_api_container.sh" \
  "$APP_DIR/deploy/start_badminton_gpu_container.sh" \
  "$APP_DIR/deploy/start_sport_gpu_container.sh" \
  "$APP_DIR/deploy/start_tennis_gpu_container.sh" \
  "$APP_DIR/deploy/stop_gpu_api_container.sh" \
  "$APP_DIR/deploy/install_lap.sh" \
  "$APP_DIR/deploy/refresh_gpu_api_from_zip.sh"

# Make the code see the persistent state through its normal paths.  This
# prevents old jobs, cache data, model weights and the secret from being lost
# when the application directory is replaced on the next deployment.
rm -rf "$APP_DIR/.gpu-api.env" "$APP_DIR/api_data" "$APP_DIR/weights"
ln -s "$ENV_FILE" "$APP_DIR/.gpu-api.env"
ln -s "$DATA_DIR" "$APP_DIR/api_data"
ln -s "$WEIGHTS_DIR" "$APP_DIR/weights"

echo "[7/7] Starting and verifying the refreshed API..."
GOOD_BADMINTON_ENV_FILE="$ENV_FILE" \
GOOD_BADMINTON_PYTHON_BIN="$PYTHON_BIN" \
  "$APP_DIR/deploy/start_${SPORT_ID}_gpu_container.sh" "$APP_DIR"

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
health_url="http://127.0.0.1:${PORT:-8080}/api/v1/health"
for _ in {1..10}; do
  if curl -fsS --max-time 5 "$health_url"; then
    break
  fi
  sleep 1
done
curl -fsS --max-time 5 "$health_url" >/dev/null || \
  fail "GPU API started but did not remain healthy; see $APP_DIR/gpu-api.log"
echo
if [[ -n "$LEGACY_APP_DIR" && "$REMOVE_LEGACY_APP" == "1" ]]; then
  # Explicit opt-in only: the fresh process has passed its health check, so
  # removing this one legacy code directory cannot remove persistent state.
  rm -rf "$LEGACY_APP_DIR"
  echo "Removed legacy application directory: $LEGACY_APP_DIR"
fi
echo "Deployment complete. Code: $APP_DIR"
echo "Persistent state (preserved on next update): $STATE_DIR"
echo "Cloud startup command: GOOD_BADMINTON_ENV_FILE=$ENV_FILE bash $APP_DIR/deploy/start_gpu_api_container.sh $APP_DIR"
