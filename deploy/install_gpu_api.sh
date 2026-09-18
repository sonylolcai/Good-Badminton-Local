#!/usr/bin/env bash
set -euo pipefail

# Deploy the model/video-only API to a CUDA 12.4 GPU instance.  This script
# intentionally does not deploy business databases, users, SSO, or frontend.
DEFAULT_APP_DIR="$HOME/good-badminton-gpu-api"
APP_DIR="${1:-$DEFAULT_APP_DIR}"
SERVICE_NAME="good-badminton-gpu-api"
STATE_DIR="${GOOD_BADMINTON_STATE_DIR:-${APP_DIR}-state}"
ENV_FILE="${STATE_DIR}/.gpu-api.env"
DATA_DIR="${STATE_DIR}/api_data"
WEIGHTS_DIR="${STATE_DIR}/weights"

[[ "$APP_DIR" == "$DEFAULT_APP_DIR" ]] || {
  echo "For safety APP_DIR must be exactly $DEFAULT_APP_DIR (got: $APP_DIR)." >&2
  exit 64
}
[[ "$STATE_DIR" == "${DEFAULT_APP_DIR}-state" ]] || {
  echo "For safety STATE_DIR must be exactly ${DEFAULT_APP_DIR}-state (got: $STATE_DIR)." >&2
  exit 64
}

mkdir -p "$STATE_DIR" "$DATA_DIR" "$WEIGHTS_DIR"
chmod 700 "$STATE_DIR" "$DATA_DIR" "$WEIGHTS_DIR"

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required" >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# PyTorch publishes CUDA 12.1 and 12.4 wheels.  CUDA 12.4 needs an NVIDIA
# 550+ driver, while CUDA 12.1 remains compatible with the common 535 driver
# (whose toolkit banner may say CUDA 12.2).  Select the newest safe wheel
# rather than assuming the image's CUDA toolkit label is sufficient.
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
driver_major="${driver_version%%.*}"
if [[ "$driver_major" =~ ^[0-9]+$ ]] && (( driver_major >= 550 )); then
  PYTORCH_CUDA_INDEX="cu124"
  PYTORCH_TORCH_VERSION="2.5.1+cu124"
  PYTORCH_TORCHVISION_VERSION="0.20.1+cu124"
else
  PYTORCH_CUDA_INDEX="cu121"
  PYTORCH_TORCH_VERSION="2.5.1+cu121"
  PYTORCH_TORCHVISION_VERSION="0.20.1+cu121"
fi
echo "Using PyTorch ${PYTORCH_CUDA_INDEX} wheels for NVIDIA driver ${driver_version}."

[[ -f "$APP_DIR/api/app.py" ]] || {
  echo "Extract the uploaded complete GPU package into $APP_DIR before installation." >&2
  exit 1
}
echo "Installing the extracted GPU package at $APP_DIR."

if [[ "${GOOD_BADMINTON_USE_SYSTEM_TORCH:-0}" == "1" ]]; then
  # Official rental images often bundle a CUDA-enabled Conda PyTorch but block
  # download.pytorch.org.  Reuse it deliberately, after proving CUDA works.
  PYTHON_BIN="${GOOD_BADMINTON_PYTHON_BIN:-python3}"
  "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "The selected system PyTorch cannot access CUDA"
print(f"Reusing system PyTorch {torch.__version__} (CUDA {torch.version.cuda}).")
PY
else
  python3 -m venv "$STATE_DIR/.venv"
  PYTHON_BIN="$STATE_DIR/.venv/bin/python"
  "$PYTHON_BIN" -m pip install --upgrade pip

  # CUDA wheels must be installed before the project packages.  The base
  # requirements file remains CPU-friendly for Windows development, so filter
  # only its torch pin/index during GPU setup.
  "$PYTHON_BIN" -m pip install \
    --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_INDEX}" \
    "torch==${PYTORCH_TORCH_VERSION}" "torchvision==${PYTORCH_TORCHVISION_VERSION}"
fi

# When the image has no public egress, callers upload Linux wheels and set
# GOOD_BADMINTON_WHEELHOUSE.  --no-index ensures pip cannot silently fall back
# to an uncontrolled public source.
PIP_SOURCE_ARGS=()
if [[ -n "${GOOD_BADMINTON_WHEELHOUSE:-}" ]]; then
  [[ -d "$GOOD_BADMINTON_WHEELHOUSE" ]] || {
    echo "GOOD_BADMINTON_WHEELHOUSE does not exist: $GOOD_BADMINTON_WHEELHOUSE" >&2
    exit 1
  }
  PIP_SOURCE_ARGS=(--no-index --find-links "$GOOD_BADMINTON_WHEELHOUSE")
fi
grep -vE '^(--extra-index-url|torch==|torchvision==)' "$APP_DIR/requirements.txt" \
  > "$APP_DIR/.requirements-server.txt"
"$PYTHON_BIN" -m pip install "${PIP_SOURCE_ARGS[@]}" -r "$APP_DIR/.requirements-server.txt"
rm -f "$APP_DIR/.requirements-server.txt"

# ByteTrack is the default production tracker.  Keep this explicit runtime
# check even though requirements.txt already declares lap: it verifies that
# the selected CUDA/Conda Python, rather than an unrelated system Python, can
# actually import it.
GOOD_BADMINTON_PYTHON_BIN="$PYTHON_BIN" \
GOOD_BADMINTON_WHEELHOUSE="${GOOD_BADMINTON_WHEELHOUSE:-}" \
  "$APP_DIR/deploy/install_lap.sh"

if [[ ! -f "$ENV_FILE" && -f "$APP_DIR/.gpu-api.env" && ! -L "$APP_DIR/.gpu-api.env" ]]; then
  cp -p "$APP_DIR/.gpu-api.env" "$ENV_FILE"
fi
if [[ ! -L "$APP_DIR/api_data" && -d "$APP_DIR/api_data" ]] && \
   [[ -z "$(find "$DATA_DIR" -mindepth 1 -print -quit)" ]]; then
  cp -a "$APP_DIR/api_data/." "$DATA_DIR/"
fi
if [[ ! -L "$APP_DIR/weights" && -d "$APP_DIR/weights" ]] && \
   [[ -z "$(find "$WEIGHTS_DIR" -mindepth 1 -print -quit)" ]]; then
  cp -a "$APP_DIR/weights/." "$WEIGHTS_DIR/"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  umask 077
  api_key="$(openssl rand -hex 32)"
  printf 'GOOD_BADMINTON_API_KEY=%s\nGOOD_BADMINTON_API_DATA_DIR=%s\nPORT=8080\n' \
    "$api_key" "$DATA_DIR" > "$ENV_FILE"
  echo "Created $ENV_FILE. Store its API key in the business service secret manager."
fi
chmod 600 "$ENV_FILE"
env_tmp="$(mktemp "${STATE_DIR}/.gpu-api.env.XXXXXX")"
grep -v '^GOOD_BADMINTON_API_DATA_DIR=' "$ENV_FILE" > "$env_tmp" || true
printf 'GOOD_BADMINTON_API_DATA_DIR=%s\n' "$DATA_DIR" >> "$env_tmp"
chmod 600 "$env_tmp"
mv "$env_tmp" "$ENV_FILE"

rm -rf "$APP_DIR/.gpu-api.env" "$APP_DIR/api_data" "$APP_DIR/weights"
ln -s "$ENV_FILE" "$APP_DIR/.gpu-api.env"
ln -s "$DATA_DIR" "$APP_DIR/api_data"
ln -s "$WEIGHTS_DIR" "$APP_DIR/weights"

PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - <<'PY'
import tempfile

from api.app import create_app

with tempfile.TemporaryDirectory() as data_dir:
    app = create_app(data_dir=data_dir, start_worker=False)
    assert app.title == "Good-Badminton multi-sport GPU API"
PY

service_file="/etc/systemd/system/${SERVICE_NAME}.service"
if command -v systemctl >/dev/null && systemctl show-environment >/dev/null 2>&1; then
  sed \
    -e "s|__GPU_USER__|$USER|g" \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    -e "s|__ENV_FILE__|$ENV_FILE|g" \
    -e "s|__PYTHON_BIN__|$PYTHON_BIN|g" \
    "$APP_DIR/deploy/${SERVICE_NAME}.service" | sudo tee "$service_file" >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now "$SERVICE_NAME"
  sudo systemctl status "$SERVICE_NAME" --no-pager
else
  # Most rented GPU images expose a long-lived Docker container rather than
  # systemd.  Keep the same API contract and record the PID for status/stop.
  pid_file="$APP_DIR/.gpu-api.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    kill "$(cat "$pid_file")"
  fi
  (
    cd "$APP_DIR"
    set -a
    # shellcheck disable=SC1091
    source "$ENV_FILE"
    set +a
    nohup "$PYTHON_BIN" -m uvicorn api.app:app --host 0.0.0.0 --port "${PORT:-8080}" \
      > "$APP_DIR/gpu-api.log" 2>&1 &
    echo $! > "$pid_file"
  )
  sleep 2
  kill -0 "$(cat "$pid_file")" 2>/dev/null || {
    echo "GPU API failed to start; see $APP_DIR/gpu-api.log" >&2
    exit 1
  }
  echo "Started container-mode GPU API (PID $(cat "$pid_file")); log: $APP_DIR/gpu-api.log"
fi
