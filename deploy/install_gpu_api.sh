#!/usr/bin/env bash
set -euo pipefail

# Deploy the model/video-only API to a CUDA 12.4 GPU instance.  This script
# intentionally does not deploy business databases, users, SSO, or frontend.
APP_DIR="${1:-$HOME/good-badminton}"
BRANCH="${2:-fixed-camera-singles-spatial-tracking}"
REPOSITORY="${GOOD_BADMINTON_REPOSITORY:-https://github.com/sonylolcai/Good-Badminton-Local.git}"
SERVICE_NAME="good-badminton-gpu-api"

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

if [[ "${GOOD_BADMINTON_SKIP_GIT_SYNC:-0}" == "1" ]]; then
  [[ -f "$APP_DIR/api/app.py" ]] || {
    echo "GOOD_BADMINTON_SKIP_GIT_SYNC=1 requires an extracted project at $APP_DIR" >&2
    exit 1
  }
  echo "Skipping Git sync; deploying the extracted source at $APP_DIR."
elif [[ ! -d "$APP_DIR/.git" ]]; then
  GIT_TERMINAL_PROMPT=0 git clone --branch "$BRANCH" --single-branch "$REPOSITORY" "$APP_DIR"
else
  GIT_TERMINAL_PROMPT=0 git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" switch "$BRANCH"
  GIT_TERMINAL_PROMPT=0 git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
fi

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
  python3 -m venv "$APP_DIR/.venv"
  PYTHON_BIN="$APP_DIR/.venv/bin/python"
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

if [[ ! -f "$APP_DIR/.gpu-api.env" ]]; then
  umask 077
  api_key="$(openssl rand -hex 32)"
  printf 'GOOD_BADMINTON_API_KEY=%s\nGOOD_BADMINTON_API_DATA_DIR=%s/api_data\nPORT=8080\n' \
    "$api_key" "$APP_DIR" > "$APP_DIR/.gpu-api.env"
  echo "Created $APP_DIR/.gpu-api.env. Store its API key in the business service secret manager."
fi

"$PYTHON_BIN" -m unittest tests.test_gpu_api -v

service_file="/etc/systemd/system/${SERVICE_NAME}.service"
if command -v systemctl >/dev/null && systemctl show-environment >/dev/null 2>&1; then
  sed \
    -e "s|__GPU_USER__|$USER|g" \
    -e "s|__APP_DIR__|$APP_DIR|g" \
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
    source "$APP_DIR/.gpu-api.env"
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
