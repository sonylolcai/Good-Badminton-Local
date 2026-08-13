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

if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPOSITORY" "$APP_DIR"
else
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" switch "$BRANCH"
  git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip

# CUDA wheels must be installed before the project packages.  The base
# requirements file remains CPU-friendly for Windows development, so filter
# only its torch pin/index during GPU setup.
"$APP_DIR/.venv/bin/python" -m pip install \
  --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_INDEX}" \
  "torch==${PYTORCH_TORCH_VERSION}" "torchvision==${PYTORCH_TORCHVISION_VERSION}"
grep -vE '^(--extra-index-url|torch==|torchvision==)' "$APP_DIR/requirements.txt" \
  > "$APP_DIR/.requirements-server.txt"
"$APP_DIR/.venv/bin/python" -m pip install -r "$APP_DIR/.requirements-server.txt"
rm -f "$APP_DIR/.requirements-server.txt"

if [[ ! -f "$APP_DIR/.gpu-api.env" ]]; then
  umask 077
  api_key="$(openssl rand -hex 32)"
  printf 'GOOD_BADMINTON_API_KEY=%s\nGOOD_BADMINTON_API_DATA_DIR=%s/api_data\nPORT=8001\n' \
    "$api_key" "$APP_DIR" > "$APP_DIR/.gpu-api.env"
  echo "Created $APP_DIR/.gpu-api.env. Store its API key in the business service secret manager."
fi

"$APP_DIR/.venv/bin/python" -m unittest tests.test_gpu_api -v

service_file="/etc/systemd/system/${SERVICE_NAME}.service"
sed \
  -e "s|__GPU_USER__|$USER|g" \
  -e "s|__APP_DIR__|$APP_DIR|g" \
  "$APP_DIR/deploy/${SERVICE_NAME}.service" | sudo tee "$service_file" >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager
