#!/usr/bin/env bash
set -euo pipefail

# Deploy the business control plane only: PostgreSQL migrations, signed edge
# ingest gateway, operator API, and Next.js SaaS console.  GPU inference stays
# on its separately secured GPU host.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_DIR="$DEFAULT_APP_DIR"
RUN_USER="goodbadminton"
RUN_GROUP="goodbadminton"
ENV_FILE="/etc/good-badminton/business.env"
OPERATOR_HOST=""
API_HOST=""
OPERATOR_CERT=""
OPERATOR_KEY=""
API_CERT=""
API_KEY=""
INSTALL_NGINX=1

usage() {
  cat <<'EOF'
Usage: sudo bash deploy/business-server/deploy-business-server.sh \
  --operator-host operator.example.com --api-host api.example.com [options]

Options:
  --app-dir PATH       Checked-out Good-Badminton repository (default: script's repository)
  --run-user USER      Unprivileged service account (default: goodbadminton)
  --env-file PATH      Root-managed runtime env file (default: /etc/good-badminton/business.env)
  --operator-host DNS  Public operator-console hostname (required)
  --api-host DNS       Public terminal/business API hostname (required)
  --operator-cert PATH TLS certificate for operator hostname
  --operator-key PATH  TLS private key for operator hostname
  --api-cert PATH      TLS certificate for API hostname
  --api-key PATH       TLS private key for API hostname
  --skip-nginx         Install/restart services and migrations only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-dir) APP_DIR="$2"; shift 2 ;;
    --run-user) RUN_USER="$2"; RUN_GROUP="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --operator-host) OPERATOR_HOST="$2"; shift 2 ;;
    --api-host) API_HOST="$2"; shift 2 ;;
    --operator-cert) OPERATOR_CERT="$2"; shift 2 ;;
    --operator-key) OPERATOR_KEY="$2"; shift 2 ;;
    --api-cert) API_CERT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --skip-nginx) INSTALL_NGINX=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ ${EUID:-$(id -u)} -eq 0 ]] || { echo "Run with sudo/root." >&2; exit 1; }
[[ -f "$APP_DIR/operator_api/main.py" && -f "$APP_DIR/business_gateway/edge_api.py" ]] || {
  echo "APP_DIR is not a Good-Badminton checkout: $APP_DIR" >&2; exit 1;
}
[[ -f "$ENV_FILE" ]] || { echo "Missing protected environment file: $ENV_FILE" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
command -v npm >/dev/null || { echo "npm (with Node.js >=20) is required" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "systemd is required" >&2; exit 1; }

node_major="$(node -p 'Number(process.versions.node.split(".")[0])')"
(( node_major >= 20 )) || { echo "Node.js >=20 is required by Next.js 16 (found $(node --version))." >&2; exit 1; }

# Systemd parses EnvironmentFile itself; source only this root-managed file to
# validate values and run the one-off migration.  It must be shell-compatible.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
for required in GOOD_BADMINTON_BUSINESS_DATABASE_URL GOOD_BADMINTON_EDGE_MASTER_KEY GOOD_BADMINTON_GPU_API_URL GOOD_BADMINTON_GPU_API_KEY NEXT_PUBLIC_OPERATOR_API_BASE_URL; do
  value="${!required:-}"
  [[ -n "$value" && "$value" != *replace-with-* ]] || { echo "Set $required in $ENV_FILE" >&2; exit 1; }
done
[[ ${#GOOD_BADMINTON_EDGE_MASTER_KEY} -ge 32 ]] || { echo "GOOD_BADMINTON_EDGE_MASTER_KEY must be at least 32 bytes." >&2; exit 1; }

if ! getent passwd "$RUN_USER" >/dev/null; then
  useradd --system --create-home --home-dir /var/lib/good-badminton --shell /usr/sbin/nologin "$RUN_USER"
fi
# Keep the checkout updateable by its existing deploy-key owner while allowing
# the unprivileged service user to create its venv, node_modules and build.
chgrp -R "$RUN_GROUP" "$APP_DIR"
chmod -R g+rwX "$APP_DIR"
find "$APP_DIR" -type d -exec chmod g+s {} +
if [[ -n "${SUDO_USER:-}" && "$SUDO_USER" != "root" ]]; then
  usermod -a -G "$RUN_GROUP" "$SUDO_USER"
fi
install -d -o "$RUN_USER" -g "$RUN_GROUP" -m 0750 \
  /var/lib/good-badminton/edge-staging /var/lib/good-badminton/edge-preview

PYTHON_BIN="$APP_DIR/.venv-business/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  runuser -u "$RUN_USER" -- python3 -m venv "$APP_DIR/.venv-business"
fi
runuser -u "$RUN_USER" -- "$PYTHON_BIN" -m pip install --upgrade pip
runuser -u "$RUN_USER" -- "$PYTHON_BIN" -m pip install -r "$APP_DIR/deploy/business-server/requirements.txt"

# Build with the public operator API origin only; server secrets are not passed
# into the browser bundle.
runuser -u "$RUN_USER" -- env NODE_ENV=production \
  NEXT_PUBLIC_OPERATOR_API_BASE_URL="$NEXT_PUBLIC_OPERATOR_API_BASE_URL" \
  npm --prefix "$APP_DIR/webui-next" ci
runuser -u "$RUN_USER" -- env NODE_ENV=production \
  NEXT_PUBLIC_OPERATOR_API_BASE_URL="$NEXT_PUBLIC_OPERATOR_API_BASE_URL" \
  npm --prefix "$APP_DIR/webui-next" run build

"$PYTHON_BIN" "$APP_DIR/deploy/business-server/apply_migrations.py"

NPM_BIN="$(command -v npm)"
for service in good-badminton-edge-gateway good-badminton-operator-api good-badminton-operator-web; do
  sed -e "s|__RUN_USER__|$RUN_USER|g" -e "s|__RUN_GROUP__|$RUN_GROUP|g" \
      -e "s|__APP_DIR__|$APP_DIR|g" -e "s|__PYTHON_BIN__|$PYTHON_BIN|g" \
      -e "s|__NPM_BIN__|$NPM_BIN|g" \
      "$APP_DIR/deploy/business-server/$service.service" > "/etc/systemd/system/$service.service"
done

if (( INSTALL_NGINX )); then
  command -v nginx >/dev/null || { echo "nginx is required unless --skip-nginx is used" >&2; exit 1; }
  [[ -n "$OPERATOR_HOST" && -n "$API_HOST" ]] || { echo "--operator-host and --api-host are required" >&2; exit 1; }
  [[ "$OPERATOR_HOST" =~ ^[A-Za-z0-9.-]+$ && "$API_HOST" =~ ^[A-Za-z0-9.-]+$ ]] || { echo "Hostnames may contain only letters, digits, dots and hyphens." >&2; exit 1; }
  OPERATOR_CERT="${OPERATOR_CERT:-/etc/letsencrypt/live/$OPERATOR_HOST/fullchain.pem}"
  OPERATOR_KEY="${OPERATOR_KEY:-/etc/letsencrypt/live/$OPERATOR_HOST/privkey.pem}"
  API_CERT="${API_CERT:-/etc/letsencrypt/live/$API_HOST/fullchain.pem}"
  API_KEY="${API_KEY:-/etc/letsencrypt/live/$API_HOST/privkey.pem}"
  for certificate in "$OPERATOR_CERT" "$OPERATOR_KEY" "$API_CERT" "$API_KEY"; do
    [[ -f "$certificate" ]] || { echo "Missing TLS file: $certificate" >&2; exit 1; }
  done
  nginx_target="/etc/nginx/conf.d/good-badminton.conf"
  nginx_temp="$(mktemp)"
  sed -e "s|__OPERATOR_HOST__|$OPERATOR_HOST|g" -e "s|__API_HOST__|$API_HOST|g" \
      -e "s|__OPERATOR_CERT__|$OPERATOR_CERT|g" -e "s|__OPERATOR_KEY__|$OPERATOR_KEY|g" \
      -e "s|__API_CERT__|$API_CERT|g" -e "s|__API_KEY__|$API_KEY|g" \
      "$APP_DIR/deploy/business-server/nginx.good-badminton.conf.template" > "$nginx_temp"
  install -m 0644 "$nginx_temp" "$nginx_target"
  rm -f "$nginx_temp"
  nginx -t
  systemctl reload nginx
fi

systemctl daemon-reload
systemctl enable --now good-badminton-edge-gateway good-badminton-operator-api good-badminton-operator-web

for endpoint in http://127.0.0.1:18080/api/v1/health http://127.0.0.1:8000/api/v1/system/readiness http://127.0.0.1:3000/; do
  curl --fail --silent --show-error --max-time 15 "$endpoint" >/dev/null
done

echo "Business server deployment completed."
echo "Check: systemctl status good-badminton-edge-gateway good-badminton-operator-api good-badminton-operator-web"
