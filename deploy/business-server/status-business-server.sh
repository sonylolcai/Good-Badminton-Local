#!/usr/bin/env bash
set -euo pipefail

for service in good-badminton-edge-gateway good-badminton-operator-api good-badminton-operator-web; do
  systemctl is-active --quiet "$service" && echo "$service: active" || echo "$service: inactive"
done

for endpoint in \
  http://127.0.0.1:18080/api/v1/health \
  http://127.0.0.1:8000/api/v1/system/readiness \
  http://127.0.0.1:3000/; do
  if curl --fail --silent --show-error --max-time 5 "$endpoint" >/dev/null; then
    echo "$endpoint: healthy"
  else
    echo "$endpoint: unavailable" >&2
  fi
done
