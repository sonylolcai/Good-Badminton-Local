#!/bin/zsh
# Installs a user-level launchd service. It deliberately never embeds the
# device secret in this script or the plist.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
APP_DIR="$HOME/Library/Application Support/GoodBadminton/venue-gateway"
ENV_FILE="$HOME/Library/Application Support/GoodBadminton/venue-gateway.env"
LOG_DIR="$HOME/Library/Logs/GoodBadminton"
PLIST_DEST="$HOME/Library/LaunchAgents/com.goodbadminton.venue-gateway.plist"
LABEL="com.goodbadminton.venue-gateway"
UID_VALUE="$(id -u)"

if ! command -v brew >/dev/null 2>&1; then
  print "Homebrew is required once to install Python and ffmpeg."
  print "Install it from https://brew.sh, then run this script again."
  exit 1
fi

brew list python@3.11 >/dev/null 2>&1 || brew install python@3.11
brew list ffmpeg >/dev/null 2>&1 || brew install ffmpeg

mkdir -p "$APP_DIR" "$LOG_DIR" "$HOME/Library/LaunchAgents"
rsync -a --delete --exclude '.venv' --exclude '*.env' --exclude 'dist' \
  "$SCRIPT_DIR/" "$APP_DIR/"

PYTHON_BIN="$(brew --prefix python@3.11)/bin/python3.11"
if [[ ! -x "$APP_DIR/.venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$APP_DIR/.venv"
fi
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$ENV_FILE" ]]; then
  cp "$APP_DIR/venue-gateway.env.example" "$ENV_FILE"
  sed -i '' "s|SPOOL_DIR=SET_BY_INSTALLER|SPOOL_DIR=$APP_DIR/spool|" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  print "Created $ENV_FILE. Copy the protected server-exported values into it before starting."
fi

sed \
  -e "s|__APP_DIR__|$APP_DIR|g" \
  -e "s|__ENV_FILE__|$ENV_FILE|g" \
  -e "s|__LOG_DIR__|$LOG_DIR|g" \
  "$APP_DIR/com.goodbadminton.venue-gateway.plist" > "$PLIST_DEST"
plutil -lint "$PLIST_DEST"

launchctl bootout "gui/$UID_VALUE/$LABEL" 2>/dev/null || true
if grep -q 'SET_BY_' "$ENV_FILE"; then
  print "The protected environment file still has placeholders; service was installed but not started."
  print "After filling it, run: launchctl bootstrap gui/$UID_VALUE $PLIST_DEST"
  exit 0
fi
launchctl bootstrap "gui/$UID_VALUE" "$PLIST_DEST"
print "Installed and started $LABEL."
print "Logs: $LOG_DIR/venue-gateway.stderr.log"
