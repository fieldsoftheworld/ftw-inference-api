#!/usr/bin/env bash
# In-place update of an already-provisioned instance (see deploy.sh for first boot).
#
# Runs as the service user from the repository root, at the commit that should be
# deployed. The caller (the `ftw-dev-deploy-api` SSM document) has already run
# `git fetch` and `git checkout --force <sha>`, so this script is always the version
# from the target commit.
#
# Exits non-zero if the service does not become healthy.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

SERVICE="ftw-inference-api"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/v1/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-90}"
HEALTH_INTERVAL=2

cd "$(git rev-parse --show-toplevel)"
echo "Updating $SERVICE to $(git rev-parse HEAD)"

echo "Installing Python dependencies..."
uv sync --frozen

mkdir -p server/data/uploads server/data/results server/logs

echo "Restarting $SERVICE (in-flight and queued tasks are dropped)..."
sudo systemctl restart "$SERVICE"

echo "Waiting up to ${HEALTH_TIMEOUT}s for $HEALTH_URL..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
until curl -fsS --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
        echo "ERROR: $SERVICE did not become healthy within ${HEALTH_TIMEOUT}s" >&2
        sudo systemctl status "$SERVICE" --no-pager >&2 || true
        sudo journalctl -u "$SERVICE" -n 100 --no-pager >&2 || true
        exit 1
    fi
    sleep "$HEALTH_INTERVAL"
done

echo "$SERVICE is healthy"
echo "Deployed commit: $(git rev-parse HEAD)"
