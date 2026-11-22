#!/usr/bin/env bash
# Start the FetalVision HTTP service (uvicorn, 1 worker). Usage: scripts/serve.sh [config] [port]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/service.yaml}"
PORT="${2:-${PORT:-8000}}"
export FETALVISION_SERVICE_CONFIG="$CONFIG"
export FETALVISION_DATA="${FETALVISION_DATA:-./data}"
exec uvicorn fetalvision.service.app:app --host 0.0.0.0 --port "$PORT" --workers 1 --log-level info
