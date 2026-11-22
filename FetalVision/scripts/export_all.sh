#!/usr/bin/env bash
# Re-export the three models to ONNX and run the parity checks. Uses the real models when they exist
# (models/<selected_run>.keras, models/head_detector.pt, models/head_segmenter.pt) and stand-ins otherwise.
# Usage: scripts/export_all.sh [configs/export.yaml] [--conf 0.001]   (extra args go to parity)
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/export.yaml}"; shift || true
export FETALVISION_DATA="${FETALVISION_DATA:-./data}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
python -m fetalvision.export.export_classifier --config "$CONFIG"
python -m fetalvision.export.export_detector   --config "$CONFIG"
python -m fetalvision.export.export_segmenter  --config "$CONFIG"
python -m fetalvision.export.parity --kind all --config "$CONFIG" "$@"
python - <<'PY'
from fetalvision.pipeline import FetalVisionPipeline
from fetalvision.common.config import load_config
p = FetalVisionPipeline.from_config(load_config("configs/service.yaml"))
print("manifest ->", p.save_manifest())
PY
