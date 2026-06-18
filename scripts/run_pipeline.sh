#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."          # cd to repo root (scripts/ is one level down)
source .venv/bin/activate

echo "Running full ML pipeline (src/run_pipeline.py) — retrains model, writes metrics.json + figures..."
echo "WARNING: this takes approximately 7 minutes to complete."
python -m src.run_pipeline
