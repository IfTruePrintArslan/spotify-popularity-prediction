#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."          # cd to repo root (scripts/ is one level down)
source .venv/bin/activate

echo "Opening EDA notebook in Jupyter (notebooks/01_eda.ipynb)..."
jupyter notebook notebooks/01_eda.ipynb
