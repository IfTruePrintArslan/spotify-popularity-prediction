#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."          # cd to repo root (scripts/ is one level down)
source .venv/bin/activate

echo "Launching Streamlit dashboard (app/streamlit_app.py)..."
streamlit run app/streamlit_app.py
