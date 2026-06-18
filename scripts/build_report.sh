#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."          # cd to repo root (scripts/ is one level down)
source .venv/bin/activate

echo "Regenerating research-paper PDF and DOCX from docs/REPORT.md..."

TITLE="Predicting Spotify Track Popularity: A Machine Learning Classification Approach Using Ensemble Methods"

pandoc docs/front_matter.md -o /tmp/front_matter.html
pandoc docs/REPORT.md \
    --include-before-body=/tmp/front_matter.html \
    -o /tmp/REPORT_paper.html \
    -s --toc --embed-resources \
    --css docs/paper.css \
    --metadata title="$TITLE"
python -m weasyprint /tmp/REPORT_paper.html docs/REPORT.pdf
pandoc docs/front_matter.md docs/REPORT.md \
    -o docs/REPORT.docx \
    --toc \
    --reference-doc=docs/reference.docx \
    --metadata title="$TITLE"

echo "Built docs/REPORT.pdf and docs/REPORT.docx"
