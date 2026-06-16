# Spotify Song Popularity Prediction

Predict whether a track is a "hit" (popularity ≥ 50) from its Spotify audio features.
Primary framing: hit-vs-flop **classification**; secondary: 0–100 **regression**.
Includes EDA, baseline + gradient-boosted models, SHAP interpretability, and a Streamlit demo.

## Project layout

```
data/{raw,processed}/   # datasets (gitignored)
notebooks/01_eda.ipynb  # exploratory analysis
src/                    # config, data, features, train, evaluate, interpret, run_pipeline
app/streamlit_app.py    # interactive demo
models/                 # saved Pipeline artifact (gitignored)
reports/figures/        # plots (gitignored)
tests/                  # pytest suite
```

## Setup

```bash
cd "ML PROJECT"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS note: `lightgbm`/`xgboost` need OpenMP — `brew install libomp` if imports fail.

## Get the dataset

Kaggle: `maharshipandya/spotify-tracks-dataset` (~114K tracks).

**Option A — Kaggle API**
1. Create a Kaggle account, then Account → "Create New Token" to download `kaggle.json`.
2. `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
3. `kaggle datasets download -d maharshipandya/spotify-tracks-dataset -p data/raw --unzip`

**Option B — Manual fallback**
Download the CSV from the dataset page and save it as `data/raw/dataset.csv`.

## Train & evaluate

```bash
python -m src.run_pipeline
```

Loads and cleans the data, cross-validates five models on PR-AUC, fits the best one,
prints held-out test metrics, saves ROC/PR/confusion + SHAP plots to `reports/figures/`,
and writes `models/best_classifier.joblib`.

## Test

```bash
pytest -v
```

## Demo

```bash
streamlit run app/streamlit_app.py
```

Move the sliders for a live hit probability. Requires a trained model (run the pipeline first).

## Definitions

- **Hit**: `popularity >= 50` (configurable via `HIT_THRESHOLD` in `src/config.py`).
- Class imbalance handled via `class_weight`/`average_precision` scoring; the best model is selected on cross-validated PR-AUC.

## Results

_To be updated after the first pipeline run: best model, PR-AUC / ROC-AUC / F1, and top SHAP drivers of popularity._
