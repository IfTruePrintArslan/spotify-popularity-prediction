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

**Option A — Kaggle API** (requires `kaggle>=2.x`, already pinned)
1. On Kaggle: **Settings → API → Create New Token**. You'll get a token like `KGAT_…` (copy it — it's shown only once).
2. Save it where the CLI reads it automatically:
   `mkdir -p ~/.kaggle && printf '%s' 'KGAT_your_token_here' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
   (Alternatively: `export KAGGLE_API_TOKEN=KGAT_your_token_here`.)
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

Trained on **89,741 tracks** after de-duplicating `track_id` (114k raw rows → ~89.7k unique). Hit rate ≈ **23.6%** (popularity ≥ 50).

Cross-validated PR-AUC (5-fold, training split):

| Model | CV PR-AUC |
|-------|-----------|
| **RandomForest** (selected) | **0.661** |
| XGBoost | 0.652 |
| LightGBM | 0.649 |
| LogisticRegression | 0.593 |
| Dummy (most-frequent) | 0.236 |

Held-out test metrics (RandomForest):

| Metric | Value |
|--------|-------|
| Accuracy | 0.824 |
| ROC-AUC | 0.864 |
| PR-AUC | 0.671 |
| F1 | 0.526 |
| Precision | 0.720 |
| Recall | 0.415 |

**Top drivers (permutation importance):** `track_genre` dominates (~0.37, ≈20× the next feature), followed by instrumentalness, duration, valence, and loudness — audio features carry real but secondary signal once genre is known. ROC/PR/confusion + SHAP summary plots are saved to `reports/figures/` (gitignored).
