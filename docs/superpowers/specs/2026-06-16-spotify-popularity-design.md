# Spotify Song Popularity Prediction — Design Spec

**Date:** 2026-06-16
**Status:** Approved (brainstorming)
**Owner:** arslan (personal project)

## 1. Overview
Predict Spotify track popularity from audio features. Primary framing: binary classification (hit vs flop). Secondary: regression (exact 0–100 score). Portfolio-grade pipeline: EDA → preprocessing → baseline + gradient-boosted models → tuning → SHAP interpretability → Streamlit demo.

## 2. Prediction Targets
### 2.1 Classification (primary)
- Label: `hit = popularity >= HIT_THRESHOLD`, default `HIT_THRESHOLD = 50`.
- Expected ~20–25% positive → class imbalance.
- Handling: stratified splits, `class_weight='balanced'` (linear/RF) / `scale_pos_weight` (XGBoost/LightGBM).
- Headline metrics: ROC-AUC, PR-AUC, F1; plus precision/recall and confusion matrix.
- Report threshold sensitivity (40/50/60 and percentile-based top-25%).

### 2.2 Regression (secondary)
- Predict continuous popularity 0–100.
- Metrics: R², MAE, RMSE.
- Note: popularity is noisy and right-skewed; modest R² expected and reported honestly.

## 3. Dataset
- Source: Kaggle `maharshipandya/spotify-tracks-dataset` (~114K rows, single CSV).
- Acquisition: Kaggle API with `kaggle.json` token → `data/raw/`. Manual-download fallback documented in README.
- Numeric features: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, time_signature.
- Categorical: explicit (bool), track_genre.
- Target: popularity.

## 4. Cleaning & Leakage Control
- Deduplicate: the same `track_id` appears under multiple genres. Drop duplicate track_ids (keep first, or aggregate genre into a list) to prevent train/test leakage of identical tracks.
- Drop identifiers/free-text with no signal: track_id, artists, album_name, track_name, unnamed index.
- Encode: explicit → {0,1}; track_genre → one-hot (or target encoding if dimensionality hurts).
- Scaling: StandardScaler for linear models only (tree models skip).
- Missing values: impute or drop (dataset largely complete; assert post-clean).
- Leakage rule: train/test split BEFORE fitting any transform; all preprocessing inside a ColumnTransformer fit on training folds only.

## 5. Project Structure
```
ML PROJECT/
  data/{raw,processed}/
  notebooks/01_eda.ipynb
  src/
    data.py        # load, clean, dedup, split
    features.py    # ColumnTransformer build (encode/scale)
    train.py       # model defs, CV, tuning, fit, save
    evaluate.py    # metrics, plots
    interpret.py   # SHAP, permutation importance
  app/streamlit_app.py
  models/          # saved Pipeline artifacts (.pkl)
  reports/figures/
  tests/
  requirements.txt
  README.md
  .gitignore
```
Core abstraction: a single sklearn `Pipeline` = `ColumnTransformer` (preprocess) + estimator, saved as one artifact via joblib so the Streamlit app loads the exact same transforms (no train/serve skew).

## 6. Modeling
- Baselines: DummyClassifier/DummyRegressor (floor), LogisticRegression (clf), LinearRegression (reg).
- Ensembles / boosting: RandomForest, XGBoost, LightGBM.
- Cross-validation: StratifiedKFold (clf), KFold (reg), k=5.
- Held-out test set (20%) untouched until final evaluation.
- Tuning: RandomizedSearchCV on the best booster (max_depth, n_estimators, learning_rate, subsample, etc.).

## 7. Evaluation
- Classification: ROC-AUC, PR-AUC, F1, precision, recall, accuracy, confusion matrix, ROC/PR curves.
- Regression: R², MAE, RMSE, predicted-vs-actual scatter, residual plot.
- Model comparison table across all models. Best model selected on CV PR-AUC (clf) / RMSE (reg).
- All figures saved to `reports/figures/`.

## 8. Interpretability
- SHAP on best model: global importance (bar), beeswarm summary, dependence plots for top features.
- Permutation importance as cross-check.
- Translate to plain insights (e.g. "high energy + danceability + low acousticness → higher hit probability").

## 9. Streamlit Demo
- Inputs/sliders for audio features + genre + explicit.
- Output: hit probability and predicted popularity score.
- Per-prediction SHAP (waterfall/force) explaining the result.
- Loads the saved Pipeline artifact only (no retraining at runtime).

## 10. Robustness & Error Handling
- Missing `kaggle.json` → clear error plus manual-download instructions.
- Schema validation on load: assert expected columns present; fail fast with a message.
- Processed-data cache to skip re-cleaning.
- Reproducibility: global `random_state`, pinned versions, model + preprocessor saved together.

## 11. Testing (pytest)
- Cleaning: dedup removes duplicate track_ids; no nulls post-clean; expected row/col counts.
- Features: ColumnTransformer output shape/column count correct; no NaN introduced.
- Pipeline: fit→predict smoke test on a small sample.
- Sanity: best model beats the Dummy baseline on the held-out metric.

## 12. Environment & Dependencies
- Python 3.11 in a virtualenv (`brew install python@3.11` if missing). Rationale: Python 3.14 wheels for shap/lightgbm/xgboost are unreliable.
- `requirements.txt` (pinned): pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, shap, streamlit, kaggle, joblib, jupyter, pytest.

## 13. Orchestration (agent delegation)
Per repo CLAUDE.md, the orchestrator delegates all code:
- Nova: venv setup, dependency install, Kaggle download, git ops, running scripts/tests.
- Penny: data.py, features.py, train.py, evaluate.py (multi-file pipeline).
- Luna: leak-safe ColumnTransformer design, interpret.py (SHAP), tuning logic.
- Annie: Streamlit polish, small fixes, plotting tweaks.

## 14. Out of Scope (YAGNI)
- Deep learning / embeddings.
- Live Spotify API calls (use the static dataset).
- Deployment/hosting beyond local Streamlit.
- Experiment-tracking infra (MLflow etc.).

## 15. Success Criteria
- Reproducible pipeline from raw CSV → trained model → metrics.
- Best classifier beats Dummy and LogisticRegression baselines on PR-AUC.
- SHAP explains the top drivers of popularity.
- Streamlit demo predicts and explains for user-entered features.
- All tests pass.
