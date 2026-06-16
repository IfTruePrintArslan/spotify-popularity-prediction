# Spotify Song Popularity Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, portfolio-grade ML pipeline that predicts Spotify track popularity from audio features — primarily hit-vs-flop classification, secondarily 0–100 regression — with SHAP interpretability and a Streamlit demo.

**Architecture:** A single scikit-learn `Pipeline` (`ColumnTransformer` preprocessing + estimator) is the core unit; it is cross-validated, the best model is tuned and persisted as one joblib artifact, then reused unchanged by evaluation, interpretation, and the Streamlit app (no train/serve skew). Code is split by responsibility under `src/` with TDD on the data/feature/train core.

**Tech Stack:** Python 3.11 (venv), pandas, numpy, scikit-learn, xgboost, lightgbm, shap, matplotlib/seaborn, streamlit, kaggle, joblib, pytest.

**Spec:** `docs/superpowers/specs/2026-06-16-spotify-popularity-design.md`

**Pinned design decisions (resolving spec latitude):**
- Dedup strategy: drop duplicate `track_id`, `keep="first"`.
- `track_genre` encoding: `OneHotEncoder(handle_unknown="ignore")` (one-hot), not target encoding.
- `explicit`: cast bool → int and treat as a numeric/passthrough feature.
- Imbalance: `class_weight="balanced"` on logreg/RF/LightGBM; XGBoost left default (scale_pos_weight is a tuning lever). Model selection on CV PR-AUC (`average_precision`).

**Dataset note:** Kaggle `maharshipandya/spotify-tracks-dataset` unzips to `data/raw/dataset.csv` with columns: `Unnamed: 0, track_id, artists, album_name, track_name, popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature, track_genre`.

**Orchestration (per repo CLAUDE.md):** Nova = env/data/git/test-runs; Penny = multi-file pipeline (data/features/train/evaluate); Luna = leak-safe ColumnTransformer + interpret/tuning; Annie = Streamlit + plot polish. Each task notes a suggested owner.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `requirements.txt` | Pinned dependencies |
| `conftest.py` | Put project root on `sys.path` so `from src import ...` works in tests |
| `src/__init__.py` | Marks `src` a package |
| `src/config.py` | Constants: paths, RANDOM_STATE, feature lists, threshold |
| `src/data.py` | Load, schema-check, clean/dedup, label, split |
| `src/features.py` | `make_preprocessor(scale)` → ColumnTransformer |
| `src/train.py` | Model registry, pipeline factory, CV, fit-best, save |
| `src/evaluate.py` | Classification + regression metrics, plots, comparison table |
| `src/interpret.py` | SHAP + permutation importance |
| `src/run_pipeline.py` | End-to-end CLI: load→clean→split→cv→fit→eval→save |
| `tests/test_data.py` | Cleaning/label/split tests |
| `tests/test_features.py` | Preprocessor output tests |
| `tests/test_train.py` | Pipeline smoke + beats-dummy |
| `notebooks/01_eda.ipynb` | Exploratory analysis |
| `app/streamlit_app.py` | Interactive demo |
| `README.md` | Setup + usage |

---

## Task 1: Environment & dependencies

**Owner:** Nova

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Ensure Python 3.11 is available**

Run: `python3.11 --version`
Expected: `Python 3.11.x`. If "command not found": `brew install python@3.11` then retry.

- [ ] **Step 2: Create and activate a venv**

```bash
cd "/Users/arslan/Desktop/ML PROJECT"
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # expect 3.11.x
```

- [ ] **Step 3: Write `requirements.txt`**

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
xgboost==2.1.1
lightgbm==4.5.0
shap==0.46.0
matplotlib==3.9.2
seaborn==0.13.2
streamlit==1.38.0
kaggle==1.6.17
joblib==1.4.2
jupyter==1.1.1
pytest==8.3.2
```

- [ ] **Step 4: Install**

Run: `pip install --upgrade pip && pip install -r requirements.txt`
Expected: all install without error. If a pinned version has no 3.11 wheel, bump to the latest compatible patch and note it.

- [ ] **Step 5: Verify imports**

Run: `python -c "import pandas, sklearn, xgboost, lightgbm, shap, streamlit, joblib; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin Python 3.11 dependencies"
```

---

## Task 2: Project scaffolding & config

**Owner:** Penny

**Files:**
- Create: `src/__init__.py`, `conftest.py`, `src/config.py`

- [ ] **Step 1: Create package markers**

`src/__init__.py` — empty file.

`conftest.py` (project root):
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
```

- [ ] **Step 2: Write `src/config.py`**

```python
from pathlib import Path

RANDOM_STATE = 42
HIT_THRESHOLD = 50
TEST_SIZE = 0.2
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RAW_CSV = DATA_RAW / "dataset.csv"
MODEL_PATH = MODELS_DIR / "best_classifier.joblib"

TARGET = "popularity"
LABEL = "hit"

NUMERIC_FEATURES = [
    "duration_ms", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature", "explicit",
]
CATEGORICAL_FEATURES = ["track_genre"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DROP_COLS = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Verify it imports and creates dirs**

Run: `python -c "from src import config; print(config.ROOT)"`
Expected: prints the project root path; `data/`, `models/`, `reports/figures/` now exist.

- [ ] **Step 4: Commit**

```bash
git add src/__init__.py conftest.py src/config.py
git commit -m "feat: add project config and package scaffolding"
```

---

## Task 3: Data loading, cleaning, labeling, splitting (TDD)

**Owner:** Penny

**Files:**
- Create: `src/data.py`, `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:
```python
import pandas as pd
import pytest
from src import config, data


def _raw_row(track_id="a", popularity=10, genre="pop"):
    return {
        "Unnamed: 0": 0, "track_id": track_id, "artists": "x",
        "album_name": "y", "track_name": "z", "popularity": popularity,
        "duration_ms": 200000, "explicit": False, "danceability": 0.5,
        "energy": 0.5, "key": 1, "loudness": -5.0, "mode": 1,
        "speechiness": 0.05, "acousticness": 0.1, "instrumentalness": 0.0,
        "liveness": 0.1, "valence": 0.5, "tempo": 120.0,
        "time_signature": 4, "track_genre": genre,
    }


def test_clean_drops_duplicate_track_ids():
    df = pd.DataFrame([_raw_row("a", genre="pop"), _raw_row("a", genre="rock")])
    cleaned = data.clean(df)
    assert len(cleaned) == 1


def test_clean_drops_identifier_columns():
    df = pd.DataFrame([_raw_row("a"), _raw_row("b")])
    cleaned = data.clean(df)
    for col in config.DROP_COLS:
        assert col not in cleaned.columns


def test_clean_casts_explicit_to_int():
    df = pd.DataFrame([_raw_row("a"), _raw_row("b")])
    cleaned = data.clean(df)
    assert set(cleaned["explicit"].unique()) <= {0, 1}


def test_clean_has_no_nulls():
    df = pd.DataFrame([_raw_row("a"), _raw_row("b")])
    cleaned = data.clean(df)
    assert cleaned.isna().sum().sum() == 0


def test_add_label_thresholds_popularity():
    df = pd.DataFrame([_raw_row("a", popularity=49), _raw_row("b", popularity=50)])
    labeled = data.add_label(data.clean(df))
    assert labeled[config.LABEL].tolist() == [0, 1]


def test_load_raw_rejects_bad_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        data.load_raw(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL (module `src.data` has no `clean`/`add_label`/`load_raw`).

- [ ] **Step 3: Write `src/data.py`**

```python
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

EXPECTED_COLS = (
    set(config.DROP_COLS)
    | set(config.NUMERIC_FEATURES)
    | set(config.CATEGORICAL_FEATURES)
    | {config.TARGET}
)


def load_raw(path=None) -> pd.DataFrame:
    path = Path(path) if path else config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it first "
            f"(see README: Kaggle 'maharshipandya/spotify-tracks-dataset')."
        )
    df = pd.read_csv(path)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="track_id", keep="first").copy()
    keep = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET]
    df["explicit"] = df["explicit"].astype(int)
    df = df.dropna(subset=keep)
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    return df.reset_index(drop=True)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.LABEL] = (df[config.TARGET] >= config.HIT_THRESHOLD).astype(int)
    return df


def split(df: pd.DataFrame, target: str = config.LABEL, stratify: bool = True):
    X = df[config.ALL_FEATURES]
    y = df[target]
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=strat,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add data loading, cleaning, labeling and split"
```

---

## Task 4: Feature preprocessing (TDD)

**Owner:** Luna (leakage-sensitive)

**Files:**
- Create: `src/features.py`, `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:
```python
import numpy as np
import pandas as pd
from src import config, features


def _frame(n=6):
    rows = []
    for i in range(n):
        rows.append({f: float(i) for f in config.NUMERIC_FEATURES})
        rows[-1]["track_genre"] = "pop" if i % 2 else "rock"
    return pd.DataFrame(rows)[config.ALL_FEATURES]


def test_preprocessor_preserves_row_count():
    X = _frame()
    out = features.make_preprocessor(scale=True).fit_transform(X)
    assert out.shape[0] == len(X)


def test_preprocessor_no_nan():
    X = _frame()
    out = features.make_preprocessor(scale=False).fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    assert not np.isnan(out).any()


def test_scaling_changes_numeric_values():
    X = _frame()
    scaled = features.make_preprocessor(scale=True).fit_transform(X)
    raw = features.make_preprocessor(scale=False).fit_transform(X)
    scaled = scaled.toarray() if hasattr(scaled, "toarray") else np.asarray(scaled)
    raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    assert not np.allclose(scaled[:, 0], raw[:, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL (`src.features` has no `make_preprocessor`).

- [ ] **Step 3: Write `src/features.py`**

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def make_preprocessor(scale: bool) -> ColumnTransformer:
    numeric_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add leak-safe ColumnTransformer preprocessor"
```

---

## Task 5: Model registry, pipeline factory, CV (TDD)

**Owner:** Penny

**Files:**
- Create: `src/train.py`, `tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:
```python
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src import config, train


def _dataset(n=240):
    rng = np.random.RandomState(0)
    base = {f: rng.normal(size=n) for f in config.NUMERIC_FEATURES}
    df = pd.DataFrame(base)
    df["track_genre"] = np.where(np.arange(n) % 2 == 0, "pop", "rock")
    # signal: energy drives the label
    y = (df["energy"] + rng.normal(scale=0.1, size=n) > 0).astype(int)
    X = df[config.ALL_FEATURES]
    return X, pd.Series(y, name=config.LABEL)


def test_registry_contains_expected_models():
    reg = train.classifier_registry()
    assert {"dummy", "logreg", "random_forest", "xgboost", "lightgbm"} <= set(reg)


def test_pipeline_fits_and_predicts():
    X, y = _dataset()
    pipe = train.build_pipeline(*train.classifier_registry()["logreg"])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_best_model_beats_dummy():
    X, y = _dataset()
    dummy = train.build_pipeline(*train.classifier_registry()["dummy"]).fit(X, y)
    logreg = train.build_pipeline(*train.classifier_registry()["logreg"]).fit(X, y)
    ap_dummy = average_precision_score(y, dummy.predict_proba(X)[:, 1])
    ap_logreg = average_precision_score(y, logreg.predict_proba(X)[:, 1])
    assert ap_logreg > ap_dummy
```

Note: `DummyClassifier(strategy="most_frequent")` supports `predict_proba`, so the test is valid.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py -v`
Expected: FAIL (`src.train` undefined functions).

- [ ] **Step 3: Write `src/train.py`**

```python
import joblib
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config
from src.features import make_preprocessor

# name -> (estimator, needs_scaling)
def classifier_registry():
    rs = config.RANDOM_STATE
    return {
        "dummy": (DummyClassifier(strategy="most_frequent"), False),
        "logreg": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=rs, n_jobs=-1,
            ),
            False,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=rs, n_jobs=-1,
            ),
            False,
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=400, learning_rate=0.05, class_weight="balanced",
                random_state=rs, n_jobs=-1, verbose=-1,
            ),
            False,
        ),
    }


def build_pipeline(estimator, needs_scaling: bool) -> Pipeline:
    return Pipeline([
        ("pre", make_preprocessor(scale=needs_scaling)),
        ("model", estimator),
    ])


def cross_validate_models(X, y) -> dict:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    results = {}
    for name, (est, scale) in classifier_registry().items():
        pipe = build_pipeline(est, scale)
        scores = cross_val_score(pipe, X, y, cv=cv,
                                 scoring="average_precision", n_jobs=-1)
        results[name] = float(scores.mean())
    return results


def fit_model(X, y, name: str) -> Pipeline:
    est, scale = classifier_registry()[name]
    pipe = build_pipeline(est, scale)
    pipe.fit(X, y)
    return pipe


def save_pipeline(pipe: Pipeline, path=None) -> None:
    joblib.dump(pipe, path or config.MODEL_PATH)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: add model registry, pipeline factory and CV"
```

---

## Task 6: Evaluation metrics & plots

**Owner:** Penny

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 1: Write `src/evaluate.py`**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src import config


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def save_classification_plots(estimator, X_test, y_test, prefix="best"):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for disp, name in (
        (ConfusionMatrixDisplay.from_estimator, "confusion_matrix"),
        (RocCurveDisplay.from_estimator, "roc_curve"),
        (PrecisionRecallDisplay.from_estimator, "pr_curve"),
    ):
        ax = disp(estimator, X_test, y_test).ax_
        ax.figure.savefig(config.FIGURES_DIR / f"{prefix}_{name}.png", bbox_inches="tight")
        plt.close(ax.figure)


def comparison_table(cv_results: dict):
    import pandas as pd
    return (
        pd.Series(cv_results, name="cv_pr_auc")
        .sort_values(ascending=False)
        .to_frame()
    )
```

- [ ] **Step 2: Smoke-check it imports**

Run: `python -c "from src import evaluate; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat: add classification metrics and evaluation plots"
```

---

## Task 7: SHAP & permutation interpretability

**Owner:** Luna

**Files:**
- Create: `src/interpret.py`

- [ ] **Step 1: Write `src/interpret.py`**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance

from src import config


def _transform(pipeline, X):
    pre = pipeline.named_steps["pre"]
    Xt = pre.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = pre.get_feature_names_out()
    return Xt, names


def shap_summary(pipeline, X_sample, prefix="best"):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Xt, names = _transform(pipeline, X_sample)
    model = pipeline.named_steps["model"]
    explainer = shap.Explainer(model, Xt, feature_names=names)
    values = explainer(Xt)
    shap.summary_plot(values, Xt, feature_names=names, show=False)
    plt.savefig(config.FIGURES_DIR / f"{prefix}_shap_summary.png", bbox_inches="tight")
    plt.close()
    return values


def permutation_importance_table(pipeline, X_test, y_test):
    import pandas as pd
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="average_precision",
        n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.Series(result.importances_mean, index=X_test.columns, name="perm_importance")
        .sort_values(ascending=False)
        .to_frame()
    )
```

Note: `shap.Explainer` auto-selects `TreeExplainer` for RF/XGBoost/LightGBM and a linear/permutation explainer otherwise. For tree models, passing the transformed background matrix is correct.

- [ ] **Step 2: Smoke-check it imports**

Run: `python -c "from src import interpret; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/interpret.py
git commit -m "feat: add SHAP and permutation importance"
```

---

## Task 8: End-to-end pipeline runner

**Owner:** Penny

**Files:**
- Create: `src/run_pipeline.py`

- [ ] **Step 1: Write `src/run_pipeline.py`**

```python
"""Run the full classification pipeline: load -> clean -> split -> CV -> fit best -> evaluate -> save."""
from src import config, data, evaluate, interpret, train


def main():
    df = data.add_label(data.clean(data.load_raw()))
    X_train, X_test, y_train, y_test = data.split(df, target=config.LABEL)

    cv_results = train.cross_validate_models(X_train, y_train)
    table = evaluate.comparison_table(cv_results)
    print("\nCV PR-AUC by model:\n", table)

    best_name = table.index[0]
    if best_name == "dummy":  # guard: never ship the floor model
        best_name = table.index[1]
    print(f"\nBest model: {best_name}")

    best = train.fit_model(X_train, y_train, best_name)
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = evaluate.classification_metrics(y_test, y_pred, y_proba)
    print("\nHeld-out test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    evaluate.save_classification_plots(best, X_test, y_test)
    interpret.shap_summary(best, X_test.sample(min(2000, len(X_test)),
                                               random_state=config.RANDOM_STATE))
    print("\nPermutation importance:\n",
          interpret.permutation_importance_table(best, X_test, y_test).head(10))

    train.save_pipeline(best)
    print(f"\nSaved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check it imports (no data needed)**

Run: `python -c "import src.run_pipeline; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/run_pipeline.py
git commit -m "feat: add end-to-end pipeline runner"
```

---

## Task 9: Acquire dataset & run the pipeline

**Owner:** Nova

**Files:** none (produces `data/raw/dataset.csv`, `models/best_classifier.joblib`, `reports/figures/*`)

> **Credential gate:** Kaggle API needs `~/.kaggle/kaggle.json` (chmod 600). If absent, STOP and ask the user to either place the token or manually download the dataset CSV to `data/raw/dataset.csv`.

- [ ] **Step 1: Confirm Kaggle credentials**

Run: `test -f ~/.kaggle/kaggle.json && echo present || echo MISSING`
If `MISSING`: pause and request the token (or manual CSV download), then continue.

- [ ] **Step 2: Download + unzip dataset**

```bash
source .venv/bin/activate
kaggle datasets download -d maharshipandya/spotify-tracks-dataset -p data/raw --unzip
ls -la data/raw   # expect dataset.csv
```

- [ ] **Step 3: Run the full pipeline**

Run: `python -m src.run_pipeline`
Expected: prints CV table, best model, held-out metrics (best beats dummy on PR-AUC), writes figures to `reports/figures/` and `models/best_classifier.joblib`.

- [ ] **Step 4: Verify artifacts exist**

Run: `ls models/ reports/figures/`
Expected: `best_classifier.joblib` and several `.png` files.

- [ ] **Step 5: Commit (code/figures only — data & model are gitignored)**

```bash
git add -A
git commit -m "chore: run pipeline; record evaluation artifacts" || echo "nothing to commit (artifacts gitignored)"
```

---

## Task 10: EDA notebook

**Owner:** Annie

**Files:**
- Create: `notebooks/01_eda.ipynb`

- [ ] **Step 1: Create the notebook with these cells**

Cell 1 (markdown): `# Spotify Popularity — EDA`

Cell 2 (code):
```python
import sys; sys.path.insert(0, "..")
import seaborn as sns, matplotlib.pyplot as plt
from src import config, data
df = data.add_label(data.clean(data.load_raw()))
df.shape
```

Cell 3 (code) — target distribution:
```python
sns.histplot(df[config.TARGET], bins=50); plt.title("Popularity distribution"); plt.show()
print("hit rate:", df[config.LABEL].mean())
```

Cell 4 (code) — feature correlations with popularity:
```python
corr = df[config.NUMERIC_FEATURES + [config.TARGET]].corr()[config.TARGET].sort_values()
corr
```

Cell 5 (code) — correlation heatmap:
```python
plt.figure(figsize=(10, 8)); sns.heatmap(df[config.NUMERIC_FEATURES].corr(), cmap="coolwarm", center=0); plt.show()
```

Cell 6 (code) — popularity by genre (top 15):
```python
top = df.groupby("track_genre")[config.TARGET].mean().sort_values(ascending=False).head(15)
top.plot(kind="barh"); plt.title("Mean popularity by genre"); plt.show()
```

- [ ] **Step 2: Execute the notebook top-to-bottom**

Run: `jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb`
Expected: executes without error (requires `data/raw/dataset.csv`).

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda.ipynb
git commit -m "docs: add EDA notebook"
```

---

## Task 11: Streamlit demo

**Owner:** Annie

**Files:**
- Create: `app/streamlit_app.py`

- [ ] **Step 1: Write `app/streamlit_app.py`**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎵")
st.title("🎵 Spotify Hit Predictor")

if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.run_pipeline` first.")
    st.stop()

model = joblib.load(config.MODEL_PATH)

st.sidebar.header("Audio features")
inputs = {
    "danceability": st.sidebar.slider("Danceability", 0.0, 1.0, 0.6),
    "energy": st.sidebar.slider("Energy", 0.0, 1.0, 0.7),
    "loudness": st.sidebar.slider("Loudness (dB)", -60.0, 2.0, -6.0),
    "speechiness": st.sidebar.slider("Speechiness", 0.0, 1.0, 0.05),
    "acousticness": st.sidebar.slider("Acousticness", 0.0, 1.0, 0.2),
    "instrumentalness": st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0),
    "liveness": st.sidebar.slider("Liveness", 0.0, 1.0, 0.15),
    "valence": st.sidebar.slider("Valence", 0.0, 1.0, 0.5),
    "tempo": st.sidebar.slider("Tempo (BPM)", 40.0, 220.0, 120.0),
    "duration_ms": st.sidebar.slider("Duration (ms)", 30000, 600000, 200000),
    "key": st.sidebar.slider("Key", 0, 11, 1),
    "mode": st.sidebar.selectbox("Mode", [0, 1], index=1),
    "time_signature": st.sidebar.selectbox("Time signature", [3, 4, 5], index=1),
    "explicit": int(st.sidebar.checkbox("Explicit", value=False)),
    "track_genre": st.sidebar.selectbox("Genre", ["pop", "rock", "hip-hop", "edm", "classical", "jazz"]),
}

row = pd.DataFrame([inputs])[config.ALL_FEATURES]
proba = float(model.predict_proba(row)[0, 1])
st.metric("Hit probability", f"{proba:.1%}")
st.progress(proba)
st.caption(f"'Hit' = popularity ≥ {config.HIT_THRESHOLD}. Model: {config.MODEL_PATH.name}")
```

- [ ] **Step 2: Launch and verify (requires a trained model from Task 9)**

Run: `streamlit run app/streamlit_app.py`
Expected: app opens; moving sliders updates the hit probability. Verify golden path (defaults show a probability) and the no-model error path (rename the model file, confirm the error message, restore it).

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: add Streamlit hit-predictor demo"
```

---

## Task 12: README

**Owner:** Annie

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** documenting: project summary; setup (`python3.11 -m venv .venv`, activate, `pip install -r requirements.txt`); Kaggle credential setup (`~/.kaggle/kaggle.json`, chmod 600) AND the manual-download fallback to `data/raw/dataset.csv`; how to run (`python -m src.run_pipeline`); how to test (`pytest -v`); how to launch the demo (`streamlit run app/streamlit_app.py`); results summary placeholder to fill after Task 9; the hit-threshold definition.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add project README"
```

---

## Task 13: Optional — regression variant & tuning

**Owner:** Luna (only if time permits; classification is the headline deliverable)

- [ ] Add `regressor_registry()` (LinearRegression, RandomForestRegressor, XGBRegressor, LGBMRegressor) and a `regression_metrics()` (R², MAE, RMSE) mirroring the classification path, selected on CV RMSE. Reuse `make_preprocessor`.
- [ ] Add `RandomizedSearchCV` tuning for the best booster (param grids: `n_estimators`, `max_depth`/`num_leaves`, `learning_rate`, `subsample`); refit and compare against the untuned model on held-out PR-AUC before replacing the saved artifact.
- [ ] Commit each as its own change.

---

## Self-Review (author checklist)

**Spec coverage:** targets §2 → Tasks 3,5,13; dataset §3 → Task 9; cleaning/leakage §4 → Tasks 3,4; structure §5 → Tasks 2–12; modeling §6 → Tasks 5,13; evaluation §7 → Task 6; interpretability §8 → Task 7; demo §9 → Task 11; robustness §10 → Tasks 3 (schema/FileNotFound), 8 (dummy guard), 9 (cred gate); testing §11 → Tasks 3,4,5; environment §12 → Task 1; success criteria §15 → Tasks 5 (beats dummy), 7 (SHAP), 9 (artifacts), 11 (demo). No gaps.

**Placeholder scan:** no TBD/TODO; every code step shows complete code. (Task 13 is explicitly optional/stretch, not a placeholder for required work.)

**Type consistency:** `make_preprocessor(scale=...)` signature consistent across features/train/interpret; registry tuples `(estimator, needs_scaling)` consumed consistently by `build_pipeline`; `config.LABEL`/`config.ALL_FEATURES`/`config.MODEL_PATH` used uniformly; `predict_proba` available on every registry model including `DummyClassifier(strategy="most_frequent")`.
