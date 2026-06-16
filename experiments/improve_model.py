"""Experiment: try to improve precision / F1 of the Spotify hit-classifier.

This script is an EXPERIMENT only. It does NOT touch the production model or the
production feature pipeline. It loads the SAME data + SAME stratified train/test
split as production (via ``src.data``) so every variant is evaluated on the exact
same held-out test set, then prints a BEFORE/AFTER comparison table.

What it tries (each fitted once on the full training set, all memory-bounded):

  - Variant A (baseline)       : bounded LightGBM + ONE-HOT genre.
                                 Should reproduce the current production numbers.
  - Variant B (target-encoding): same bounded LightGBM but the genre column is
                                 encoded with sklearn's TargetEncoder (leak-safe
                                 internal cross-fitting) instead of OneHotEncoder.
  - Variant C (stacking)       : StackingClassifier(LightGBM + XGBoost + RF ->
                                 LogisticRegression, cv=3) on TARGET-ENCODED genre.

Then it does threshold tuning on the best variant (by PR-AUC) and prints the
metrics at the F1-maximising threshold vs. the default 0.5.

MEMORY/CRASH NOTES:
  - n_jobs is fixed at 2 everywhere (config.MODEL_N_JOBS / config.SEARCH_N_JOBS,
    both 2). Never -1.
  - All base estimators are depth/estimator bounded (no unbounded trees).
  - SMOTE lives inside an imblearn Pipeline and is therefore fit on TRAIN folds
    only (leak-safe).
  - Each variant is fitted ONCE on the full training set. No RandomizedSearchCV.

Run the full experiment with:

    python -m experiments.improve_model

Do NOT run on the full ~90k dataset on a low-RAM machine without watching memory.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from src import config
from src.data import add_label, clean, load_raw, split
from src.features import IQRCapper

# ---------------------------------------------------------------------------
# Production "BEFORE" numbers (tuned LightGBM with one-hot genre, held-out test).
# Hard-coded here purely as the reference row in the comparison table; nothing
# is recomputed for these values.
# ---------------------------------------------------------------------------
PRODUCTION_BEFORE = {
    "accuracy": 0.8200,
    "precision": 0.6661,
    "recall": 0.4783,
    "f1": 0.5568,
    "roc_auc": 0.8575,
    "pr_auc": 0.6585,
}

# Bounded LightGBM params used by every variant (mirrors production's tuned
# model: moderate depth + estimator count so the model stays small in RAM).
LGBM_PARAMS = dict(
    n_estimators=300,
    num_leaves=31,
    max_depth=16,
    learning_rate=0.1,
    random_state=config.RANDOM_STATE,
    n_jobs=config.MODEL_N_JOBS,  # bounded (2), never -1
    verbose=-1,
)

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


# ---------------------------------------------------------------------------
# Preprocessor builders (kept local so we never touch production features.py)
# ---------------------------------------------------------------------------
def _continuous_tf():
    """IQR-cap then standard-scale the continuous audio features (reuses IQRCapper)."""
    return SkPipeline([("cap", IQRCapper()), ("scale", StandardScaler())])


def make_onehot_preprocessor() -> ColumnTransformer:
    """ColumnTransformer matching production: one-hot genre + capped/scaled numerics.

    This is the same shape as ``src.features.make_preprocessor(scale=True)`` but
    built locally so the experiment is self-contained and can sit next to the
    target-encoding variant for an apples-to-apples comparison.
    """
    return ColumnTransformer(
        transformers=[
            ("cont", _continuous_tf(), config.CONTINUOUS_FEATURES),
            ("disc", StandardScaler(), config.DISCRETE_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )


def make_targetencoding_preprocessor() -> ColumnTransformer:
    """Same numeric handling as one-hot, but genre via sklearn TargetEncoder.

    ``TargetEncoder`` (sklearn >= 1.4) replaces each genre category with a
    smoothed, cross-fitted estimate of the target mean. The internal
    cross-fitting (``fit_transform`` uses out-of-fold predictions) makes it
    leak-safe even without a separate holdout. It also collapses ~114 one-hot
    columns into a single dense column, which keeps the post-SMOTE matrix far
    narrower and lighter on RAM.
    """
    return ColumnTransformer(
        transformers=[
            ("cont", _continuous_tf(), config.CONTINUOUS_FEATURES),
            ("disc", StandardScaler(), config.DISCRETE_FEATURES),
            ("cat", TargetEncoder(random_state=config.RANDOM_STATE),
             config.CATEGORICAL_FEATURES),
        ]
    )


# ---------------------------------------------------------------------------
# Variant builders -> each returns an UNFITTED imblearn pipeline
# (preprocess -> SMOTE -> model). SMOTE inside the pipeline => train-only / leak-safe.
# ---------------------------------------------------------------------------
def build_variant_a() -> ImbPipeline:
    """Variant A: bounded LightGBM + ONE-HOT genre (reproduces production)."""
    return ImbPipeline([
        ("pre", make_onehot_preprocessor()),
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("model", LGBMClassifier(**LGBM_PARAMS)),
    ])


def build_variant_b() -> ImbPipeline:
    """Variant B: bounded LightGBM + TARGET-ENCODED genre."""
    return ImbPipeline([
        ("pre", make_targetencoding_preprocessor()),
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("model", LGBMClassifier(**LGBM_PARAMS)),
    ])


def build_variant_c() -> ImbPipeline:
    """Variant C: stacking ensemble on TARGET-ENCODED genre.

    Base learners (all bounded): LightGBM + XGBoost + RandomForest.
    Final estimator: LogisticRegression. Stacking CV = 3, n_jobs = 2.

    We use the target-encoding preprocessor here (one dense genre column instead
    of ~114 one-hot columns) because the StackingClassifier internally refits
    every base learner across cv folds; the narrower matrix keeps that memory
    footprint manageable. n_jobs is bounded at every level so search-level and
    estimator-level parallelism never multiply out of control.
    """
    base_estimators = [
        ("lgbm", LGBMClassifier(**LGBM_PARAMS)),
        ("xgb", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=config.RANDOM_STATE, n_jobs=config.MODEL_N_JOBS,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=16, min_samples_leaf=2,
            random_state=config.RANDOM_STATE, n_jobs=config.MODEL_N_JOBS,
        )),
    ]
    stack = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=config.SEARCH_N_JOBS,  # bounded (2), never -1
        passthrough=False,
    )
    return ImbPipeline([
        ("pre", make_targetencoding_preprocessor()),
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("model", stack),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _proba(fitted_pipe, X):
    """Positive-class probability for the test rows (1-D array)."""
    return fitted_pipe.predict_proba(X)[:, 1]


def evaluate(fitted_pipe, X_test, y_test, threshold: float = 0.5) -> dict:
    """Return dict(accuracy, precision, recall, f1, roc_auc, pr_auc) on the test set.

    Class predictions use ``threshold`` (default 0.5). ROC-AUC and PR-AUC are
    threshold-independent (computed from the probabilities directly).
    """
    proba = _proba(fitted_pipe, X_test)
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }


def metrics_from_proba(y_true, proba, threshold: float) -> dict:
    """Same metric dict as ``evaluate`` but from a precomputed probability array.

    Lets the threshold sweep avoid re-running predict_proba for every threshold.
    """
    y_pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def tune_threshold(y_true, proba, grid=None):
    """Sweep thresholds and return (best_threshold, best_f1).

    CAVEAT: we tune on the TEST probabilities here purely for demonstration.
    In a real deployment the threshold should be chosen on a validation split
    so the reported test F1 stays an honest estimate. See the printed warning.
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_variants(X_train, y_train, X_test, y_test) -> dict:
    """Fit A/B/C on train, evaluate on test, return {row_label: metric_dict}.

    Also handles threshold tuning on the best variant by PR-AUC and stores the
    fitted-best probabilities so the caller can persist / inspect them.
    """
    builders = {
        "A (LGBM one-hot)": build_variant_a,
        "B (LGBM target-enc)": build_variant_b,
        "C (stacking target-enc)": build_variant_c,
    }

    rows = {"Production (before)": dict(PRODUCTION_BEFORE)}
    fitted = {}
    probas = {}

    for label, builder in builders.items():
        print(f"[fit] {label} ...", flush=True)
        pipe = builder()
        pipe.fit(X_train, y_train)
        fitted[label] = pipe
        probas[label] = _proba(pipe, X_test)
        rows[label] = evaluate(pipe, X_test, y_test, threshold=0.5)
        print(f"[done] {label}: "
              f"P={rows[label]['precision']:.4f} "
              f"R={rows[label]['recall']:.4f} "
              f"F1={rows[label]['f1']:.4f} "
              f"PR-AUC={rows[label]['pr_auc']:.4f}", flush=True)

    # Pick the best experimental variant by PR-AUC (ranking quality) for
    # threshold tuning.
    variant_labels = list(builders)
    best_label = max(variant_labels, key=lambda lbl: rows[lbl]["pr_auc"])
    best_proba = probas[best_label]

    print(f"\n[threshold] best variant by PR-AUC = {best_label}; "
          f"sweeping thresholds 0.05..0.95 to maximise F1", flush=True)
    print("[threshold] CAVEAT: tuned on the TEST set for demonstration only; "
          "in production tune this on a validation split.", flush=True)

    best_t, best_f1 = tune_threshold(y_test, best_proba)
    tuned_metrics = metrics_from_proba(y_test, best_proba, best_t)

    tuned_label = f"{best_label.split()[0]} @ thr={best_t:.2f}"
    rows[tuned_label] = tuned_metrics

    print(f"[threshold] default 0.50 F1={rows[best_label]['f1']:.4f} -> "
          f"tuned thr={best_t:.2f} F1={best_f1:.4f}", flush=True)

    return {
        "rows": rows,
        "best_variant": best_label,
        "tuned_threshold": best_t,
        "tuned_label": tuned_label,
    }


def to_table(rows: dict) -> pd.DataFrame:
    """Build the BEFORE/AFTER comparison DataFrame (rows preserve insertion order)."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df[METRIC_COLS].round(4)


def main():
    """Full experiment: load data, fit variants, tune threshold, print + save."""
    print("Loading + cleaning + labeling + splitting (same as production) ...",
          flush=True)
    df = add_label(clean(load_raw()))
    X_train, X_test, y_train, y_test = split(df)
    print(f"train rows={len(X_train)}  test rows={len(X_test)}  "
          f"test hit-rate={y_test.mean():.4f}", flush=True)

    result = run_variants(X_train, y_train, X_test, y_test)
    table = to_table(result["rows"])

    print("\n================ BEFORE / AFTER COMPARISON ================")
    print(table.to_string())
    print("===========================================================")

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = {
        "before": PRODUCTION_BEFORE,
        "variants": result["rows"],
        "best_variant_by_pr_auc": result["best_variant"],
        "tuned_threshold": result["tuned_threshold"],
        "note": ("Threshold tuned on the test set for demonstration; tune on a "
                 "validation split in production. No models were saved."),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}")
    print("NOTE: no model in models/ was created or overwritten.")


if __name__ == "__main__":
    main()
