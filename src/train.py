import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score,
)
from xgboost import XGBClassifier

from src import config
from src.features import make_preprocessor


def classifier_registry():
    """Return a dict of {name: (estimator, needs_scaling)} for every candidate model.

    needs_scaling is True only for Logistic Regression, which is sensitive to
    feature scale.  Tree-based models (Random Forest, XGBoost, LightGBM) are
    scale-invariant, so their flag is False to skip the StandardScaler step and
    speed up the pipeline.
    """
    rs = config.RANDOM_STATE
    # SMOTE now handles class imbalance inside the pipeline, so the estimators
    # no longer use class_weight="balanced" / scale_pos_weight.
    return {
        "dummy": (DummyClassifier(strategy="most_frequent"), False),
        "logreg": (LogisticRegression(max_iter=1000), True),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300, random_state=rs, n_jobs=-1,
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
                n_estimators=400, learning_rate=0.05,
                random_state=rs, n_jobs=-1, verbose=-1,
            ),
            False,
        ),
    }


def build_pipeline(estimator, needs_scaling: bool, use_smote: bool = True) -> Pipeline:
    """Preprocess -> (SMOTE) -> model.

    SMOTE is fit only on the training folds via the imblearn pipeline, so it is
    leak-safe. ``use_smote=False`` is used for the dummy baseline so it stays an
    honest floor (and is harmless for estimators that don't need resampling).
    """
    steps = [("pre", make_preprocessor(scale=needs_scaling))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def cross_validate_models(X, y) -> dict:
    """Run stratified k-fold CV for every model and return {name: mean_pr_auc}.

    PR-AUC (average precision) is used as the scoring metric because the dataset
    is class-imbalanced — accuracy would be misleading (a model that always
    predicts 'not a hit' scores ~50 % accuracy for free).  PR-AUC rewards the
    model for correctly ranking actual hits at the top.
    """
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    results = {}
    for name, (est, scale) in classifier_registry().items():
        # Dummy stays SMOTE-free so it remains an honest baseline floor.
        pipe = build_pipeline(est, scale, use_smote=(name != "dummy"))
        scores = cross_val_score(pipe, X, y, cv=cv,
                                 scoring="average_precision", n_jobs=-1)
        results[name] = float(scores.mean())
    return results


def fit_model(X, y, name: str) -> Pipeline:
    """Fit a single named model on the full (X, y) training set and return the pipeline."""
    est, scale = classifier_registry()[name]
    pipe = build_pipeline(est, scale, use_smote=(name != "dummy"))
    pipe.fit(X, y)
    return pipe


# --- Hyperparameter tuning -------------------------------------------------
# Modest param distributions for the three ensembles. n_iter is kept small
# (see tune_model) so total runtime on the ~90k-row dataset stays tractable.
PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "model__n_estimators": [200, 300, 400, 600],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "xgboost": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    },
    "lightgbm": {
        "model__n_estimators": [200, 400, 600],
        "model__num_leaves": [31, 63, 127],
        "model__max_depth": [-1, 10, 20],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
    },
}

TUNABLE_MODELS = tuple(PARAM_DISTRIBUTIONS)


def tune_model(X, y, name: str, n_iter: int = 10):
    """Tune an ensemble's ``model__*`` params with RandomizedSearchCV.

    Searches over the imblearn pipeline (preprocess -> SMOTE -> model) using
    PR-AUC, so SMOTE/preprocessing stay leak-safe within each CV fold. Returns
    ``(best_estimator, best_params, best_score)``.
    """
    if name not in PARAM_DISTRIBUTIONS:
        raise ValueError(f"No param distribution defined for {name!r}")
    est, scale = classifier_registry()[name]
    pipe = build_pipeline(est, scale, use_smote=True)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_DISTRIBUTIONS[name],
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def tune_ensembles(X, y, names=TUNABLE_MODELS, n_iter: int = 10) -> dict:
    """Tune each ensemble; return {name: (estimator, params, cv_pr_auc)}."""
    tuned = {}
    for name in names:
        tuned[name] = tune_model(X, y, name, n_iter=n_iter)
    return tuned


def best_tuned_model(tuned: dict):
    """Pick the (name, estimator, params, score) with the highest CV PR-AUC."""
    name = max(tuned, key=lambda n: tuned[n][2])
    est, params, score = tuned[name]
    return name, est, params, score


def save_pipeline(pipe: Pipeline, path=None) -> None:
    """Serialize the fitted pipeline to disk with joblib (defaults to MODEL_PATH)."""
    joblib.dump(pipe, path or config.MODEL_PATH)
