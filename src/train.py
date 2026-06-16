import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split,
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
        # max_depth bounds the trees so the serialized forest stays small.
        # An unbounded RandomForest on ~90k SMOTE rows grew to ~590MB on disk
        # and exhausted RAM; capping depth + estimators keeps it well under
        # ~100MB.  n_jobs is modest (config.MODEL_N_JOBS) so it does not
        # multiply with the search-level parallelism.
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=16, min_samples_leaf=2,
                random_state=rs, n_jobs=config.MODEL_N_JOBS,
            ),
            False,
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=6,
                subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                random_state=rs, n_jobs=config.MODEL_N_JOBS,
            ),
            False,
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=16,
                random_state=rs, n_jobs=config.MODEL_N_JOBS, verbose=-1,
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
        # n_jobs is bounded (config.SEARCH_N_JOBS) instead of -1: each parallel
        # fold copies the post-SMOTE training data, so one copy per core would
        # exhaust RAM on the full ~90k-row set.
        scores = cross_val_score(pipe, X, y, cv=cv,
                                 scoring="average_precision",
                                 n_jobs=config.SEARCH_N_JOBS)
        results[name] = float(scores.mean())
    return results


def fit_model(X, y, name: str) -> Pipeline:
    """Fit a single named model on the full (X, y) training set and return the pipeline."""
    est, scale = classifier_registry()[name]
    pipe = build_pipeline(est, scale, use_smote=(name != "dummy"))
    pipe.fit(X, y)
    return pipe


# --- Hyperparameter tuning -------------------------------------------------
# Modest param distributions for the three ensembles.  Every grid is bounded so
# no draw can produce a giant, memory-hungry model:
#   - max_depth NEVER includes None/-1 (unbounded trees were the 590MB / OOM
#     culprit), and is capped at moderate depths.
#   - n_estimators is capped (<=300) so forests / boosters stay small on disk.
#   - min_samples_leaf >= 1 keeps RandomForest leaves from exploding in count.
# n_iter is kept small (see tune_model) so total runtime stays tractable.
PARAM_DISTRIBUTIONS = {
    "random_forest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [12, 16, 20],          # bounded: no None
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "xgboost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [4, 6, 8],             # bounded
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.9, 1.0],
    },
    "lightgbm": {
        "model__n_estimators": [100, 200, 300],
        "model__num_leaves": [31, 63],             # smaller -> smaller model
        "model__max_depth": [8, 12, 16],           # bounded: no -1 (unbounded)
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
    },
}

TUNABLE_MODELS = tuple(PARAM_DISTRIBUTIONS)


def _tuning_subsample(X, y):
    """Return a stratified subsample of (X, y) for hyperparameter search.

    RandomizedSearchCV refits a model for every (param x fold) combination, and
    each fit copies the (post-SMOTE) data.  Doing that on all ~90k rows is what
    exhausted RAM.  We instead search on a stratified ~TUNE_SAMPLE_SIZE-row
    sample to find good params cheaply, then refit the winner on the FULL set
    (see tune_model).  If the data is already small enough, return it unchanged.
    """
    n = len(X)
    if n <= config.TUNE_SAMPLE_SIZE:
        return X, y
    # train_test_split with stratify gives us a class-balanced subsample; we
    # keep the "train" side sized to TUNE_SAMPLE_SIZE and drop the remainder.
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=config.TUNE_SAMPLE_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return X_sub, y_sub


def tune_model(X, y, name: str, n_iter: int = 10):
    """Tune an ensemble's ``model__*`` params with RandomizedSearchCV.

    Searches over the imblearn pipeline (preprocess -> SMOTE -> model) using
    PR-AUC, so SMOTE/preprocessing stay leak-safe within each CV fold.

    Memory strategy:
      1. The SEARCH runs on a stratified subsample (config.TUNE_SAMPLE_SIZE) so
         the many refits during the search stay cheap.
      2. The winning params are then used to REFIT the estimator on the FULL
         (X, y), so the returned model is trained on all the data.
      3. n_jobs is bounded (config.SEARCH_N_JOBS), not -1, to cap parallel data
         copies.

    Returns ``(best_estimator, best_params, best_score)``.
    """
    if name not in PARAM_DISTRIBUTIONS:
        raise ValueError(f"No param distribution defined for {name!r}")
    est, scale = classifier_registry()[name]
    pipe = build_pipeline(est, scale, use_smote=True)

    # 1) Search for the best params on a small stratified subsample.
    X_search, y_search = _tuning_subsample(X, y)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_DISTRIBUTIONS[name],
        n_iter=n_iter,
        scoring="average_precision",
        cv=3,
        n_jobs=config.SEARCH_N_JOBS,
        random_state=config.RANDOM_STATE,
    )
    search.fit(X_search, y_search)

    # 2) Refit the winning configuration on the FULL training data so the final
    #    model benefits from every row (only the SEARCH was subsampled).
    best_estimator = build_pipeline(est, scale, use_smote=True)
    best_estimator.set_params(**search.best_params_)
    best_estimator.fit(X, y)

    return best_estimator, search.best_params_, float(search.best_score_)


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
