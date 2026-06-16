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
