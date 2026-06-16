import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src import config, train


def _dataset(n=240):
    rng = np.random.RandomState(0)
    base = {f: rng.normal(size=n) for f in config.NUMERIC_FEATURES}
    df = pd.DataFrame(base)
    df["track_genre"] = np.where(np.arange(n) % 2 == 0, "pop", "rock")
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
