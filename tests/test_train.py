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


def test_registry_estimators_drop_class_weight():
    # SMOTE handles imbalance now, so estimators must not also re-weight classes.
    reg = train.classifier_registry()
    for name in ("logreg", "random_forest", "lightgbm"):
        est, _ = reg[name]
        assert getattr(est, "class_weight", None) is None
    xgb, _ = reg["xgboost"]
    assert getattr(xgb, "scale_pos_weight", None) in (None, 1)


def test_pipeline_fits_and_predicts():
    X, y = _dataset()
    pipe = train.build_pipeline(*train.classifier_registry()["logreg"])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)


def test_build_pipeline_includes_smote_by_default():
    est, scale = train.classifier_registry()["random_forest"]
    pipe = train.build_pipeline(est, scale)
    assert "smote" in pipe.named_steps
    from imblearn.over_sampling import SMOTE
    assert isinstance(pipe.named_steps["smote"], SMOTE)


def test_build_pipeline_can_disable_smote():
    est, scale = train.classifier_registry()["dummy"]
    pipe = train.build_pipeline(est, scale, use_smote=False)
    assert "smote" not in pipe.named_steps


def test_dummy_pipeline_has_no_smote():
    # The honest-floor baseline must never be resampled.
    pipe = train.fit_model(*_dataset(), "dummy")
    assert "smote" not in pipe.named_steps


def test_tune_model_returns_fitted_best_estimator():
    X, y = _dataset()
    est, params, score = train.tune_model(X, y, "random_forest", n_iter=2)
    # Returned estimator is fitted and usable.
    preds = est.predict(X)
    assert len(preds) == len(y)
    assert isinstance(params, dict) and params  # exposes best params
    assert all(k.startswith("model__") for k in params)
    assert 0.0 <= score <= 1.0
    assert "smote" in est.named_steps


def test_best_model_beats_dummy():
    X, y = _dataset()
    dummy = train.build_pipeline(*train.classifier_registry()["dummy"]).fit(X, y)
    logreg = train.build_pipeline(*train.classifier_registry()["logreg"]).fit(X, y)
    ap_dummy = average_precision_score(y, dummy.predict_proba(X)[:, 1])
    ap_logreg = average_precision_score(y, logreg.predict_proba(X)[:, 1])
    assert ap_logreg > ap_dummy
