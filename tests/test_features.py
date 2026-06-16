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


def test_iqr_capper_clips_extreme_value_to_upper_bound():
    rng = np.random.RandomState(0)
    col = rng.normal(size=200).reshape(-1, 1)
    capper = features.IQRCapper().fit(col)
    extreme = np.array([[1e6]])  # far above any plausible upper fence
    out = capper.transform(extreme)
    assert out[0, 0] == capper.upper_[0]
    assert out[0, 0] < 1e6


def test_iqr_capper_clips_extreme_value_to_lower_bound():
    rng = np.random.RandomState(1)
    col = rng.normal(size=200).reshape(-1, 1)
    capper = features.IQRCapper().fit(col)
    out = capper.transform(np.array([[-1e6]]))
    assert out[0, 0] == capper.lower_[0]


def test_iqr_capper_leaves_inlier_unchanged():
    col = np.arange(100, dtype=float).reshape(-1, 1)
    capper = features.IQRCapper().fit(col)
    median = np.array([[50.0]])
    assert capper.transform(median)[0, 0] == 50.0


def test_iqr_capper_accepts_dataframe_and_returns_array():
    df = pd.DataFrame({"a": list(range(50)), "b": list(range(50))})
    out = features.IQRCapper().fit_transform(df)
    assert isinstance(out, np.ndarray)
    assert out.shape == (50, 2)


def test_continuous_features_are_capped_in_preprocessor():
    # An extreme duration_ms should be winsorized by the continuous pipeline.
    X = _frame(n=40)
    X = X.copy()
    X.loc[0, "duration_ms"] = 1e9
    out = features.make_preprocessor(scale=False).fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else np.asarray(out)
    # First transformer ("cont") emits CONTINUOUS_FEATURES in order; duration_ms
    # is index 0. The extreme value must be clipped well below 1e9.
    assert out[0, 0] < 1e9
