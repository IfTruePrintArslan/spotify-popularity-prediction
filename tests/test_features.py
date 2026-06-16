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
