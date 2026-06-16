import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorize outliers to the Tukey IQR fences.

    ``fit`` learns per-column Q1, Q3, IQR and the bounds
    ``[Q1 - factor*IQR, Q3 + factor*IQR]``; ``transform`` clips each column to
    its learned bounds. Fitting only on training data (inside a CV fold /
    pipeline) keeps the transform leak-safe. Accepts a numpy array or a
    DataFrame and always returns a numpy array.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = self._to_array(X)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.q1_ = q1
        self.q3_ = q3
        self.iqr_ = iqr
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._to_array(X)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        # Capping is element-wise, so names pass through unchanged. Needed so the
        # enclosing ColumnTransformer/Pipeline can build feature names (SHAP).
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        n = getattr(self, "n_features_in_", 0)
        return np.asarray([f"x{i}" for i in range(n)], dtype=object)

    @staticmethod
    def _to_array(X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return np.asarray(X, dtype=float)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Build a ColumnTransformer that preprocesses all feature groups.

    Continuous features are IQR-capped then optionally scaled.
    Discrete/binary features skip capping (it would destroy their range) and
    are only scaled when requested.  Categorical features are one-hot encoded.
    The transformer is returned unfitted so it can be embedded inside a
    cross-validation or imblearn pipeline and fitted only on training data.
    """
    # Continuous features: cap outliers first, then (optionally) scale.
    cont_steps = [("cap", IQRCapper())]
    if scale:
        cont_steps.append(("scale", StandardScaler()))
    continuous_tf = Pipeline(cont_steps)

    # Discrete / near-binary features: no capping; scale only when requested.
    discrete_tf = StandardScaler() if scale else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("cont", continuous_tf, config.CONTINUOUS_FEATURES),
            ("disc", discrete_tf, config.DISCRETE_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )
