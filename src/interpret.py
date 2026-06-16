"""Model interpretability: SHAP feature-importance plots and permutation importance.

Two complementary methods are used here to explain which features drive the
model's predictions:

  1. SHAP (SHapley Additive exPlanations) — assigns each feature a contribution
     score for every individual prediction.  The beeswarm summary plot shows
     the distribution of contributions across all test samples.

  2. Permutation importance — measures how much the model's score drops when a
     single feature column is randomly shuffled (breaking any relationship
     between that feature and the target).  A big drop means the feature was
     important; a small drop means the model barely used it.
"""

import matplotlib
matplotlib.use("Agg")  # use a non-interactive backend so plots save without a display
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance

from src import config


def _transform(pipeline, X):
    """Extract the preprocessed feature matrix and feature names from the pipeline.

    Pulls the 'pre' step out of the pipeline, runs transform on X, converts the
    result to a dense numpy array (some transformers return sparse matrices), and
    returns both the array and the human-readable column names.  These names are
    passed to SHAP so the summary plot labels axes with real feature names rather
    than column indices.
    """
    pre = pipeline.named_steps["pre"]
    Xt = pre.transform(X)
    # Some sklearn transformers (e.g. OneHotEncoder) return sparse matrices to
    # save memory.  SHAP needs a dense array, so convert if necessary.
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = pre.get_feature_names_out()
    return Xt, names


def shap_summary(pipeline, X_sample, prefix="best"):
    """Compute SHAP values and save a beeswarm summary plot to the figures directory.

    A TreeExplainer works with tree-based models (Random Forest, XGBoost,
    LightGBM) by walking each tree in the ensemble and tracking exactly how
    much each feature shifts the prediction.  This is much faster than the
    model-agnostic KernelExplainer and gives exact (not approximated) values
    for tree models.

    Parameters
    ----------
    pipeline : fitted imblearn/sklearn Pipeline with 'pre' and 'model' steps.
    X_sample : a sample of rows from the test set (a few hundred to a few
               thousand rows is enough — SHAP is slow on the full dataset).
    prefix : string prefix for the saved filename (default 'best').
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Preprocess the sample the same way the model sees it during prediction.
    Xt, names = _transform(pipeline, X_sample)

    model = pipeline.named_steps["model"]

    # shap.Explainer auto-selects TreeExplainer for tree-based models.
    explainer = shap.Explainer(model, Xt, feature_names=names)

    # check_additivity=False: TreeExplainer's additivity check can fail on
    # RandomForest due to tiny float-precision sums; the explanations are valid.
    values = explainer(Xt, check_additivity=False)

    # Binary tree classifiers return a 3-D array shaped (n_samples, n_features,
    # n_classes).  We slice [..., 1] to keep only the positive class ('hit=1')
    # so the beeswarm plot shows a standard single-class view that is easy to read.
    if np.asarray(values.values).ndim == 3:
        values = values[..., 1]

    shap.summary_plot(values, show=False)
    plt.savefig(config.FIGURES_DIR / f"{prefix}_shap_summary.png", bbox_inches="tight")
    plt.close()
    return values


def permutation_importance_table(pipeline, X_test, y_test):
    """Return a DataFrame ranking features by permutation importance on the test set.

    Permutation importance works by shuffling one feature column at a time and
    measuring how much the model's average precision (PR-AUC) decreases.  A
    large decrease means the model relied heavily on that feature; near-zero
    means the feature had little effect.  The result is model-agnostic — it
    works with any fitted estimator without needing access to the model internals.

    n_repeats=5 shuffles each feature five times and averages the score drops,
    which reduces noise from a single lucky or unlucky shuffle.
    """
    import pandas as pd

    result = permutation_importance(
        pipeline, X_test, y_test, scoring="average_precision",
        n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1,
    )

    # Build a tidy Series from the mean importance across the 5 repeats,
    # then sort descending so the most important features appear first.
    return (
        pd.Series(result.importances_mean, index=X_test.columns, name="perm_importance")
        .sort_values(ascending=False)
        .to_frame()
    )
