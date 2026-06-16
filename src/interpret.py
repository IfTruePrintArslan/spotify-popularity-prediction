import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance

from src import config


def _transform(pipeline, X):
    pre = pipeline.named_steps["pre"]
    Xt = pre.transform(X)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    names = pre.get_feature_names_out()
    return Xt, names


def shap_summary(pipeline, X_sample, prefix="best"):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    Xt, names = _transform(pipeline, X_sample)
    model = pipeline.named_steps["model"]
    explainer = shap.Explainer(model, Xt, feature_names=names)
    values = explainer(Xt)
    shap.summary_plot(values, Xt, feature_names=names, show=False)
    plt.savefig(config.FIGURES_DIR / f"{prefix}_shap_summary.png", bbox_inches="tight")
    plt.close()
    return values


def permutation_importance_table(pipeline, X_test, y_test):
    import pandas as pd
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="average_precision",
        n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    return (
        pd.Series(result.importances_mean, index=X_test.columns, name="perm_importance")
        .sort_values(ascending=False)
        .to_frame()
    )
