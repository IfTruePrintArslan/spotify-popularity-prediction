import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src import config


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def save_classification_plots(estimator, X_test, y_test, prefix="best"):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for disp, name in (
        (ConfusionMatrixDisplay.from_estimator, "confusion_matrix"),
        (RocCurveDisplay.from_estimator, "roc_curve"),
        (PrecisionRecallDisplay.from_estimator, "pr_curve"),
    ):
        ax = disp(estimator, X_test, y_test).ax_
        ax.figure.savefig(config.FIGURES_DIR / f"{prefix}_{name}.png", bbox_inches="tight")
        plt.close(ax.figure)


def comparison_table(cv_results: dict):
    import pandas as pd
    return (
        pd.Series(cv_results, name="cv_pr_auc")
        .sort_values(ascending=False)
        .to_frame()
    )


def _core_metrics(y_true, y_pred, y_proba) -> dict:
    """The three headline metrics used for the over/underfitting comparison."""
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def overfitting_report(estimator, X_train, y_train, X_test, y_test):
    """Compare TRAIN vs TEST metrics (PR-AUC, ROC-AUC, F1) and the gap.

    A large train>>test gap signals overfitting; uniformly low scores signal
    underfitting. Returns a DataFrame with ``train``, ``test`` and ``gap``
    (train - test) columns.
    """
    import pandas as pd

    train_m = _core_metrics(
        y_train, estimator.predict(X_train),
        estimator.predict_proba(X_train)[:, 1],
    )
    test_m = _core_metrics(
        y_test, estimator.predict(X_test),
        estimator.predict_proba(X_test)[:, 1],
    )
    df = pd.DataFrame({"train": train_m, "test": test_m})
    df["gap"] = df["train"] - df["test"]
    return df
