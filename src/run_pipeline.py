"""Run the full classification pipeline.

load -> clean -> split -> CV-compare (SMOTE pipelines) -> tune ensembles ->
select best tuned model -> fit -> held-out test metrics -> train-vs-test
overfitting report -> plots + SHAP -> save model -> persist metrics.json.
"""
import json

from src import config, data, evaluate, interpret, train


def main():
    """Execute every stage of the pipeline end-to-end and print progress to stdout.

    Stages: load -> clean -> label -> train/test split -> cross-validate all
    models -> tune the three ensembles -> select and refit the best tuned model
    -> evaluate on held-out test set -> overfitting report -> save plots and
    SHAP summary -> persist the final model to disk.
    """
    df = data.add_label(data.clean(data.load_raw()))
    X_train, X_test, y_train, y_test = data.split(df, target=config.LABEL)

    # 1) Cross-validate every model (all SMOTE pipelines except dummy).
    cv_results = train.cross_validate_models(X_train, y_train)
    table = evaluate.comparison_table(cv_results)
    print("\nCV PR-AUC by model:\n", table)

    # 2) Hyperparameter-tune the three ensembles via RandomizedSearchCV.
    print("\nTuning ensembles (RandomizedSearchCV, PR-AUC, cv=3)...")
    tuned = train.tune_ensembles(X_train, y_train)
    print("\nTuned CV PR-AUC by ensemble:")
    for name, (_, params, score) in tuned.items():
        print(f"  {name}: {score:.4f}")
        print(f"    best params: {params}")

    # 3) Select the overall best tuned model.  tune_model already refit it on
    #    the FULL training set (the search itself only used a subsample to save
    #    memory), so no extra .fit is needed here.
    best_name, best, best_params, best_score = train.best_tuned_model(tuned)
    print(f"\nSelected best tuned model: {best_name} (CV PR-AUC={best_score:.4f})")
    print(f"Best params: {best_params}")

    # 4) Held-out test metrics.
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = evaluate.classification_metrics(y_test, y_pred, y_proba)
    print("\nHeld-out test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 5) Train-vs-test overfitting / underfitting analysis.
    of_report = evaluate.overfitting_report(best, X_train, y_train, X_test, y_test)
    print("\nTrain vs Test (overfitting check, gap = train - test):\n", of_report)

    # 6) Plots + SHAP interpretability.
    evaluate.save_classification_plots(best, X_test, y_test)
    interpret.shap_summary(best, X_test.sample(min(2000, len(X_test)),
                                               random_state=config.RANDOM_STATE))
    # Compute permutation importance once and reuse for both the print and the JSON.
    perm_table = interpret.permutation_importance_table(best, X_test, y_test)
    print("\nPermutation importance:\n", perm_table.head(10))

    # 7) Persist metrics to JSON so the dashboard can read reproducible numbers
    #    without re-running the pipeline.  All values are taken directly from the
    #    objects computed above — nothing is hardcoded here.

    # Model size in MB (0 if model file not yet written — will be updated on next run).
    model_size_mb = round(config.MODEL_PATH.stat().st_size / 1e6, 1) if config.MODEL_PATH.exists() else None

    metrics_dict = {
        "best_model": f"{best_name} (tuned)",
        "model_size_mb": model_size_mb,
        # cv_results is {model_name: mean_pr_auc} from cross_validate_models
        "cv_pr_auc": {k: round(float(v), 4) for k, v in cv_results.items()},
        # tuned is {model_name: (pipeline, params, score)} from tune_ensembles
        "tuned_cv_pr_auc": {k: round(float(v[2]), 4) for k, v in tuned.items()},
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        # of_report is a DataFrame with columns train, test, gap and index = metric name
        "overfitting": {
            metric: {
                "train": round(float(of_report.loc[metric, "train"]), 4),
                "test": round(float(of_report.loc[metric, "test"]), 4),
            }
            for metric in of_report.index
        },
        # Strip the pipeline prefix ("model__") from param keys so the JSON is readable.
        f"best_params_{best_name.lower().replace(' ', '_').rstrip('_')}": {
            k.replace("model__", ""): v for k, v in best_params.items()
        },
        # perm_table is a DataFrame with index=feature and column 'perm_importance'
        "permutation_importance": {
            feat: round(float(imp), 4)
            for feat, imp in perm_table["perm_importance"].head(10).items()
        },
    }

    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w") as _f:
        json.dump(metrics_dict, _f, indent=2)
    print(f"\nSaved metrics -> {config.METRICS_PATH}")

    # 8) Persist the chosen model.
    train.save_pipeline(best)
    print(f"\nSaved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
