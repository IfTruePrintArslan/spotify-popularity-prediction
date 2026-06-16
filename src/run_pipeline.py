"""Run the full classification pipeline.

load -> clean -> split -> CV-compare (SMOTE pipelines) -> tune ensembles ->
select best tuned model -> fit -> held-out test metrics -> train-vs-test
overfitting report -> plots + SHAP -> save model.
"""
from src import config, data, evaluate, interpret, train


def main():
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

    # 3) Select the overall best tuned model and fit it on the full train set.
    best_name, best, best_params, best_score = train.best_tuned_model(tuned)
    print(f"\nSelected best tuned model: {best_name} (CV PR-AUC={best_score:.4f})")
    print(f"Best params: {best_params}")
    best.fit(X_train, y_train)

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
    print("\nPermutation importance:\n",
          interpret.permutation_importance_table(best, X_test, y_test).head(10))

    # 7) Persist the chosen model.
    train.save_pipeline(best)
    print(f"\nSaved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
