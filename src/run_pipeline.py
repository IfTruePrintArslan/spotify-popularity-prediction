"""Run the full classification pipeline: load -> clean -> split -> CV -> fit best -> evaluate -> save."""
from src import config, data, evaluate, interpret, train


def main():
    df = data.add_label(data.clean(data.load_raw()))
    X_train, X_test, y_train, y_test = data.split(df, target=config.LABEL)

    cv_results = train.cross_validate_models(X_train, y_train)
    table = evaluate.comparison_table(cv_results)
    print("\nCV PR-AUC by model:\n", table)

    best_name = table.index[0]
    if best_name == "dummy":  # guard: never ship the floor model
        best_name = table.index[1]
    print(f"\nBest model: {best_name}")

    best = train.fit_model(X_train, y_train, best_name)
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]
    metrics = evaluate.classification_metrics(y_test, y_pred, y_proba)
    print("\nHeld-out test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    evaluate.save_classification_plots(best, X_test, y_test)
    interpret.shap_summary(best, X_test.sample(min(2000, len(X_test)),
                                               random_state=config.RANDOM_STATE))
    print("\nPermutation importance:\n",
          interpret.permutation_importance_table(best, X_test, y_test).head(10))

    train.save_pipeline(best)
    print(f"\nSaved model -> {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
