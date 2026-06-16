"""Streamlit dashboard for the Spotify Hit Prediction project.

Tabs
----
1. Predict        -- enter audio features, get hit probability from the trained model.
2. Model Results  -- CV comparison, test metrics, overfitting table, hyperparameters,
                     permutation importance.
3. Figures        -- saved evaluation plots (confusion matrix, ROC, PR, SHAP).
4. EDA            -- exploratory data analysis built from the raw dataset on demand.
5. Report         -- the written project report rendered as Markdown.
"""

import sys
from pathlib import Path

# Make sure the project root (one level above app/) is on sys.path so that
# `from src import config` and `from src import data` resolve correctly
# regardless of the working directory Streamlit picks at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import pandas as pd
import streamlit as st

from src import config

# ---------------------------------------------------------------------------
# Page-level settings
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")
st.title("Spotify Hit Predictor")
st.caption(
    "An ML pipeline that classifies Spotify tracks as 'hits' "
    f"(popularity >= {config.HIT_THRESHOLD}) or 'flops'."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_predict, tab_results, tab_figures, tab_eda, tab_report = st.tabs(
    ["Predict", "Model Results", "Figures", "EDA", "Report"]
)

# ===========================================================================
# Tab 1 — Predict
# ===========================================================================

with tab_predict:
    st.header("Predict: Hit or Flop?")
    st.write(
        "Adjust the audio features below and the model will estimate the "
        "probability that the track becomes a hit."
    )

    @st.cache_resource
    def load_model():
        """Load the trained classifier once and cache it for the session."""
        return joblib.load(config.MODEL_PATH)

    if not config.MODEL_PATH.exists():
        st.error(
            f"Trained model not found at `{config.MODEL_PATH}`. "
            "Run `python -m src.run_pipeline` first."
        )
        st.stop()

    model = load_model()

    # --- input widgets (identical layout and defaults to the original app) ---
    col_left, col_right = st.columns(2)

    with col_left:
        danceability     = st.slider("Danceability",        0.0,   1.0,    0.6)
        energy           = st.slider("Energy",              0.0,   1.0,    0.7)
        loudness         = st.slider("Loudness (dB)",     -60.0,   2.0,   -6.0)
        speechiness      = st.slider("Speechiness",         0.0,   1.0,    0.05)
        acousticness     = st.slider("Acousticness",        0.0,   1.0,    0.2)
        instrumentalness = st.slider("Instrumentalness",    0.0,   1.0,    0.0)
        liveness         = st.slider("Liveness",            0.0,   1.0,    0.15)

    with col_right:
        valence        = st.slider("Valence",               0.0,   1.0,    0.5)
        tempo          = st.slider("Tempo (BPM)",          40.0, 220.0,  120.0)
        duration_ms    = st.slider("Duration (ms)",       30000, 600000, 200000)
        key            = st.slider("Key",                     0,    11,      1)
        mode           = st.selectbox("Mode",              [0, 1],    index=1)
        time_signature = st.selectbox("Time Signature",   [3, 4, 5], index=1)
        explicit       = int(st.checkbox("Explicit",      value=False))
        track_genre    = st.selectbox(
            "Genre",
            ["pop", "rock", "hip-hop", "edm", "classical", "jazz"],
        )

    inputs = {
        "danceability":     danceability,
        "energy":           energy,
        "loudness":         loudness,
        "speechiness":      speechiness,
        "acousticness":     acousticness,
        "instrumentalness": instrumentalness,
        "liveness":         liveness,
        "valence":          valence,
        "tempo":            tempo,
        "duration_ms":      duration_ms,
        "key":              key,
        "mode":             mode,
        "time_signature":   time_signature,
        "explicit":         explicit,
        "track_genre":      track_genre,
    }

    # Build a single-row DataFrame in the exact column order the pipeline expects.
    row = pd.DataFrame([inputs])[config.ALL_FEATURES]
    proba = float(model.predict_proba(row)[0, 1])

    st.divider()
    verdict = "HIT" if proba >= 0.5 else "FLOP"
    st.metric("Hit Probability", f"{proba:.1%}", delta=verdict)
    st.progress(proba)
    st.caption(
        f"'Hit' = popularity >= {config.HIT_THRESHOLD}.  "
        f"Model: {config.MODEL_PATH.name}"
    )

# ===========================================================================
# Tab 2 — Model Results
# ===========================================================================

with tab_results:
    st.header("Model Results")

    if not config.METRICS_PATH.exists():
        st.info(
            f"Metrics file not found at `{config.METRICS_PATH}`. "
            "Run `python -m src.run_pipeline` first."
        )
    else:
        with open(config.METRICS_PATH) as f:
            m = json.load(f)

        # --- Headline ---
        st.subheader("Best Model")
        c1, c2 = st.columns(2)
        c1.metric("Best Model", m["best_model"])
        c2.metric("Model Size", f"{m['model_size_mb']} MB")

        st.divider()

        # --- CV PR-AUC comparison table ---
        st.subheader("Cross-Validation PR-AUC Comparison")
        st.write(
            "PR-AUC (area under the precision-recall curve) averaged over "
            f"{config.CV_FOLDS} folds. Higher is better."
        )

        # Merge baseline and tuned scores into one tidy table.
        baseline = m.get("cv_pr_auc", {})
        tuned    = m.get("tuned_cv_pr_auc", {})
        all_models = sorted(
            set(baseline) | set(tuned),
            key=lambda k: tuned.get(k, baseline.get(k, 0)),
            reverse=True,
        )
        cv_rows = []
        for mdl in all_models:
            cv_rows.append({
                "Model":        mdl,
                "Baseline CV PR-AUC": baseline.get(mdl, "-"),
                "Tuned CV PR-AUC":    tuned.get(mdl, "-"),
            })
        st.table(pd.DataFrame(cv_rows).set_index("Model"))

        st.divider()

        # --- Test-set metric cards ---
        st.subheader("Test-Set Metrics")
        tm = m.get("test_metrics", {})
        mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
        mc1.metric("Accuracy",  f"{tm.get('accuracy',  0):.3f}")
        mc2.metric("ROC-AUC",   f"{tm.get('roc_auc',   0):.3f}")
        mc3.metric("PR-AUC",    f"{tm.get('pr_auc',    0):.3f}")
        mc4.metric("F1",        f"{tm.get('f1',        0):.3f}")
        mc5.metric("Precision", f"{tm.get('precision', 0):.3f}")
        mc6.metric("Recall",    f"{tm.get('recall',    0):.3f}")

        st.divider()

        # --- Overfitting table ---
        st.subheader("Overfitting Analysis (Train vs. Test)")
        st.write("Gap = train score - test score. Larger gap indicates more overfitting.")
        ov = m.get("overfitting", {})
        ov_rows = []
        for metric_name, scores in ov.items():
            train_val = scores.get("train", float("nan"))
            test_val  = scores.get("test",  float("nan"))
            gap       = round(train_val - test_val, 4)
            ov_rows.append({
                "Metric": metric_name,
                "Train":  train_val,
                "Test":   test_val,
                "Gap (Train - Test)": gap,
            })
        st.table(pd.DataFrame(ov_rows).set_index("Metric"))

        st.divider()

        # --- Best hyperparameters ---
        st.subheader("Best LightGBM Hyperparameters")
        st.json(m.get("best_params_lightgbm", {}))

        st.divider()

        # --- Permutation importance bar chart ---
        st.subheader("Permutation Feature Importance")
        pi = m.get("permutation_importance", {})
        if pi:
            importance_series = (
                pd.Series(pi)
                .sort_values(ascending=False)
                .rename("Mean decrease in PR-AUC")
            )
            st.bar_chart(importance_series)
            st.caption(
                "Permutation importance: mean drop in test PR-AUC when a feature's "
                "values are randomly shuffled. 'track_genre' dominates by a large "
                "margin, confirming that genre is the strongest predictor of hit status."
            )

# ===========================================================================
# Tab 3 — Figures
# ===========================================================================

with tab_figures:
    st.header("Evaluation Figures")

    figures = [
        ("best_confusion_matrix.png", "Confusion Matrix"),
        ("best_roc_curve.png",        "ROC Curve"),
        ("best_pr_curve.png",         "Precision-Recall Curve"),
        ("best_shap_summary.png",     "SHAP Summary Plot"),
    ]

    for filename, caption in figures:
        fig_path = config.FIGURES_DIR / filename
        if fig_path.exists():
            left, mid, right = st.columns([1, 3, 1])
            with mid:
                st.image(str(fig_path), caption=caption, width=650)
        else:
            st.info(
                f"Figure `{filename}` not found at `{fig_path}`. "
                "Run `python -m src.run_pipeline` to generate it."
            )

# ===========================================================================
# Tab 4 — EDA
# ===========================================================================

with tab_eda:
    st.header("Exploratory Data Analysis")

    @st.cache_data
    def load_eda_data(max_rows: int = 20000):
        """Load, clean, and label the raw dataset; sample for EDA performance."""
        from src.data import load_raw, clean, add_label

        df = add_label(clean(load_raw()))
        if len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=config.RANDOM_STATE)
        return df

    if not config.RAW_CSV.exists():
        st.info(
            f"Raw dataset not found at `{config.RAW_CSV}`. "
            "Download the Kaggle 'spotify-tracks-dataset' CSV and place it there."
        )
    else:
        try:
            df_eda = load_eda_data()
        except Exception as exc:
            st.error(f"Could not load dataset: {exc}")
            df_eda = None

        if df_eda is not None:
            n_shown = len(df_eda)
            st.caption(
                f"Charts based on a random sample of {n_shown:,} rows "
                "(capped at 20,000 for dashboard performance)."
            )

            import matplotlib.pyplot as plt
            import seaborn as sns

            # --- Popularity distribution ---
            st.subheader("Popularity Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.hist(df_eda[config.TARGET], bins=50, edgecolor="white")
            ax1.axvline(config.HIT_THRESHOLD, color="red", linestyle="--",
                        label=f"Hit threshold ({config.HIT_THRESHOLD})")
            ax1.set_xlabel("Popularity")
            ax1.set_ylabel("Count")
            ax1.set_title("Popularity Distribution")
            ax1.legend()
            st.pyplot(fig1)
            plt.close(fig1)

            # --- Hit / flop count ---
            st.subheader("Hit vs. Flop Count")
            label_counts = df_eda[config.LABEL].value_counts().rename(
                index={0: "Flop", 1: "Hit"}
            )
            st.bar_chart(label_counts)

            # --- Numeric correlation heatmap ---
            st.subheader("Numeric Feature Correlation")
            corr = df_eda[config.NUMERIC_FEATURES].corr()
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.4, ax=ax2,
            )
            ax2.set_title("Pearson Correlation — Numeric Features")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

            # --- Top-15 genres ---
            st.subheader("Top 15 Genres by Track Count")
            top_genres = (
                df_eda["track_genre"]
                .value_counts()
                .head(15)
                .rename("Track Count")
            )
            st.bar_chart(top_genres)

# ===========================================================================
# Tab 5 — Report
# ===========================================================================

with tab_report:
    st.header("Project Report")

    report_path = config.ROOT / "docs" / "REPORT.md"
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info(
            f"Report not found at `{report_path}`. "
            "Create `docs/REPORT.md` to display it here."
        )
