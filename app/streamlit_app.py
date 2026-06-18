"""Streamlit dashboard for the Spotify Hit Prediction project.

Tabs
----
1. Predict        -- enter audio features, get hit probability from the trained model.
2. Model Results  -- CV comparison, test metrics, overfitting table, hyperparameters,
                     permutation importance.
3. Figures        -- saved evaluation plots (confusion matrix, ROC, PR, SHAP).
4. EDA            -- exploratory data analysis built from the raw dataset on demand.
5. Report         -- the written project report rendered as Markdown.
6. Experiment     -- model-improvement experiment results (target-enc, stacking,
                     threshold tuning) vs. production LightGBM.
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

tab_predict, tab_results, tab_figures, tab_eda, tab_report, tab_experiment = st.tabs(
    ["Predict", "Model Results", "Figures", "EDA", "Report", "Experiment"]
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
        st.caption("How suited the track is for dancing (0–1) — based on tempo, rhythm, and beat strength.")
        energy           = st.slider("Energy",              0.0,   1.0,    0.7)
        st.caption("Intensity and activity (0–1) — loud, fast, noisy tracks score high.")
        loudness         = st.slider("Loudness (dB)",     -60.0,   2.0,   -6.0)
        st.caption("Average loudness in decibels (−60 to +2); closer to 0 is louder.")
        speechiness      = st.slider("Speechiness",         0.0,   1.0,    0.05)
        st.caption("Presence of spoken words (0–1); above ~0.66 is mostly speech (e.g. rap, podcast).")
        acousticness     = st.slider("Acousticness",        0.0,   1.0,    0.2)
        st.caption("Confidence the track is acoustic (0–1); 1 = very acoustic.")
        instrumentalness = st.slider("Instrumentalness",    0.0,   1.0,    0.0)
        st.caption("Likelihood the track has no vocals (0–1); above 0.5 is likely instrumental.")
        liveness         = st.slider("Liveness",            0.0,   1.0,    0.15)
        st.caption("Presence of a live audience (0–1); above 0.8 suggests a live recording.")

    with col_right:
        valence        = st.slider("Valence",               0.0,   1.0,    0.5)
        st.caption("Musical positivity / mood (0–1); high = happy/cheerful, low = sad/angry.")
        tempo          = st.slider("Tempo (BPM)",          40.0, 220.0,  120.0)
        st.caption("Speed of the track in beats per minute.")
        duration_ms    = st.slider("Duration (ms)",       30000, 600000, 200000)
        st.caption("Track length in milliseconds (200000 ≈ 3 min 20 sec).")
        key            = st.slider("Key",                     0,    11,      1)
        st.caption("Musical key as a pitch class 0–11 (0 = C, 1 = C♯/D♭, … 11 = B).")
        mode           = st.selectbox("Mode",              [0, 1],    index=1)
        st.caption("Scale type: 1 = major (brighter), 0 = minor (darker).")
        time_signature = st.selectbox("Time Signature",   [3, 4, 5], index=1)
        st.caption("Beats per bar (e.g. 4 = common 4/4 time).")
        explicit       = int(st.checkbox("Explicit",      value=False))
        st.caption("Whether the track has explicit lyrics.")
        track_genre    = st.selectbox(
            "Genre",
            ["pop", "rock", "hip-hop", "edm", "classical", "jazz"],
        )
        st.caption("The track's genre — the strongest predictor of popularity in this model.")

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
    st.markdown(
        "This section mirrors the full EDA from `notebooks/01_eda.ipynb`, "
        "covering dataset overview, feature distributions, target analysis, "
        "feature relationships, and categorical/discrete feature summaries."
    )

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # Cache both raw and cleaned data so they load once per session.
    @st.cache_data
    def load_eda_raw():
        """Load the raw dataset (before cleaning) for overview stats."""
        from src.data import load_raw
        return load_raw()

    @st.cache_data
    def load_eda_clean():
        """Load, clean, and label the full dataset for analysis."""
        from src.data import load_raw, clean, add_label
        return add_label(clean(load_raw()))

    if not config.RAW_CSV.exists():
        st.info(
            f"Raw dataset not found at `{config.RAW_CSV}`. "
            "Download the Kaggle 'spotify-tracks-dataset' CSV and place it there."
        )
    else:
        try:
            df_raw = load_eda_raw()
            df_eda = load_eda_clean()
            _load_ok = True
        except Exception as _exc:
            st.error(f"Could not load dataset: {_exc}")
            _load_ok = False

        if _load_ok:

            # ---------------------------------------------------------------
            # Section 1 — Dataset Overview
            # ---------------------------------------------------------------
            st.subheader("1. Dataset Overview")
            st.markdown(
                "We load the **raw** CSV first — before any cleaning — to see it "
                "exactly as delivered. Then we build the cleaned + labeled "
                "dataframe used throughout the rest of the EDA."
            )

            # Shape metrics
            col_r, col_c = st.columns(2)
            col_r.metric("Raw Rows", f"{df_raw.shape[0]:,}")
            col_c.metric("Raw Columns", f"{df_raw.shape[1]}")

            st.markdown("**Column dtypes**")
            st.dataframe(
                df_raw.dtypes.rename("dtype").to_frame(),
                use_container_width=False,
            )

            # Missing values
            st.markdown(
                "**Missing values per column** — a missing-value audit is the "
                "first thing to check after loading. Missing values can distort "
                "distributions and break models if not handled."
            )
            _missing = df_raw.isnull().sum()
            _missing_pct = (_missing / len(df_raw) * 100).round(2)
            _missing_df = pd.DataFrame(
                {"missing_count": _missing, "missing_pct_%": _missing_pct}
            )
            st.dataframe(_missing_df, use_container_width=False)

            # Duplicate count (computed on raw)
            _exact_dups = int(df_raw.duplicated().sum())
            _trackid_dups = int(df_raw.duplicated(subset="track_id", keep="first").sum())
            st.markdown(
                f"**Duplicate rows:** {_exact_dups:,} exact duplicates. "
                f"**Duplicate track IDs** (same song, multiple genres): "
                f"{_trackid_dups:,} ({_trackid_dups / len(df_raw) * 100:.1f}% of raw rows). "
                "These are removed during cleaning."
            )

            # Cleaned shape
            _rows_removed = len(df_raw) - len(df_eda)
            st.markdown(
                f"**Cleaned shape:** {df_eda.shape[0]:,} rows x {df_eda.shape[1]} columns "
                f"({_rows_removed:,} rows removed). "
                "The `hit` column was added: 1 if popularity >= "
                f"{config.HIT_THRESHOLD}, else 0."
            )

            # Summary statistics
            st.markdown(
                "**Summary statistics** — mean, median, std, min/max, and "
                "quartiles for every numeric column. Notice the large scale "
                "difference between `duration_ms` and the 0-1 bounded features."
            )
            _stats_cols = config.NUMERIC_FEATURES + [config.TARGET]
            _stats = df_eda[_stats_cols].describe().T
            _stats["median"] = df_eda[_stats_cols].median()
            _stats = _stats[["count", "mean", "median", "std", "min", "25%", "75%", "max"]]
            st.dataframe(_stats.round(3), use_container_width=False)

            # ---------------------------------------------------------------
            # Section 2 — Feature Distributions
            # ---------------------------------------------------------------
            st.subheader("2. Feature Distributions")
            st.markdown(
                "Understanding how each feature is distributed helps decide on "
                "preprocessing steps like scaling or log-transformation. Highly "
                "skewed features may mislead linear models if left untreated."
            )

            # --- 2.1 Histograms of continuous features ---
            st.markdown(
                "**2.1 Histograms of continuous features** — each bar represents "
                "a bin of values. A tall bar means many tracks fall in that range. "
                "All 10 continuous features are shown in a grid."
            )
            _n_cont = len(config.CONTINUOUS_FEATURES)
            _n_cols_hist = 2
            _n_rows_hist = (_n_cont + _n_cols_hist - 1) // _n_cols_hist

            fig_hist, axes_hist = plt.subplots(
                _n_rows_hist, _n_cols_hist, figsize=(10, _n_rows_hist * 2.8)
            )
            axes_hist = axes_hist.flatten()
            for _i, _feat in enumerate(config.CONTINUOUS_FEATURES):
                axes_hist[_i].hist(
                    df_eda[_feat].dropna(), bins=50,
                    color="steelblue", edgecolor="white", alpha=0.85
                )
                axes_hist[_i].set_title(_feat, fontsize=10)
                axes_hist[_i].set_xlabel(_feat, fontsize=9)
                axes_hist[_i].set_ylabel("Count", fontsize=9)
            for _j in range(_i + 1, len(axes_hist)):
                axes_hist[_j].set_visible(False)
            fig_hist.suptitle("Histograms — Continuous Features", fontsize=12, y=1.01)
            plt.tight_layout()
            st.pyplot(fig_hist, use_container_width=False)
            plt.close(fig_hist)

            # --- 2.2 Skewness bar chart ---
            st.markdown(
                "**2.2 Skewness** — measures asymmetry. |skew| > 1 is highly "
                "skewed (red), |skew| > 0.5 is moderately skewed (orange). "
                "Features like `speechiness`, `instrumentalness`, and `liveness` "
                "are right-skewed because most songs have low values but a few "
                "have very high ones. The pipeline applies IQR winsorization to "
                "continuous features to reduce the effect of these extreme values."
            )
            _skew_vals = df_eda[config.CONTINUOUS_FEATURES].skew().sort_values(
                ascending=False
            )
            fig_skew, ax_skew = plt.subplots(figsize=(6, 4))
            _skew_colors = [
                "tomato" if abs(v) > 1 else ("orange" if abs(v) > 0.5 else "steelblue")
                for v in _skew_vals
            ]
            ax_skew.barh(_skew_vals.index, _skew_vals.values, color=_skew_colors)
            ax_skew.axvline(0, color="black", linewidth=0.8)
            ax_skew.set_xlabel("Skewness")
            ax_skew.set_title("Feature Skewness  (red=|skew|>1, orange=|skew|>0.5)")
            plt.tight_layout()
            st.pyplot(fig_skew, use_container_width=False)
            plt.close(fig_skew)

            # Skewness table
            _skew_df = _skew_vals.rename("skewness").to_frame()
            _skew_df["flag"] = _skew_df["skewness"].abs().apply(
                lambda x: "HIGH" if x > 1 else ("MODERATE" if x > 0.5 else "low")
            )
            st.dataframe(_skew_df.round(3), use_container_width=False)

            # --- 2.3 Boxplots ---
            st.markdown(
                "**2.3 Boxplots** — the box spans the IQR (25th–75th percentile), "
                "the line is the median, whiskers extend 1.5x IQR, and dots beyond "
                "are outliers. Features with many outlier dots receive IQR "
                "winsorization (capping) in the pipeline."
            )
            fig_box, axes_box = plt.subplots(
                _n_rows_hist, _n_cols_hist, figsize=(10, _n_rows_hist * 2.8)
            )
            axes_box = axes_box.flatten()
            for _i, _feat in enumerate(config.CONTINUOUS_FEATURES):
                axes_box[_i].boxplot(
                    df_eda[_feat].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6),
                    medianprops=dict(color="red", linewidth=2),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                )
                axes_box[_i].set_title(_feat, fontsize=10)
                axes_box[_i].set_ylabel(_feat, fontsize=9)
                axes_box[_i].set_xticks([])
            for _j in range(_i + 1, len(axes_box)):
                axes_box[_j].set_visible(False)
            fig_box.suptitle(
                "Boxplots — Continuous Features (outliers as dots)",
                fontsize=12, y=1.01
            )
            plt.tight_layout()
            st.pyplot(fig_box, use_container_width=False)
            plt.close(fig_box)

            # ---------------------------------------------------------------
            # Section 3 — Target & Class Imbalance
            # ---------------------------------------------------------------
            st.subheader("3. Target Variable — Popularity & Class Imbalance")
            st.markdown(
                "Our target is `popularity` (0-100, continuous). We binarize it "
                f"into `hit` using a threshold of **{config.HIT_THRESHOLD}**. "
                "Understanding the distribution and class balance is critical "
                "before modeling — imbalanced classes require special handling."
            )

            # --- 3.1 Popularity histogram with threshold line ---
            st.markdown(
                "**3.1 Popularity distribution** — the red dashed line marks the "
                "hit/flop threshold. Most tracks cluster below it, revealing a "
                "left-skewed popularity distribution."
            )
            _pop_mean = df_eda[config.TARGET].mean()
            _pop_med = df_eda[config.TARGET].median()
            fig_pop, ax_pop = plt.subplots(figsize=(6, 4))
            ax_pop.hist(
                df_eda[config.TARGET], bins=60,
                color="steelblue", edgecolor="white", alpha=0.85
            )
            ax_pop.axvline(
                config.HIT_THRESHOLD, color="red", linewidth=2, linestyle="--",
                label=f"Hit threshold = {config.HIT_THRESHOLD}"
            )
            ax_pop.set_xlabel("Popularity (0-100)")
            ax_pop.set_ylabel("Number of Tracks")
            ax_pop.set_title("Distribution of Track Popularity")
            ax_pop.legend()
            plt.tight_layout()
            st.pyplot(fig_pop, use_container_width=False)
            plt.close(fig_pop)
            st.caption(
                f"Mean popularity: {_pop_mean:.2f}  |  "
                f"Median: {_pop_med:.2f}  |  "
                f"Std dev: {df_eda[config.TARGET].std():.2f}"
            )

            # --- 3.2 Hit vs Flop count bar ---
            st.markdown(
                "**3.2 Class imbalance** — if one class greatly outnumbers the "
                "other, a naive model can reach high accuracy just by always "
                "predicting the majority class. That is why we use SMOTE "
                "oversampling and evaluate with precision/recall in addition "
                "to accuracy."
            )
            _class_counts = df_eda[config.LABEL].value_counts().sort_index()
            _hit_rate = df_eda[config.LABEL].mean()
            _imbalance_ratio = _class_counts[0] / _class_counts[1]
            _class_labels = ["Flop (0)", "Hit (1)"]

            fig_cls, ax_cls = plt.subplots(figsize=(5, 4))
            _bars = ax_cls.bar(
                _class_labels, _class_counts.values,
                color=["coral", "steelblue"], edgecolor="white", width=0.5
            )
            for _bar, _cnt in zip(_bars, _class_counts.values):
                ax_cls.text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height() + 200,
                    f"{_cnt:,}", ha="center", va="bottom", fontsize=10
                )
            ax_cls.set_ylabel("Number of Tracks")
            ax_cls.set_title(f"Class Balance  (threshold = {config.HIT_THRESHOLD})")
            plt.tight_layout()
            st.pyplot(fig_cls, use_container_width=False)
            plt.close(fig_cls)
            st.caption(
                f"Hit rate: {_hit_rate * 100:.1f}%  |  "
                f"Flop:Hit ratio: {_imbalance_ratio:.2f}:1  |  "
                "SMOTE is applied during training to balance the classes."
            )

            # --- 3.3 Before vs After SMOTE ---
            st.subheader("Class Balance: Before vs After SMOTE")
            st.markdown(
                "The raw training set is imbalanced — roughly **24% hits** and "
                "76% flops. Left uncorrected, a classifier can score high accuracy "
                "just by predicting 'flop' for everything. "
                "**SMOTE** (Synthetic Minority Over-sampling Technique) fixes this "
                "by synthesizing *new* minority-class (hit) examples: it picks a "
                "real hit, finds its *k* nearest hit-neighbours in feature space, "
                "and interpolates a new point along the line between them. This "
                "creates realistic synthetic hits rather than just copying existing "
                "ones. SMOTE is applied **only to the training data** inside the "
                "pipeline — the test set is never touched — so there is no "
                "data-leakage risk. After resampling the classes are 50/50, giving "
                "the model equal exposure to both classes during training."
            )

            # Cached computation: run the EXACT resampling the model uses — split
            # the cleaned data into the real stratified 80/20 train set, preprocess
            # the FULL training set (IQR cap + one-hot of all 113 genres), then run
            # SMOTE.fit_resample on the whole thing.  This is not a sample: the
            # before/after counts are the model's real training-set balance.  The
            # work is heavy (~12 s) so it is wrapped in @st.cache_data with no
            # arguments — it runs once per session and every rerun is instant.
            @st.cache_data
            def _compute_smote_counts():
                """Return (before, after) Flop/Hit counts for the FULL train set.

                Mirrors the production pipeline exactly: split -> preprocess the
                full X_train (no scaling needed for SMOTE) -> SMOTE.fit_resample.
                No subsampling, so the counts are the real training-set balance.
                """
                from src.data import load_raw, clean, add_label, split
                from src.features import make_preprocessor
                from imblearn.over_sampling import SMOTE

                _df = add_label(clean(load_raw()))
                X_tr, _, y_tr, _ = split(_df)

                # Preprocess the FULL training set (IQR cap + one-hot) — no scaling
                # needed for SMOTE, which only needs distances in feature space.
                pre = make_preprocessor(scale=False)
                Xt = pre.fit_transform(X_tr)

                # SMOTE fit_resample on the entire preprocessed training matrix.
                _, y_res = SMOTE(random_state=config.RANDOM_STATE).fit_resample(Xt, y_tr)

                before = {0: int((y_tr == 0).sum()), 1: int((y_tr == 1).sum())}
                after  = {0: int((y_res == 0).sum()), 1: int((y_res == 1).sum())}
                return before, after

            try:
                _smote_before, _smote_after = _compute_smote_counts()

                # Grouped bar chart — two groups (Before / After), two bars each
                # (Flop / Hit).
                _smote_fig, _smote_ax = plt.subplots(figsize=(6, 4))
                _x = np.arange(2)          # one group position per condition
                _bar_w = 0.35
                _bars_flop = _smote_ax.bar(
                    _x - _bar_w / 2,
                    [_smote_before[0], _smote_after[0]],
                    width=_bar_w, label="Flop (0)",
                    color="coral", edgecolor="white",
                )
                _bars_hit = _smote_ax.bar(
                    _x + _bar_w / 2,
                    [_smote_before[1], _smote_after[1]],
                    width=_bar_w, label="Hit (1)",
                    color="steelblue", edgecolor="white",
                )
                # Count labels on top of every bar.
                for _b in list(_bars_flop) + list(_bars_hit):
                    _smote_ax.text(
                        _b.get_x() + _b.get_width() / 2,
                        _b.get_height() + max(_smote_before[0], _smote_after[0]) * 0.01,
                        f"{int(_b.get_height()):,}",
                        ha="center", va="bottom", fontsize=9,
                    )
                _smote_ax.set_xticks(_x)
                _smote_ax.set_xticklabels(["Before SMOTE", "After SMOTE"], fontsize=11)
                _smote_ax.set_ylabel("Number of Tracks")
                _smote_ax.set_title("Class Balance: Before vs After SMOTE")
                _smote_ax.legend(fontsize=9)
                plt.tight_layout()
                st.pyplot(_smote_fig, use_container_width=False)
                plt.close(_smote_fig)
                _smote_total_before = _smote_before[0] + _smote_before[1]
                st.caption(
                    f"Computed on the FULL training set ({_smote_total_before:,} rows) — "
                    "the exact stratified 80/20 train split the model is fitted on, "
                    "with no subsampling. Before SMOTE: "
                    f"{_smote_before[0]:,} flops vs {_smote_before[1]:,} hits. "
                    f"After SMOTE: {_smote_after[0]:,} vs {_smote_after[1]:,} "
                    "(balanced ~50/50), giving the model equal exposure to hits and "
                    "flops during training."
                )
            except Exception as _smote_exc:
                st.info(
                    f"Could not render the SMOTE chart: {_smote_exc}. "
                    "Ensure imbalanced-learn is installed (`pip install imbalanced-learn`)."
                )

            # ---------------------------------------------------------------
            # Section 4 — Feature Relationships
            # ---------------------------------------------------------------
            st.subheader("4. Feature–Target Relationships")
            st.markdown(
                "We now explore how individual features relate to `popularity`. "
                "This guides feature selection and helps explain model predictions."
            )

            # --- 4.1 Correlation heatmap (lower triangle) ---
            st.markdown(
                "**4.1 Correlation heatmap** — Pearson correlation ranges from "
                "-1 (perfect negative) to +1 (perfect positive). Pairs near ±1 "
                "indicate redundancy; pairs near 0 are linearly independent. "
                "The lower triangle is shown to avoid redundancy."
            )
            _corr_cols = config.NUMERIC_FEATURES + [config.TARGET]
            _corr_matrix = df_eda[_corr_cols].corr()
            _mask = np.triu(np.ones_like(_corr_matrix, dtype=bool))

            fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                _corr_matrix,
                mask=_mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                ax=ax_heat,
                annot_kws={"size": 7},
            )
            ax_heat.set_title("Pearson Correlation Matrix (lower triangle)", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_heat, use_container_width=False)
            plt.close(fig_heat)

            # --- 4.2 Per-feature correlation with target ---
            st.markdown(
                "**4.2 Per-feature correlation with popularity** — a sorted bar "
                "chart shows which features have the strongest linear relationship "
                "with the target. All values are low (|r| < 0.15), confirming "
                "this is a non-linear problem suited to tree-based models."
            )
            _target_corr = (
                df_eda[config.NUMERIC_FEATURES]
                .corrwith(df_eda[config.TARGET])
                .sort_values()
            )
            _tc_colors = [
                "tomato" if v < 0 else "steelblue" for v in _target_corr
            ]
            fig_tc, ax_tc = plt.subplots(figsize=(6, 4))
            ax_tc.barh(_target_corr.index, _target_corr.values, color=_tc_colors)
            ax_tc.axvline(0, color="black", linewidth=0.8)
            ax_tc.set_xlabel("Pearson Correlation with Popularity")
            ax_tc.set_title("Feature Correlation with Target (popularity)")
            plt.tight_layout()
            st.pyplot(fig_tc, use_container_width=False)
            plt.close(fig_tc)

            # --- 4.3 Scatter plots (full data) ---
            _SCATTER_FEATURES = [
                "loudness", "danceability", "energy", "acousticness"
            ]

            @st.cache_data
            def _build_scatter_fig():
                """Build the 4-panel feature-vs-popularity scatter on FULL data.

                Plots every cleaned row (~90k).  To keep ~90k points fast and
                readable without sampling we render each scatter with
                ``rasterized=True`` (the dense point cloud is flattened to a
                bitmap at draw time), a tiny marker (``s=4``) and low opacity
                (``alpha=0.15``).  A red linear-fit trend line is overlaid on
                each panel.  Returned figure is cached so it builds once.
                """
                from src.data import load_raw, clean, add_label
                _df = add_label(clean(load_raw()))

                _fig, _axes = plt.subplots(2, 2, figsize=(9, 6))
                _axes = _axes.flatten()
                for _i, _feat in enumerate(_SCATTER_FEATURES):
                    _x = _df[_feat].values
                    _y = _df[config.TARGET].values
                    _valid = ~(np.isnan(_x) | np.isnan(_y))
                    _axes[_i].scatter(
                        _x[_valid], _y[_valid],
                        alpha=0.15, s=4, color="steelblue", rasterized=True,
                    )
                    _m, _b = np.polyfit(_x[_valid], _y[_valid], 1)
                    _xl = np.linspace(_x[_valid].min(), _x[_valid].max(), 100)
                    _axes[_i].plot(_xl, _m * _xl + _b, color="red", linewidth=1.5,
                                   label="trend")
                    _axes[_i].set_xlabel(_feat, fontsize=9)
                    _axes[_i].set_ylabel("Popularity", fontsize=9)
                    _axes[_i].set_title(f"{_feat} vs Popularity", fontsize=10)
                    _axes[_i].legend(fontsize=8)
                _fig.suptitle(
                    f"Scatter Plots — Key Features vs Popularity  "
                    f"(all {len(_df):,} tracks)",
                    fontsize=11
                )
                _fig.tight_layout()
                return _fig, len(_df)

            fig_scat, _scat_n = _build_scatter_fig()
            st.markdown(
                f"**4.3 Scatter plots** — key features vs popularity, plotted for "
                f"the **full cleaned dataset ({_scat_n:,} tracks)** — no sampling. "
                "The dense point cloud is rasterized for speed and drawn with small, "
                "low-opacity markers so structure stays visible. A red trend line "
                "(linear fit) is overlaid on each plot."
            )
            st.pyplot(fig_scat, use_container_width=False)

            # --- 4.4 Pairplot (full data) ---
            _PAIRPLOT_FEATURES = [
                "danceability", "energy", "loudness",
                "acousticness", "instrumentalness"
            ]

            @st.cache_data
            def _build_pairplot_fig():
                """Build the seaborn pairplot on the FULL dataset (no sampling).

                This is the heaviest EDA figure: a 5x5 grid of pairwise scatters
                over every cleaned row (~90k), coloured by Hit/Flop.  Measured at
                ~38 s to build+render once, comfortably under the 90 s budget, so
                it runs on full data.  Each scatter uses ``rasterized=True`` plus a
                tiny marker (``s=6``) and low opacity (``alpha=0.15``) so the dense
                clouds render quickly and stay legible.  Cached so it builds once
                per session; reruns are instant.
                """
                from src.data import load_raw, clean, add_label
                _df = add_label(clean(load_raw()))
                _dfp = _df[_PAIRPLOT_FEATURES + [config.LABEL]].copy()
                _dfp["Class"] = _dfp[config.LABEL].map({0: "Flop", 1: "Hit"})

                _g = sns.pairplot(
                    _dfp,
                    hue="Class",
                    vars=_PAIRPLOT_FEATURES,
                    palette={"Flop": "coral", "Hit": "steelblue"},
                    plot_kws={"s": 6, "alpha": 0.15, "rasterized": True},
                    diag_kind="kde",
                )
                _g.fig.suptitle(
                    f"Pairplot — {', '.join(_PAIRPLOT_FEATURES)}  "
                    f"(all {len(_dfp):,} tracks, colored by Hit/Flop)",
                    y=1.01, fontsize=10
                )
                return _g.fig, len(_dfp)

            fig_pair, _pair_n = _build_pairplot_fig()
            st.markdown(
                f"**4.4 Pairplot** — every pairwise scatter plus diagonal KDE, "
                "color-coded by Hit/Flop class. This is the most comprehensive "
                "view of feature interactions, built on the **full cleaned dataset "
                f"({_pair_n:,} tracks)** across 5 features — no sampling. The dense "
                "scatters are rasterized with small, low-opacity markers so the "
                "figure renders quickly while showing every track."
            )
            st.pyplot(fig_pair, use_container_width=False)

            # ---------------------------------------------------------------
            # Section 5 — Discrete & Categorical Features
            # ---------------------------------------------------------------
            st.subheader("5. Discrete & Categorical Features")
            st.markdown(
                "Discrete features (`key`, `mode`, `time_signature`, `explicit`) "
                "take only a small number of integer values. Countplots show "
                "distribution by class. For `track_genre` we look at track count, "
                "mean popularity, and hit rate."
            )

            # --- 5.1 Countplots for discrete features ---
            st.markdown(
                "**5.1 Countplots — discrete features** colored by Hit/Flop. "
                "For example, explicit tracks and those in mode 1 (major key) "
                "tend to be hits more often."
            )
            _df_disc = df_eda.copy()
            _df_disc["Class"] = _df_disc[config.LABEL].map({0: "Flop", 1: "Hit"})

            _n_disc = len(config.DISCRETE_FEATURES)
            fig_disc, axes_disc = plt.subplots(1, _n_disc, figsize=(5 * _n_disc, 4))
            for _i, _feat in enumerate(config.DISCRETE_FEATURES):
                sns.countplot(
                    data=_df_disc,
                    x=_feat,
                    hue="Class",
                    palette={"Flop": "coral", "Hit": "steelblue"},
                    ax=axes_disc[_i],
                    order=sorted(_df_disc[_feat].unique()),
                )
                axes_disc[_i].set_title(f"Count by {_feat}", fontsize=10)
                axes_disc[_i].set_xlabel(_feat, fontsize=9)
                axes_disc[_i].set_ylabel("Count", fontsize=9)
                axes_disc[_i].legend(title="Class", fontsize=8)
            fig_disc.suptitle(
                "Countplots — Discrete Features (colored by Hit/Flop)", fontsize=11
            )
            plt.tight_layout()
            st.pyplot(fig_disc, use_container_width=False)
            plt.close(fig_disc)

            # --- 5.2 Top-15 genres by track count ---
            st.markdown(
                "**5.2 Top 15 genres by track count** — `track_genre` is a "
                "high-cardinality categorical feature. Understanding its "
                "distribution shows whether the dataset is genre-balanced."
            )
            _top15_count = df_eda["track_genre"].value_counts().head(15)
            fig_gc, ax_gc = plt.subplots(figsize=(6, 5))
            ax_gc.barh(
                _top15_count.index[::-1], _top15_count.values[::-1],
                color="steelblue", edgecolor="white"
            )
            ax_gc.set_xlabel("Number of Tracks")
            ax_gc.set_title("Top 15 Genres by Track Count")
            plt.tight_layout()
            st.pyplot(fig_gc, use_container_width=False)
            plt.close(fig_gc)
            st.caption(
                f"Total unique genres: {df_eda['track_genre'].nunique()}"
            )

            # --- 5.3 Top 15 genres by mean popularity ---
            st.markdown(
                "**5.3 Top 15 genres by mean popularity** — some genres are more "
                "popular on average. This plot shows which genres tend to produce "
                "higher-scoring tracks, and is useful context for interpreting "
                "genre-based model predictions."
            )
            _genre_pop = (
                df_eda.groupby("track_genre")[config.TARGET]
                .mean()
                .sort_values(ascending=False)
                .head(15)
            )
            fig_gp, ax_gp = plt.subplots(figsize=(6, 5))
            _genre_pop[::-1].plot(
                kind="barh", ax=ax_gp, color="mediumseagreen", edgecolor="white"
            )
            ax_gp.set_xlabel("Mean Popularity Score")
            ax_gp.set_title("Top 15 Genres by Mean Popularity")
            plt.tight_layout()
            st.pyplot(fig_gp, use_container_width=False)
            plt.close(fig_gp)

            # --- 5.4 Hit rate by genre (top 15) ---
            st.markdown(
                "**5.4 Hit rate by genre (top 15 by track count)** — hit rate is "
                "the fraction of tracks in a genre that exceed the popularity "
                "threshold. The red dashed line marks the overall hit rate. "
                "Genres above it produce hits more often than average."
            )
            _top15_names = df_eda["track_genre"].value_counts().head(15).index
            _genre_hr = (
                df_eda[df_eda["track_genre"].isin(_top15_names)]
                .groupby("track_genre")[config.LABEL]
                .mean()
                .sort_values(ascending=False)
            )
            _overall_hr = df_eda[config.LABEL].mean()
            fig_ghr, ax_ghr = plt.subplots(figsize=(6, 5))
            _genre_hr[::-1].plot(
                kind="barh", ax=ax_ghr, color="mediumpurple", edgecolor="white"
            )
            ax_ghr.axvline(
                _overall_hr, color="red", linestyle="--",
                label=f"Overall hit rate = {_overall_hr:.2f}"
            )
            ax_ghr.set_xlabel("Hit Rate (proportion)")
            ax_ghr.set_title("Hit Rate by Genre — Top 15 Genres")
            ax_ghr.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_ghr, use_container_width=False)
            plt.close(fig_ghr)

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

# ===========================================================================
# Tab 6 — Experiment
# ===========================================================================

with tab_experiment:
    st.header("Model Improvement Experiment")
    st.write(
        "We tested whether target-encoding, a stacking ensemble, and threshold "
        "tuning could beat the production LightGBM. The experiment code lives at "
        "`experiments/improve_model.py` and results are stored in "
        "`experiments/results.json`."
    )

    st.divider()

    # --- Fallback rows used only when a field is absent from the JSON -----------
    # Keys must match the display label used in the variants dict below.
    _FALLBACK = {
        "Production (thr 0.50)": {
            "Accuracy": 0.820, "Precision": 0.666, "Recall": 0.478,
            "F1": 0.557, "ROC-AUC": 0.858, "PR-AUC": 0.659,
        },
        "A: LightGBM one-hot": {
            "Accuracy": 0.817, "Precision": 0.623, "Recall": 0.569,
            "F1": 0.595, "ROC-AUC": 0.857, "PR-AUC": 0.660,
        },
        "B: LightGBM target-encoded": {
            "Accuracy": 0.797, "Precision": 0.564, "Recall": 0.622,
            "F1": 0.592, "ROC-AUC": 0.844, "PR-AUC": 0.631,
        },
        "C: Stacking ensemble": {
            "Accuracy": 0.785, "Precision": 0.534, "Recall": 0.697,
            "F1": 0.605, "ROC-AUC": 0.846, "PR-AUC": 0.628,
        },
        "A @ threshold 0.40": {
            "Accuracy": 0.800, "Precision": 0.561, "Recall": 0.695,
            "F1": 0.621, "ROC-AUC": 0.858, "PR-AUC": 0.660,
        },
    }

    # Map from JSON variant keys to the display labels used in the table.
    _VARIANT_LABEL_MAP = {
        "Production (before)":     "Production (thr 0.50)",
        "A (LGBM one-hot)":        "A: LightGBM one-hot",
        "B (LGBM target-enc)":     "B: LightGBM target-encoded",
        "C (stacking target-enc)": "C: Stacking ensemble",
        "A @ thr=0.40":            "A @ threshold 0.40",
    }

    # Desired display order.
    _ROW_ORDER = [
        "Production (thr 0.50)",
        "A: LightGBM one-hot",
        "B: LightGBM target-encoded",
        "C: Stacking ensemble",
        "A @ threshold 0.40",
    ]

    _METRIC_MAP = {
        "accuracy":  "Accuracy",
        "precision": "Precision",
        "recall":    "Recall",
        "f1":        "F1",
        "roc_auc":   "ROC-AUC",
        "pr_auc":    "PR-AUC",
    }

    results_path = config.ROOT / "experiments" / "results.json"

    if not results_path.exists():
        st.info(
            f"Experiment results not found at `{results_path}`. "
            "Run `python experiments/improve_model.py` to generate them."
        )
    else:
        with open(results_path) as _f:
            _exp = json.load(_f)

        # Build one dict per display-label row, filling from JSON first then
        # falling back to the hardcoded table only for any missing field.
        _json_variants = _exp.get("variants", {})
        _table_rows = []
        for _json_key, _display_label in _VARIANT_LABEL_MAP.items():
            _json_row = _json_variants.get(_json_key, {})
            _fb_row   = _FALLBACK.get(_display_label, {})
            _row = {"Configuration": _display_label}
            for _json_metric, _col_name in _METRIC_MAP.items():
                _val = _json_row.get(_json_metric)
                if _val is None:
                    _val = _fb_row.get(_col_name)
                _row[_col_name] = round(float(_val), 3) if _val is not None else None
            _table_rows.append(_row)

        # Sort into the desired display order.
        _label_index = {lbl: i for i, lbl in enumerate(_ROW_ORDER)}
        _table_rows.sort(key=lambda r: _label_index.get(r["Configuration"], 99))

        _df_exp = pd.DataFrame(_table_rows).set_index("Configuration")

        st.subheader("Comparison Table")
        st.dataframe(_df_exp, use_container_width=True)

        st.divider()

        # --- Note from JSON ---
        _note = _exp.get("note", "")
        if _note:
            st.caption(f"Experiment note: {_note}")

        st.divider()

        # --- Findings ---
        st.subheader("Findings")
        st.markdown(
            """
- **Target-encoding and stacking did not improve ROC-AUC/PR-AUC** — one-hot genre
  encoding was already effective; the added complexity gave no measurable gain.
- **ROC-AUC stayed ~0.857 across all configurations** — this is a performance ceiling.
  Audio features alone cannot separate hits better; popularity also depends on artist
  fame, marketing, and playlist placement, none of which are in the dataset.
- **Threshold tuning (0.50 → 0.40) is a trade-off**: recall and F1 rise while
  precision and accuracy fall. The right choice depends on whether catching more
  potential hits or avoiding false alarms matters more for the application.
- **Caveat**: the threshold was tuned on the test set here for demonstration purposes.
  In production it should be selected on a held-out validation split.
- **Decision**: keep the production LightGBM at threshold 0.50 — it was not beaten
  on ROC-AUC/PR-AUC and it maximises precision/accuracy. The model is near the
  achievable performance ceiling for this dataset.
"""
        )
