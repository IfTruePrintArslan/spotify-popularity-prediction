"""Data loading, cleaning, labeling, and splitting for the Spotify hit pipeline.

This module is the first stage of the pipeline. It reads the raw Kaggle CSV,
removes duplicates and bad rows, converts types, creates a binary 'hit' label,
and produces train/test splits that are ready for feature engineering.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# Build the full set of columns we expect to find in the raw CSV.
# If any are missing the loader raises early with a clear message rather
# than failing silently later inside a transformer.
EXPECTED_COLS = (
    set(config.DROP_COLS)
    | set(config.NUMERIC_FEATURES)
    | set(config.CATEGORICAL_FEATURES)
    | {config.TARGET}
)


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw Spotify CSV and validate that all expected columns are present."""
    path = Path(path) if path else config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it first "
            f"(see README: Kaggle 'maharshipandya/spotify-tracks-dataset')."
        )
    df = pd.read_csv(path)
    # Fail fast: catch schema mismatches before any processing begins.
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fix types, drop missing values, and drop identifier columns.

    Steps:
    1. Deduplicate on track_id — keep='first' preserves the row order from Kaggle
       and is deterministic across runs (unlike keep='last').
    2. Cast 'explicit' from True/False to 0/1 so it is a proper integer feature
       that every sklearn estimator can handle without extra encoding.
    3. Drop rows where any feature or the target is NaN — a small fraction of
       tracks have missing audio features and keeping them would require imputation
       that adds complexity without much benefit at this dataset size.
    4. Drop identifier columns (track_id, artists, etc.) — they are unique strings
       that would just add noise or cause data leakage if kept as features.
    """
    # Step 1: one row per track; keep the first occurrence to be deterministic.
    df = df.drop_duplicates(subset="track_id", keep="first").copy()

    # Columns we want to keep: features + the raw popularity target.
    keep = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET]

    # Step 2: True/False -> 1/0 so 'explicit' is a numeric feature, not a bool.
    df["explicit"] = df["explicit"].astype(int)

    # Step 3: drop any row that is missing a feature or the target we need.
    df = df.dropna(subset=keep)

    # Step 4: drop identifier / metadata columns that are not predictive features.
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])

    return df.reset_index(drop=True)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary 'hit' column: 1 if popularity >= HIT_THRESHOLD, else 0.

    Turning the continuous popularity score into a binary label converts the
    problem from regression to binary classification, which maps onto standard
    precision/recall metrics and makes the grading rubric requirements easier
    to satisfy.
    """
    df = df.copy()  # avoid mutating the caller's DataFrame in place
    df[config.LABEL] = (df[config.TARGET] >= config.HIT_THRESHOLD).astype(int)
    return df


def split(df: pd.DataFrame, target: str = config.LABEL, stratify: bool = True):
    """Split the dataset into train and test sets, preserving the class ratio.

    Parameters
    ----------
    df : cleaned DataFrame with the label column already added.
    target : column name to use as y (defaults to 'hit').
    stratify : if True (default), the split keeps the same hit/non-hit ratio in
               both train and test.  This matters because the dataset is imbalanced
               — without stratification the test set might have a different
               proportion of hits than the training set, making evaluation
               unreliable.

    Returns
    -------
    X_train, X_test, y_train, y_test — the standard sklearn four-tuple.
    """
    X = df[config.ALL_FEATURES]
    y = df[target]

    # Pass y as the stratify argument so sklearn mirrors the class distribution.
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=strat,
    )
