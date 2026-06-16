from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

EXPECTED_COLS = (
    set(config.DROP_COLS)
    | set(config.NUMERIC_FEATURES)
    | set(config.CATEGORICAL_FEATURES)
    | {config.TARGET}
)


def load_raw(path=None) -> pd.DataFrame:
    path = Path(path) if path else config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Download it first "
            f"(see README: Kaggle 'maharshipandya/spotify-tracks-dataset')."
        )
    df = pd.read_csv(path)
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="track_id", keep="first").copy()
    keep = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + [config.TARGET]
    df["explicit"] = df["explicit"].astype(int)
    df = df.dropna(subset=keep)
    df = df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])
    return df.reset_index(drop=True)


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.LABEL] = (df[config.TARGET] >= config.HIT_THRESHOLD).astype(int)
    return df


def split(df: pd.DataFrame, target: str = config.LABEL, stratify: bool = True):
    X = df[config.ALL_FEATURES]
    y = df[target]
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=strat,
    )
