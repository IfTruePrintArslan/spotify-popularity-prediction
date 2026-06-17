"""Central configuration file for the Spotify hit-prediction pipeline.

All paths, numeric hyperparameters, and feature lists live here so every other
module can import them from a single source.  Changing a value here propagates
everywhere automatically — no hunting through multiple files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment-level constants
# ---------------------------------------------------------------------------

# Seed used for every random operation (train/test split, CV shuffle, SMOTE,
# model training).  Fixing it makes results reproducible run-to-run.
RANDOM_STATE = 42

# A track is labelled a 'hit' if its Spotify popularity score is AT LEAST this
# value.  Popularity ranges 0-100; 50 is a reasonable mid-point that keeps the
# positive class large enough to learn from while still being selective.
HIT_THRESHOLD = 50

# Fraction of data held out for the final test evaluation (20 %).
TEST_SIZE = 0.2

# Number of folds used in cross-validation.  5-fold is the standard choice:
# enough folds to get a stable estimate without taking too long to run.
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Memory / runtime guards
# ---------------------------------------------------------------------------
# The raw training set is ~90k rows; once SMOTE oversamples the minority class
# and the ~113 genres are one-hot encoded, the in-memory matrix gets very wide.
# RandomizedSearchCV refits a model on every (param x fold) combination, so
# tuning on the full set repeatedly copies that big matrix across workers and
# exhausted RAM (the machine crashed).  We therefore tune on a STRATIFIED
# subsample of this size, then REFIT the chosen model on the full training set.
TUNE_SAMPLE_SIZE = 30000

# Cap on parallel workers used by RandomizedSearchCV / cross_val_score.  Each
# worker holds its own copy of the (post-SMOTE) data, so n_jobs=-1 would spawn
# one copy per CPU core and blow up memory.  2 keeps some speed-up while
# bounding peak RAM.  Tree estimators get their own modest n_jobs (see train.py)
# so search-level x estimator-level parallelism does not multiply.
SEARCH_N_JOBS = 2

# n_jobs handed to the tree ensembles themselves.  Kept small so it does not
# multiply with SEARCH_N_JOBS (2 search workers x many estimator threads each
# would over-subscribe the CPU and duplicate data in memory).
MODEL_N_JOBS = 2

# ---------------------------------------------------------------------------
# Directory / file paths (all derived from the project root automatically)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
RAW_CSV = DATA_RAW / "dataset.csv"
MODEL_PATH = MODELS_DIR / "best_classifier.joblib"

# ---------------------------------------------------------------------------
# Target and label names
# ---------------------------------------------------------------------------

# Raw regression target from the Kaggle CSV (0-100 integer).
TARGET = "popularity"

# Binary label column created by data.add_label() from TARGET + HIT_THRESHOLD.
LABEL = "hit"

# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

# Continuous numeric features — these are float-valued audio descriptors with
# a real numeric range.  They receive IQR outlier capping (winsorization) before
# scaling, because extreme outliers can distort StandardScaler and hurt models.
CONTINUOUS_FEATURES = [
    "duration_ms", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

# Discrete / near-binary numeric features — these take only a small number of
# integer values (e.g. key: 0-11, mode: 0/1, explicit: 0/1).  Applying IQR
# capping to them would collapse their entire range to a constant, destroying
# the information they carry.  They skip capping and are only (optionally) scaled.
DISCRETE_FEATURES = ["key", "mode", "time_signature", "explicit"]

# Union of both numeric lists; kept so other modules can reference all numerics
# in one place without re-combining the two lists themselves.
NUMERIC_FEATURES = CONTINUOUS_FEATURES + DISCRETE_FEATURES

# The one categorical feature: encoded with OneHotEncoder inside the pipeline.
CATEGORICAL_FEATURES = ["track_genre"]

# All features passed into the model (numerics + categoricals).
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that exist in the raw CSV but must be dropped before modelling.
# "Unnamed: 0" and "Unnamed: 0.1" are artifact index columns that Kaggle CSVs
# sometimes include (the raw CSV has both variants).
# track_id, artists, album_name, track_name are unique identifiers / free-text
# strings — keeping them would either cause data leakage or produce thousands
# of meaningless one-hot columns.
DROP_COLS = ["Unnamed: 0", "Unnamed: 0.1", "track_id", "artists", "album_name", "track_name"]

# ---------------------------------------------------------------------------
# Ensure required directories exist (runs on import, harmless if they exist)
# ---------------------------------------------------------------------------

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
