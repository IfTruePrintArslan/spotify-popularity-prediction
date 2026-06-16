from pathlib import Path

RANDOM_STATE = 42
HIT_THRESHOLD = 50
TEST_SIZE = 0.2
CV_FOLDS = 5

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RAW_CSV = DATA_RAW / "dataset.csv"
MODEL_PATH = MODELS_DIR / "best_classifier.joblib"

TARGET = "popularity"
LABEL = "hit"

# Continuous numeric features get outlier capping (IQR winsorization) + scaling.
CONTINUOUS_FEATURES = [
    "duration_ms", "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
# Discrete / near-binary numeric features: capping would destroy them, so they
# are passed through (optionally scaled) but never winsorized.
DISCRETE_FEATURES = ["key", "mode", "time_signature", "explicit"]
# Kept as the union so existing references (data cleaning, schema checks) work.
NUMERIC_FEATURES = CONTINUOUS_FEATURES + DISCRETE_FEATURES
CATEGORICAL_FEATURES = ["track_genre"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DROP_COLS = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
