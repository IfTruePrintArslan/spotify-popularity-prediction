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

NUMERIC_FEATURES = [
    "duration_ms", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature", "explicit",
]
CATEGORICAL_FEATURES = ["track_genre"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
DROP_COLS = ["Unnamed: 0", "track_id", "artists", "album_name", "track_name"]

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
