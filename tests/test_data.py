import pandas as pd
import pytest
from src import config, data


def _raw_row(track_id="a", popularity=10, genre="pop"):
    return {
        "Unnamed: 0": 0, "track_id": track_id, "artists": "x",
        "album_name": "y", "track_name": "z", "popularity": popularity,
        "duration_ms": 200000, "explicit": False, "danceability": 0.5,
        "energy": 0.5, "key": 1, "loudness": -5.0, "mode": 1,
        "speechiness": 0.05, "acousticness": 0.1, "instrumentalness": 0.0,
        "liveness": 0.1, "valence": 0.5, "tempo": 120.0,
        "time_signature": 4, "track_genre": genre,
    }


def test_clean_drops_duplicate_track_ids():
    df = pd.DataFrame([_raw_row("a", genre="pop"), _raw_row("a", genre="rock")])
    cleaned = data.clean(df)
    assert len(cleaned) == 1


def test_clean_drops_identifier_columns():
    df = pd.DataFrame([_raw_row("a"), _raw_row("b")])
    cleaned = data.clean(df)
    for col in config.DROP_COLS:
        assert col not in cleaned.columns


def test_clean_casts_explicit_to_int():
    df = pd.DataFrame([_raw_row("a"), _raw_row("b")])
    cleaned = data.clean(df)
    assert set(cleaned["explicit"].unique()) <= {0, 1}


def test_clean_has_no_nulls():
    df = pd.DataFrame([_raw_row("a"), _raw_row("b")])
    cleaned = data.clean(df)
    assert cleaned.isna().sum().sum() == 0


def test_add_label_thresholds_popularity():
    df = pd.DataFrame([_raw_row("a", popularity=49), _raw_row("b", popularity=50)])
    labeled = data.add_label(data.clean(df))
    assert labeled[config.LABEL].tolist() == [0, 1]


def test_load_raw_rejects_bad_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        data.load_raw(bad)
