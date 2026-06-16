import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import streamlit as st

from src import config

st.set_page_config(page_title="Spotify Hit Predictor", page_icon="🎵")
st.title("🎵 Spotify Hit Predictor")

if not config.MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.run_pipeline` first.")
    st.stop()

model = joblib.load(config.MODEL_PATH)

st.sidebar.header("Audio features")
inputs = {
    "danceability": st.sidebar.slider("Danceability", 0.0, 1.0, 0.6),
    "energy": st.sidebar.slider("Energy", 0.0, 1.0, 0.7),
    "loudness": st.sidebar.slider("Loudness (dB)", -60.0, 2.0, -6.0),
    "speechiness": st.sidebar.slider("Speechiness", 0.0, 1.0, 0.05),
    "acousticness": st.sidebar.slider("Acousticness", 0.0, 1.0, 0.2),
    "instrumentalness": st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0),
    "liveness": st.sidebar.slider("Liveness", 0.0, 1.0, 0.15),
    "valence": st.sidebar.slider("Valence", 0.0, 1.0, 0.5),
    "tempo": st.sidebar.slider("Tempo (BPM)", 40.0, 220.0, 120.0),
    "duration_ms": st.sidebar.slider("Duration (ms)", 30000, 600000, 200000),
    "key": st.sidebar.slider("Key", 0, 11, 1),
    "mode": st.sidebar.selectbox("Mode", [0, 1], index=1),
    "time_signature": st.sidebar.selectbox("Time signature", [3, 4, 5], index=1),
    "explicit": int(st.sidebar.checkbox("Explicit", value=False)),
    "track_genre": st.sidebar.selectbox("Genre", ["pop", "rock", "hip-hop", "edm", "classical", "jazz"]),
}

row = pd.DataFrame([inputs])[config.ALL_FEATURES]
proba = float(model.predict_proba(row)[0, 1])
st.metric("Hit probability", f"{proba:.1%}")
st.progress(proba)
st.caption(f"'Hit' = popularity ≥ {config.HIT_THRESHOLD}. Model: {config.MODEL_PATH.name}")
