"""
app.py
-------
Streamlit web app for the Speech Emotion Recognition project.
Gives a proper visual interface (waveform + predicted emotion + confidence chart)
instead of plain terminal output — great for demos / LinkedIn video.

Supports two input modes:
  1. Upload a .wav file
  2. Record live from your microphone (browser mic)

Author: Wareesha Khan
Project: CodeAlpha Machine Learning Internship — Task 2

Run with:
    streamlit run app.py
"""

import os
import sys
import io
import tempfile
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import streamlit as st
import joblib
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from feature_extraction import extract_features  # noqa: E402

MODEL_DIR = "models"

EMOTION_EMOJIS = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲",
    "neutral": "😐",
    "calm": "😌",
}

st.set_page_config(page_title="Speech Emotion Recognition | Wareesha Khan", page_icon="🎙️", layout="centered")


@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "emotion_model.keras")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
        return None, None, None

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    return model, scaler, le


def save_audio_to_wav(raw_bytes: bytes) -> str:
    """
    Takes raw audio bytes (from file upload OR mic recording), decodes them
    with soundfile/librosa (which handles wav/ogg/webm containers), and
    re-saves as a clean, standard .wav file on disk. This avoids format
    mismatches that can silently break feature extraction for mic recordings.
    """
    audio, sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    sf.write(tmp_path, audio, sr, format="WAV", subtype="PCM_16")
    return tmp_path


def analyze_audio(raw_bytes: bytes, scaler, le, model):
    """Runs the full pipeline: decode -> waveform plot -> feature extraction -> prediction."""
    tmp_path = None
    try:
        tmp_path = save_audio_to_wav(raw_bytes)

        # Waveform plot
        audio, sr = librosa.load(tmp_path, sr=None)
        if len(audio) == 0:
            st.warning("⚠️ The recording seems empty. Please record a bit longer (2-3 seconds) and try again.")
            return

        fig, ax = plt.subplots(figsize=(8, 2.5))
        librosa.display.waveshow(audio, sr=sr, ax=ax, color="#6366F1")
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        st.pyplot(fig)
        plt.close(fig)

        with st.spinner("Analyzing speech..."):
            features = extract_features(tmp_path)
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_scaled = features_scaled.reshape(1, features_scaled.shape[1], 1)
            probs = model.predict(features_scaled, verbose=0)[0]

        pred_idx = np.argmax(probs)
        pred_label = le.classes_[pred_idx]
        emoji = EMOTION_EMOJIS.get(pred_label, "🎭")

        st.markdown(f"## {emoji} Predicted Emotion: **{pred_label.upper()}**")
        st.progress(float(probs[pred_idx]))
        st.write(f"Confidence: **{probs[pred_idx]*100:.2f}%**")

        st.subheader("Confidence across all emotions")
        df = pd.DataFrame({
            "Emotion": le.classes_,
            "Confidence": probs * 100,
        }).sort_values("Confidence", ascending=False)

        st.bar_chart(df.set_index("Emotion"))

    except Exception as e:
        st.error(f"❌ Something went wrong while analyzing the audio: {e}")
        with st.expander("Show full error details (for debugging)"):
            st.code(traceback.format_exc())
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


st.title("🎙️ Speech Emotion Recognition")
st.caption("CodeAlpha Machine Learning Internship — Task 2 | MFCC + CNN-LSTM")
st.markdown("**Developed by: Wareesha Khan**")

model, scaler, le = load_artifacts()

if model is None:
    st.error(
        "⚠️ No trained model found in `models/`. "
        "Please train it first by running:\n\n"
        "`python src/train_model.py --data_path data`"
    )
    st.stop()

tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Live"])

with tab_upload:
    st.write("Upload a `.wav` audio clip and the model will predict the speaker's emotion.")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"], key="uploader")

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        analyze_audio(uploaded_file.getvalue(), scaler, le, model)
    else:
        st.info("👆 Upload an audio file to get started.")

with tab_record:
    st.write("Click the mic, record 2-3 seconds of speech, then stop — the model will analyze it live.")
    mic_audio = st.audio_input("Record your voice", key="mic")

    if mic_audio is not None:
        st.audio(mic_audio, format="audio/wav")
        analyze_audio(mic_audio.getvalue(), scaler, le, model)
    else:
        st.info("🎙️ Click the microphone icon above to record your voice.")

st.divider()
st.caption("Built for CodeAlpha Machine Learning Internship — Task 2")
st.caption("👩‍💻 Developed by **Wareesha Khan**")