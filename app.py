"""
app.py
-------
Streamlit web app for the Speech Emotion Recognition project.
Gives a proper visual interface (waveform + predicted emotion + confidence chart)
instead of plain terminal output — great for demos / LinkedIn video.

Supports two input modes:
  1. Upload a .wav file
  2. Record live from your microphone (browser mic) — requires Streamlit >= 1.36

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
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

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")


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


def analyze_audio(tmp_path: str, scaler, le, model):
    """Runs the full pipeline: waveform plot + feature extraction + prediction."""
    # Waveform plot
    audio, sr = librosa.load(tmp_path, sr=None)
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


st.title("🎙️ Speech Emotion Recognition")
st.caption("CodeAlpha Machine Learning Internship — Task 2 | MFCC + CNN-LSTM")
st.caption(f"Streamlit version running: `{st.__version__}`")

model, scaler, le = load_artifacts()

if model is None:
    st.error(
        "⚠️ No trained model found in `models/`. "
        "Please train it first by running:\n\n"
        "`python src/train_model.py --data_path data`"
    )
    st.stop()

# st.audio_input was added in Streamlit 1.36 — guard against older versions
# so the app degrades gracefully instead of crashing with an AttributeError.
HAS_MIC_INPUT = hasattr(st, "audio_input")

if HAS_MIC_INPUT:
    tab_upload, tab_record = st.tabs(["📁 Upload Audio", "🎤 Record Live"])
else:
    tab_upload = st.container()
    tab_record = None
    st.warning(
        f"🎤 Live mic recording needs Streamlit ≥ 1.36, but this app is running "
        f"**{st.__version__}**. Update `requirements.txt` to `streamlit==1.38.0` "
        f"(or newer), then go to **Manage app → Reboot** (or delete and redeploy) "
        f"on Streamlit Cloud so it reinstalls the correct version. "
        f"File upload still works below in the meantime."
    )

with tab_upload:
    st.write("Upload a `.wav` audio clip and the model will predict the speaker's emotion.")
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"], key="uploader")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.audio(uploaded_file, format="audio/wav")
        analyze_audio(tmp_path, scaler, le, model)
        os.unlink(tmp_path)
    else:
        st.info("👆 Upload an audio file to get started.")

if HAS_MIC_INPUT:
    with tab_record:
        st.write("Click the mic, record a few seconds of speech, then stop — the model will analyze it live.")
        mic_audio = st.audio_input("Record your voice", key="mic")

        if mic_audio is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(mic_audio.read())
                tmp_path = tmp.name

            st.audio(mic_audio, format="audio/wav")
            analyze_audio(tmp_path, scaler, le, model)
            os.unlink(tmp_path)
        else:
            st.info("🎙️ Click the microphone icon above to record your voice.")

st.divider()
st.caption("Built for CodeAlpha Machine Learning Internship — Task 2")
