"""
predict.py
-----------
Loads the trained model and predicts the emotion of a single .wav file.

Author: Wareesha Khan

Usage:
    python src/predict.py --file path/to/audio.wav
"""

import argparse
import numpy as np
import joblib
from tensorflow.keras.models import load_model

from feature_extraction import extract_features


def predict_emotion(file_path: str, model_dir: str = "models"):
    model = load_model(f"{model_dir}/emotion_model.keras")
    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    le = joblib.load(f"{model_dir}/label_encoder.pkl")

    features = extract_features(file_path)
    features = scaler.transform(features.reshape(1, -1))
    features = features.reshape(1, features.shape[1], 1)

    probs = model.predict(features)[0]
    pred_idx = np.argmax(probs)
    pred_label = le.classes_[pred_idx]

    print(f"\nPredicted Emotion: {pred_label.upper()}")
    print("\nConfidence per class:")
    for label, p in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
        print(f"  {label:10s}: {p*100:5.2f}%")

    return pred_label, dict(zip(le.classes_, probs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a speech .wav file")
    parser.add_argument("--file", type=str, required=True, help="Path to .wav file")
    parser.add_argument("--model_dir", type=str, default="models")
    args = parser.parse_args()
    predict_emotion(args.file, args.model_dir)
