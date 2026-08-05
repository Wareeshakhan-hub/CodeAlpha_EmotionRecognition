"""
feature_extraction.py
----------------------
Extracts audio features (MFCC, Chroma, Mel-Spectrogram) from speech files
for the Speech Emotion Recognition project (CodeAlpha - Task 2).

Author: Wareesha Khan

Works out of the box with the RAVDESS dataset filename convention:
    03-01-06-01-02-01-12.wav
    modality-vocalChannel-EMOTION-intensity-statement-repetition-actor

Emotion codes (RAVDESS):
    01 = neutral, 02 = calm, 03 = happy, 04 = sad,
    05 = angry, 06 = fearful, 07 = disgust, 08 = surprised

If you use TESS or EMO-DB instead, only the `emotion_from_filename`
function needs to change (see comments inside it).
"""

import os
import glob
import numpy as np
import librosa

# ----------------------------------------------------------------------
# RAVDESS emotion code -> human readable label
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# The subset of emotions we will actually train on.
# (You can add/remove emotions here — fewer classes = higher accuracy)
OBSERVED_EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised", "calm"]


def emotion_from_filename(file_path: str) -> str:
    """
    Extracts the emotion label from a RAVDESS-style filename.
    Change this function if you switch datasets (TESS/EMO-DB have
    different naming conventions).
    """
    filename = os.path.basename(file_path)
    parts = filename.split("-")
    code = parts[2]
    return RAVDESS_EMOTIONS.get(code, "unknown")


def extract_features(file_path: str, mfcc=True, chroma=True, mel=True, n_mfcc=40):
    """
    Loads an audio file and extracts a fixed-length feature vector.

    Parameters
    ----------
    file_path : str
        Path to the .wav file.
    mfcc, chroma, mel : bool
        Which feature types to include.
    n_mfcc : int
        Number of MFCC coefficients to extract.

    Returns
    -------
    np.ndarray
        1-D feature vector (mean-pooled over time).
    """
    with librosa.warnings.catch_warnings() if hasattr(librosa, "warnings") else _dummy_ctx():
        pass

    audio, sample_rate = librosa.load(file_path, sr=None, res_type="kaiser_fast")
    result = np.array([])

    if chroma:
        stft = np.abs(librosa.stft(audio))

    if mfcc:
        mfccs = np.mean(
            librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0
        )
        result = np.hstack((result, mfccs))

    if chroma:
        chroma_feat = np.mean(
            librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
        )
        result = np.hstack((result, chroma_feat))

    if mel:
        mel_feat = np.mean(
            librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0
        )
        result = np.hstack((result, mel_feat))

    return result


class _dummy_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def load_dataset(data_path: str, test_size: float = 0.2):
    """
    Walks through `data_path` (RAVDESS folder structure: Actor_01/, Actor_02/, ...),
    extracts features for every .wav file, and returns a train/test split.

    Parameters
    ----------
    data_path : str
        Root folder containing the RAVDESS actor subfolders.
    test_size : float
        Fraction of data reserved for testing.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    X, y = [], []
    files = glob.glob(os.path.join(data_path, "Actor_*", "*.wav"))

    if not files:
        raise FileNotFoundError(
            f"No .wav files found under '{data_path}'. "
            "Download the RAVDESS dataset and place the Actor_01..Actor_24 "
            "folders inside the 'data/' directory."
        )

    for file in tqdm(files, desc="Extracting features"):
        emotion = emotion_from_filename(file)
        if emotion not in OBSERVED_EMOTIONS:
            continue
        try:
            features = extract_features(file)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue
        X.append(features)
        y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
