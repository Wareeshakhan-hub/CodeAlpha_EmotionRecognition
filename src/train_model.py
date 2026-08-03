"""
train_model.py
----------------
Trains a CNN + LSTM deep learning model for Speech Emotion Recognition
using MFCC / Chroma / Mel-Spectrogram features extracted from RAVDESS audio.

Usage:
    python src/train_model.py --data_path data --epochs 60

CodeAlpha - Machine Learning Internship - Task 2
Author: Wareesha Khan
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from feature_extraction import load_dataset


def build_model(input_shape, num_classes):
    """CNN + LSTM hybrid architecture for emotion classification."""
    model = Sequential([
        Conv1D(256, kernel_size=5, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_history.png"))
    plt.close()


def plot_confusion(y_true, y_pred, labels, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Speech Emotion Recognition")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()


def main(args):
    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading dataset & extracting features (this can take a few minutes)...")
    X_train, X_test, y_train, y_test = load_dataset(args.data_path, test_size=args.test_size)
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]} | Feature dim: {X_train.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    y_train_cat = to_categorical(y_train_enc, num_classes)
    y_test_cat = to_categorical(y_test_enc, num_classes)

    # Reshape for Conv1D/LSTM: (samples, timesteps, features=1)
    X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_r = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_model(input_shape=(X_train_r.shape[1], 1), num_classes=num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train_r, y_train_cat,
        validation_data=(X_test_r, y_test_cat),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_probs = model.predict(X_test_r)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test_enc

    acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Save artifacts
    model.save(os.path.join(args.model_dir, "emotion_model.keras"))
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(args.model_dir, "label_encoder.pkl"))

    plot_history(history, args.model_dir)
    plot_confusion(
        [le.classes_[i] for i in y_true],
        [le.classes_[i] for i in y_pred],
        list(le.classes_),
        args.model_dir,
    )

    print(f"\nModel, scaler, label encoder, and plots saved to '{args.model_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition model")
    parser.add_argument("--data_path", type=str, default="data", help="Path to RAVDESS dataset root")
    parser.add_argument("--model_dir", type=str, default="models", help="Where to save the trained model")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)