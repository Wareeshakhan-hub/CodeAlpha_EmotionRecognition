# CodeAlpha_EmotionRecognition 🎙️😃😢😡

**CodeAlpha Machine Learning Internship — Task 2**
Speech Emotion Recognition using MFCC features + CNN-LSTM deep learning model.

👩‍💻 **Developed by: Wareesha Khan**

---

## 📌 Objective
Recognize human emotions (happy, sad, angry, fearful, disgust, surprised, neutral, calm) from raw speech audio using deep learning and speech signal processing.

## 🧠 Approach
1. **Feature Extraction** — MFCC (Mel-Frequency Cepstral Coefficients), Chroma, and Mel-Spectrogram features are extracted from each audio clip using `librosa`.
2. **Model** — A hybrid **CNN + LSTM** neural network (Conv1D layers for local pattern extraction + LSTM layers for temporal dependencies).
3. **Evaluation** — Accuracy, classification report (Precision/Recall/F1), and a confusion matrix.

## 📂 Project Structure
```
CodeAlpha_EmotionRecognition/
├── data/                     # Put the RAVDESS dataset here (Actor_01 ... Actor_24)
├── models/                   # Trained model, scaler, label encoder, plots (created after training)
├── src/
│   ├── feature_extraction.py # MFCC/Chroma/Mel feature extraction + dataset loader
│   ├── train_model.py        # Trains the CNN-LSTM model
│   └── predict.py            # Predicts emotion for a single new audio file
├── app.py                    # Streamlit GUI for visual demo (waveform + prediction)
├── requirements.txt
├── packages.txt              # System packages needed on Streamlit Cloud (ffmpeg, libsndfile1)
└── README.md
```

## 📥 Dataset Setup
This project uses the **RAVDESS** dataset (recommended — free & well-labeled).

1. Download it from Zenodo: https://zenodo.org/record/1188976 (file: `Audio_Speech_Actors_01-24.zip`)
2. Extract it, and place the `Actor_01` ... `Actor_24` folders directly inside the `data/` folder:
   ```
   data/
   ├── Actor_01/
   │   ├── 03-01-01-01-01-01-01.wav
   │   └── ...
   ├── Actor_02/
   └── ...
   ```
   You can also use **TESS** or **EMO-DB** — just update `emotion_from_filename()` in
   `src/feature_extraction.py` to match that dataset's naming convention.

## ⚙️ Installation
```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Train the Model
```bash
cd src
python train_model.py --data_path ../data --epochs 60
```
This will:
- Extract features from every `.wav` file
- Train the CNN-LSTM model with early stopping
- Save `emotion_model.keras`, `scaler.pkl`, `label_encoder.pkl` in `models/`
- Save `training_history.png` and `confusion_matrix.png` in `models/`

## 🔮 Predict Emotion on a New Audio File
```bash
python predict.py --file path/to/your_audio.wav
```
Example output:
```
Predicted Emotion: HAPPY

Confidence per class:
  happy     : 87.42%
  surprised :  6.15%
  neutral   :  3.20%
  ...
```

## 🖥️ Visual Demo App (Streamlit GUI)
Instead of just terminal output, you get a proper visual interface — perfect for
your LinkedIn demo video:

```bash
streamlit run app.py
```

This opens a browser window with **two ways to test the model**:
- **📁 Upload Audio** — upload any `.wav` file
- **🎤 Record Live** — record straight from your microphone in the browser and get an instant prediction

Both show the waveform, the predicted emotion (with emoji), and a confidence bar chart for every class.

> ⚠️ Make sure you've trained the model first (`python src/train_model.py ...`) —
> the app loads `models/emotion_model.keras`, `models/scaler.pkl`, and
> `models/label_encoder.pkl`, which are only created after training.
>
> 🎤 Live mic recording uses `st.audio_input`, which requires **Streamlit ≥ 1.36**
> (already pinned in `requirements.txt`) and browser microphone permission.

## ☁️ Deploy to the Cloud (get a shareable link)
You can host this app for free on **Streamlit Community Cloud** so anyone can open
it from a link — no installation needed. Great to include in your LinkedIn post
alongside the demo video.

1. Push this project to GitHub as `CodeAlpha_EmotionRecognition` (see checklist below).
   Make sure `models/emotion_model.keras`, `models/scaler.pkl`, and
   `models/label_encoder.pkl` are committed too — the deployed app needs them
   (they're excluded by `.gitignore` by default, so remove those lines or
   `git add -f` them before pushing).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"** → select your `CodeAlpha_EmotionRecognition` repo →
   set **Main file path** to `app.py` → click **Deploy**.
4. Streamlit Cloud automatically installs everything in `requirements.txt` and
   the system packages listed in `packages.txt` (needed for `librosa`/`soundfile`
   to work — `ffmpeg` and `libsndfile1`).
5. After a couple of minutes you'll get a public link like:
   `https://your-username-codealpha-emotionrecognition.streamlit.app`

> 💡 Note: mic recording (`st.audio_input`) needs HTTPS to access the browser
> microphone — Streamlit Cloud serves over HTTPS automatically, so it works
> out of the box once deployed.

## 📊 Results
After training on RAVDESS (8 emotion classes), this architecture typically achieves
**~65–75% test accuracy** — reasonable given RAVDESS's small size (~1,440 clips) and
number of classes. You can improve accuracy by:
- Using data augmentation (pitch shift, noise injection, time stretch)
- Adding more datasets (TESS, EMO-DB) to increase training data
- Reducing to fewer emotion classes (e.g., happy/sad/angry/neutral only)
- Trying a Bidirectional LSTM or adding an attention layer

## 🛠️ Tech Stack
- Python, NumPy, Pandas
- Librosa (audio processing)
- TensorFlow / Keras (CNN + LSTM)
- Scikit-learn (preprocessing, metrics)
- Matplotlib / Seaborn (visualization)

## 📝 Submission Checklist (per CodeAlpha instructions)
- [ ] Push this project to GitHub as **`CodeAlpha_EmotionRecognition`**
- [ ] (Optional but recommended) Deploy the app on Streamlit Community Cloud and get a live link
- [ ] Record a short video explaining the project and post it on LinkedIn, tagging **@CodeAlpha**
- [ ] Include the GitHub repo link (and live app link, if deployed) in the LinkedIn post
- [ ] Submit the task via the CodeAlpha submission form (shared in WhatsApp group)

---
*Built for the CodeAlpha Machine Learning Internship — Task 2: Emotion Recognition from Speech.*

**Author: Wareesha Khan**