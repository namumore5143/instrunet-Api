import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import base64
import io
IMG_SIZE = 128
SR = 22050

LABEL_NAMES = [
    "Cello", "Clarinet", "Flute", "Acoustic Guitar", "Electric Guitar",
    "Organ", "Piano", "Saxophone", "Trumpet", "Violin", "Human Voice"
]

MODEL_PATH = "multilabel_instrument_model_FINAL.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully")

def audio_to_mel_chunks(audio_path, chunk_duration=3.0):
    y, sr = librosa.load(audio_path, sr=SR)
    chunk_len = int(chunk_duration * sr)

    specs = []

    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]

        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))

        mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_db = librosa.util.fix_length(mel_db, size=IMG_SIZE, axis=1)
        mel_db = mel_db[:IMG_SIZE, :IMG_SIZE]

        mel_db = mel_db / 255.0
        mel_db = mel_db.reshape(IMG_SIZE, IMG_SIZE, 1)

        specs.append(mel_db)

    return np.array(specs)

def predict_instrument_percentages(audio_path, top_k=4):
    specs = audio_to_mel_chunks(audio_path)

    preds = []
    for spec in specs:
        p = model.predict(spec.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
        preds.append(p)

    avg_pred = np.mean(preds, axis=0)

    results = {LABEL_NAMES[i]: float(avg_pred[i] * 100) for i in range(len(LABEL_NAMES))}

    pairs = list(zip(LABEL_NAMES, avg_pred))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]

    return results, top

def generate_waveform(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)

    plt.figure(figsize=(8,3))
    plt.plot(y)
    plt.title("Waveform")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SR)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(8,3))
    librosa.display.specshow(mel_db, sr=sr)
    plt.title("Spectrogram")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
