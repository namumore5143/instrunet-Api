from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from predictor import (
    predict_instrument_percentages,
    generate_waveform,
    generate_spectrogram
)

app = FastAPI(title="Instrument AI API")

# ✅ ADD THIS (VERY IMPORTANT)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "Instrument AI API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    # Save uploaded file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction
    all_results, top4 = predict_instrument_percentages(temp_path)

    # Generate graphs
    waveform = generate_waveform(temp_path)
    spectrogram = generate_spectrogram(temp_path)

    # Delete temp file
    os.remove(temp_path)

    return {
        "success": True,

        "top_4": [
            {"instrument": name, "confidence": float(score * 100)}
            for name, score in top4
        ],

        "all_predictions": all_results,

        "waveform": waveform,

        "spectrogram": spectrogram
    }