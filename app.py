from fastapi import FastAPI, UploadFile, File
import shutil
import os
from predictor import predict_instrument_percentages

app = FastAPI(title="Instrument AI API")

@app.get("/")
def home():
    return {"status": "Instrument AI API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    all_results, top4 = predict_instrument_percentages(temp_path)

    os.remove(temp_path)

    return {
        "success": True,
        "top_4": [
            {"instrument": name, "confidence": float(score * 100)}
            for name, score in top4
        ],
        "all_predictions": all_results
    }
