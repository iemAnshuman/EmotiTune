from fastapi import FastAPI, File, UploadFile, HTTPException
from contextlib import asynccontextmanager
import torch
import numpy as np
import os
import shutil
import yaml
import sys

# Add project root to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.audio.preprocess import extract_audio_features
from src.audio.model import CNNEmotionClassifier

# --- CONFIG SETUP ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

# Global state for model and classes
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model before the API starts receiving requests
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODELS_DIR = os.path.join(PROJECT_ROOT, CONFIG['model']['dir'])
    MODEL_PATH = os.path.join(MODELS_DIR, CONFIG['model']['name'])
    CLASSES_PATH = os.path.join(MODELS_DIR, 'classes.npy')

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise RuntimeError("Model not found. Train the model first.")

    classes = np.load(CLASSES_PATH)
    # Input shape based on known feature extraction (65 features x 174 time steps)
    input_shape = (1, 65, CONFIG['audio']['max_pad_len'])
    
    model = CNNEmotionClassifier(num_classes=len(classes), input_shape=input_shape)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    ml_models["cnn"] = model
    ml_models["classes"] = classes
    ml_models["device"] = device
    print(f"✅ Model loaded on {device}")
    yield
    # Clean up ML models and release resources
    ml_models.clear()

app = FastAPI(title="Emotitune API", description="Real-time Speech Emotion Recognition", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Emotitune API is running. POST audio files to /predict"}

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith(('.wav', '.mp3')):
         raise HTTPException(status_code=400, detail="Only .wav and .mp3 files are supported")

    # Save temporary file because librosa needs a file path
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Run inference
        features = extract_audio_features(temp_path)
        if features is None:
             raise HTTPException(status_code=422, detail="Could not extract audio features")

        device = ml_models["device"]
        features_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outputs = ml_models["cnn"](features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)
            
            predicted_label = ml_models["classes"][predicted_idx.item()]
            confidence = probabilities[0][predicted_idx.item()].item()

        return {
            "filename": file.filename,
            "emotion": predicted_label,
            "confidence": float(f"{confidence:.4f}")
        }

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)