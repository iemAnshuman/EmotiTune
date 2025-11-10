import torch
import numpy as np
import os
import yaml
import argparse
from src.audio.preprocess import extract_audio_features
from src.audio.model import CNNEmotionClassifier

# --- CONFIG SETUP ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

def load_model(device):
    MODELS_DIR = os.path.join(PROJECT_ROOT, CONFIG['model']['dir'])
    MODEL_PATH = os.path.join(MODELS_DIR, CONFIG['model']['name'])
    CLASSES_PATH = os.path.join(MODELS_DIR, 'classes.npy')

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError("Model or classes not found. Run training first!")

    # Load classes to know how many outputs the model has
    classes = np.load(CLASSES_PATH)
    
    # We need the input shape to initialize the model. 
    # We can cheat by extracting features from a dummy silence of the right length
    # OR we just know it from our preprocessing.
    # Based on standard librosa defaults: MFCC(40) + Chroma(12) + Contrast(7) + Tonnetz(6) = 65
    # Height = 65, Width = 174 (from config max_pad_len)
    input_shape = (1, 65, CONFIG['audio']['max_pad_len'])

    model = CNNEmotionClassifier(num_classes=len(classes), input_shape=input_shape)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, classes

def predict_emotion(audio_path, model, classes, device):
    print(f"Processing {audio_path}...")
    features = extract_audio_features(audio_path)
    
    if features is None:
        return "Error processing audio"

    # Prepare for model: Add batch dimension (1, ...) and channel dimension (1, ...)
    # Current shape is (65, 174) -> Want (1, 1, 65, 174)
    features_tensor = torch.tensor(features).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(features_tensor)
        # Get probabilities using Softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get the most likely class
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = classes[predicted_idx.item()]
        confidence = probabilities[0][predicted_idx.item()].item()

    return predicted_label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Emotitune: Predict emotion from an audio file.')
    parser.add_argument('audio_file', type=str, help='Path to the .wav or .mp3 file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model, classes = load_model(device)
        label, conf = predict_emotion(args.audio_file, model, classes, device)
        print("-" * 30)
        print(f"Predicted Emotion: {label.upper()}")
        print(f"Confidence: {conf*100:.2f}%")
        print("-" * 30)
    except Exception as e:
        print(f"An error occurred: {e}")