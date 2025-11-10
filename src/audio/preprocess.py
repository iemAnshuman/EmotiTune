import os
import yaml
import librosa
import numpy as np

from librosa.feature import chroma_stft, spectral_contrast, tonnetz, mfcc
from librosa.effects import harmonic

# --- LOAD CONFIGURATION ---
# This ensures we can find config.yaml regardless of where we run the script from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

# Define emotion mappings
RAVDESS_EMOTIONS = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

CREMA_EMOTIONS = {
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful',
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

def _pad_or_trim(feature, max_pad_len=CONFIG['audio']['max_pad_len']):
    if feature.shape[1] < max_pad_len:
        pad_width = max_pad_len - feature.shape[1]
        return np.pad(feature, ((0,0),(0,pad_width)), mode='constant')
    else:
        return feature[:, :max_pad_len]

def extract_audio_features(file_path):
    try:
        # Load with standard rate from config
        y, sr = librosa.load(file_path, sr=CONFIG['audio']['sampling_rate'], res_type='kaiser_fast')

        # Extract features using config parameters
        mfcc_feat = mfcc(y=y, sr=sr, n_mfcc=CONFIG['audio']['n_mfcc'])
        chroma_feat = chroma_stft(y=y, sr=sr)
        contrast_feat = spectral_contrast(y=y, sr=sr)
        tonnetz_feat = tonnetz(y=harmonic(y), sr=sr)

        # Pad all to same time axis
        mfcc_feat = _pad_or_trim(mfcc_feat)
        chroma_feat = _pad_or_trim(chroma_feat)
        contrast_feat = _pad_or_trim(contrast_feat)
        tonnetz_feat = _pad_or_trim(tonnetz_feat)

        # Stack all features vertically
        return np.vstack([mfcc_feat, chroma_feat, contrast_feat, tonnetz_feat])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def parse_ravdess_filename(filename):
    parts = filename.split('-')
    return RAVDESS_EMOTIONS.get(parts[2])

def parse_crema_filename(filename):
    parts = filename.split('_')
    return CREMA_EMOTIONS.get(parts[2])

def load_ravdess(path):
    X, y = [], []
    # Normalize path to work from project root
    full_path = os.path.join(PROJECT_ROOT, path)
    for root, _, files in os.walk(full_path):
        for file in files:
            if file.endswith('.wav'):
                label = parse_ravdess_filename(file)
                if label:
                    features = extract_audio_features(os.path.join(root, file))
                    if features is not None:
                        X.append(features)
                        y.append(label)
    return X, y

def load_crema():
    X, y = [], []
    full_path = os.path.join(PROJECT_ROOT, CONFIG['data']['raw']['crema_d'])
    for file in os.listdir(full_path):
        if file.endswith('.wav'):
            label = parse_crema_filename(file)
            if label:
                features = extract_audio_features(os.path.join(full_path, file))
                if features is not None:
                    X.append(features)
                    y.append(label)
    return X, y

def load_all_datasets():
    print("Loading RAVDESS Speech...")
    X_rav, y_rav = load_ravdess(CONFIG['data']['raw']['ravdess_speech'])
    print("Loading RAVDESS Song...")
    X_song, y_song = load_ravdess(CONFIG['data']['raw']['ravdess_song'])
    print("Loading CREMA-D...")
    X_crema, y_crema = load_crema()

    # Combine all
    X = X_rav + X_song + X_crema
    y = y_rav + y_song + y_crema

    print(f"Total samples loaded: {len(X)}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_all_datasets()
    print(f"Feature matrix shape: {X.shape}")