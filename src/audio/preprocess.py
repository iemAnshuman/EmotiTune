import os
import yaml
import librosa
import numpy as np
from librosa.feature import chroma_stft, spectral_contrast, tonnetz, mfcc
from librosa.effects import harmonic

# Import our new logger
from src.utils import setup_logger

# --- CONFIG & LOGGER SETUP ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger('preprocess', log_file='logs/data_processing.log')

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
        y, sr = librosa.load(file_path, sr=CONFIG['audio']['sampling_rate'], res_type='kaiser_fast')
        
        # If audio is too short, it might cause issues with some features
        if len(y) < sr * 0.1: # less than 0.1 seconds
             logger.warning(f"Audio file too short, skipping: {file_path}")
             return None

        mfcc_feat = mfcc(y=y, sr=sr, n_mfcc=CONFIG['audio']['n_mfcc'])
        chroma_feat = chroma_stft(y=y, sr=sr)
        contrast_feat = spectral_contrast(y=y, sr=sr)
        tonnetz_feat = tonnetz(y=harmonic(y), sr=sr)

        mfcc_feat = _pad_or_trim(mfcc_feat)
        chroma_feat = _pad_or_trim(chroma_feat)
        contrast_feat = _pad_or_trim(contrast_feat)
        tonnetz_feat = _pad_or_trim(tonnetz_feat)

        return np.vstack([mfcc_feat, chroma_feat, contrast_feat, tonnetz_feat])
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None

def parse_ravdess_filename(filename):
    parts = filename.split('-')
    if len(parts) < 3: return None
    return RAVDESS_EMOTIONS.get(parts[2])

def parse_crema_filename(filename):
    parts = filename.split('_')
    if len(parts) < 3: return None
    return CREMA_EMOTIONS.get(parts[2])

def load_ravdess(path, dataset_name="RAVDESS"):
    X, y = [], []
    full_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full_path):
        logger.warning(f"{dataset_name} path not found: {full_path}")
        return [], []

    logger.info(f"Loading {dataset_name} from {full_path}")
    for root, _, files in os.walk(full_path):
        for file in files:
            if file.endswith('.wav'):
                label = parse_ravdess_filename(file)
                if label:
                    features = extract_audio_features(os.path.join(root, file))
                    if features is not None:
                        X.append(features)
                        y.append(label)
    logger.info(f"Loaded {len(X)} samples from {dataset_name}")
    return X, y

def load_crema():
    X, y = [], []
    full_path = os.path.join(PROJECT_ROOT, CONFIG['data']['raw']['crema_d'])
    if not os.path.exists(full_path):
        logger.warning(f"CREMA-D path not found: {full_path}")
        return [], []

    logger.info(f"Loading CREMA-D from {full_path}")
    for file in os.listdir(full_path):
        if file.endswith('.wav'):
            label = parse_crema_filename(file)
            if label:
                features = extract_audio_features(os.path.join(full_path, file))
                if features is not None:
                    X.append(features)
                    y.append(label)
    logger.info(f"Loaded {len(X)} samples from CREMA-D")
    return X, y

def load_all_datasets():
    logger.info("Starting dataset loading process...")
    X_rav, y_rav = load_ravdess(CONFIG['data']['raw']['ravdess_speech'], "RAVDESS Speech")
    X_song, y_song = load_ravdess(CONFIG['data']['raw']['ravdess_song'], "RAVDESS Song")
    X_crema, y_crema = load_crema()

    X = X_rav + X_song + X_crema
    y = y_rav + y_song + y_crema

    if len(X) == 0:
        logger.critical("No data loaded! Check your paths in config.yaml")
        raise RuntimeError("No data loaded")

    logger.info(f"Total samples loaded: {len(X)}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    load_all_datasets()