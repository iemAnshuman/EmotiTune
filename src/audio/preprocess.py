import os
import sys

# --- CRITICAL FIX: PATH SETUP ---
# This ensures the script can always find the 'src' module, 
# even when run from weird locations like Colab notebooks.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------

import yaml
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from librosa.feature import chroma_stft, spectral_contrast, tonnetz, mfcc
from librosa.effects import harmonic

from src.utils import setup_logger

CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

logger = setup_logger('preprocess', log_file='logs/data_processing.log')

RAVDESS_EMOTIONS = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}
CREMA_EMOTIONS = {'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'}

def _pad_or_trim(feature, max_pad_len=CONFIG['audio']['max_pad_len']):
    if feature.shape[1] < max_pad_len:
        pad_width = max_pad_len - feature.shape[1]
        return np.pad(feature, ((0,0),(0,pad_width)), mode='constant')
    else:
        return feature[:, :max_pad_len]

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=CONFIG['audio']['sampling_rate'], res_type='kaiser_fast')
        if len(y) < sr * 0.1: return None
        mfcc_feat = mfcc(y=y, sr=sr, n_mfcc=CONFIG['audio']['n_mfcc'])
        chroma_feat = chroma_stft(y=y, sr=sr)
        contrast_feat = spectral_contrast(y=y, sr=sr)
        tonnetz_feat = tonnetz(y=harmonic(y), sr=sr)
        return np.vstack([_pad_or_trim(f) for f in [mfcc_feat, chroma_feat, contrast_feat, tonnetz_feat]])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_file(file_path, dataset_type):
    filename = os.path.basename(file_path)
    label = RAVDESS_EMOTIONS.get(filename.split('-')[2]) if dataset_type == 'ravdess' and len(filename.split('-')) >= 3 else \
            CREMA_EMOTIONS.get(filename.split('_')[2]) if dataset_type == 'crema' and len(filename.split('_')) >= 3 else None
    if label:
        features = extract_audio_features(file_path)
        if features is not None: return features, label
    return None

def load_dataset_concurrent(path, dataset_type, max_workers=None):
    full_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full_path):
        logger.warning(f"Path not found: {full_path}")
        return [], []
    file_paths = [os.path.join(root, file) for root, _, files in os.walk(full_path) for file in files if file.endswith('.wav')]
    logger.info(f"Found {len(file_paths)} files in {dataset_type}. Processing...")
    X, y = [], []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, fp, dataset_type): fp for fp in file_paths}
        for future in tqdm(as_completed(futures), total=len(file_paths), desc=f"Loading {dataset_type}", leave=False):
            result = future.result()
            if result:
                X.append(result[0])
                y.append(result[1])
    return X, y

def load_all_datasets():
    logger.info("Starting concurrent dataset loading...")
    X_rav_speech, y_rav_speech = load_dataset_concurrent(CONFIG['data']['raw']['ravdess_speech'], 'ravdess')
    X_rav_song, y_rav_song = load_dataset_concurrent(CONFIG['data']['raw']['ravdess_song'], 'ravdess')
    X_crema, y_crema = load_dataset_concurrent(CONFIG['data']['raw']['crema_d'], 'crema')
    X = X_rav_speech + X_rav_song + X_crema
    y = y_rav_speech + y_rav_song + y_crema
    if len(X) == 0:
        logger.critical("No data loaded! Check paths in config.yaml")
        raise RuntimeError("No data loaded")
    logger.info(f"Total samples loaded: {len(X)}")
    
    # SAVE RESULTS
    processed_dir = os.path.join(PROJECT_ROOT, CONFIG['data']['processed'])
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, 'X.npy'), np.array(X))
    np.save(os.path.join(processed_dir, 'y.npy'), np.array(y))
    logger.info(f"Saved processed data to {processed_dir}")

if __name__ == "__main__":
    load_all_datasets()