import os
import yaml
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from librosa.feature import chroma_stft, spectral_contrast, tonnetz, mfcc
from librosa.effects import harmonic

from src.utils import setup_logger

# --- CONFIG & LOGGER SETUP ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')

# Load config once at module level for worker processes
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Logger needs to be pickleable for multiprocessing, so we set it up inside functions or use a simple one here
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
    """
    Worker function that must be self-contained for pickling in multiprocessing.
    """
    try:
        # Re-load config inside worker if necessary, but module-level usually works on Linux/macOS.
        # For strict Windows compatibility, you might need to pass parameters explicitly.
        y, sr = librosa.load(file_path, sr=CONFIG['audio']['sampling_rate'], res_type='kaiser_fast')
        
        if len(y) < sr * 0.1:
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
        # Avoid complex logging in workers to prevent deadlocks
        print(f"Error processing {file_path}: {e}")
        return None

def parse_ravdess_filename(filename):
    parts = filename.split('-')
    if len(parts) < 3: return None
    return RAVDESS_EMOTIONS.get(parts[2])

def parse_crema_filename(filename):
    parts = filename.split('_')
    if len(parts) < 3: return None
    return CREMA_EMOTIONS.get(parts[2])

def process_file(file_path, dataset_type):
    """Helper to process a single file and return its features and label."""
    filename = os.path.basename(file_path)
    label = None
    if dataset_type == 'ravdess':
        label = parse_ravdess_filename(filename)
    elif dataset_type == 'crema':
        label = parse_crema_filename(filename)
    
    if label:
        features = extract_audio_features(file_path)
        if features is not None:
            return features, label
    return None

def load_dataset_concurrent(path, dataset_type, max_workers=None):
    """Loads a dataset using parallel processing."""
    full_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full_path):
        logger.warning(f"Path not found: {full_path}")
        return [], []

    file_paths = []
    for root, _, files in os.walk(full_path):
        for file in files:
            if file.endswith('.wav'):
                file_paths.append(os.path.join(root, file))

    logger.info(f"Found {len(file_paths)} files in {dataset_type}. Starting parallel processing...")
    
    X, y = [], []
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, fp, dataset_type): fp for fp in file_paths}
        
        # Process results as they complete with a progress bar
        for future in tqdm(as_completed(future_to_file), total=len(file_paths), desc=f"Loading {dataset_type}", leave=False):
            result = future.result()
            if result:
                features, label = result
                X.append(features)
                y.append(label)

    return X, y

def load_all_datasets():
    logger.info("Starting concurrent dataset loading...")
    
    # We can load datasets sequentially, but each dataset uses internal parallelism
    X_rav_speech, y_rav_speech = load_dataset_concurrent(CONFIG['data']['raw']['ravdess_speech'], 'ravdess')
    X_rav_song, y_rav_song = load_dataset_concurrent(CONFIG['data']['raw']['ravdess_song'], 'ravdess')
    X_crema, y_crema = load_dataset_concurrent(CONFIG['data']['raw']['crema_d'], 'crema')

    X = X_rav_speech + X_rav_song + X_crema
    y = y_rav_speech + y_rav_song + y_crema

    if len(X) == 0:
        logger.critical("No data loaded! Check your paths in config.yaml")
        raise RuntimeError("No data loaded")

    logger.info(f"Total samples loaded: {len(X)}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # This guard is crucial for multiprocessing on some platforms
    load_all_datasets()