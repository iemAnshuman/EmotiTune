import numpy as np
import os
import sys
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.audio.preprocess import load_all_datasets

# Load Config
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

def save_features():
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed'])
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    X, y = load_all_datasets()
    
    np.save(os.path.join(PROCESSED_DIR, 'X.npy'), X)
    np.save(os.path.join(PROCESSED_DIR, 'y.npy'), y)
    print(f"Saved features to {PROCESSED_DIR}")

if __name__ == "__main__":
    save_features()