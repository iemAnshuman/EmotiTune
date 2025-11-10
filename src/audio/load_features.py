import numpy as np
import os
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
with open(os.path.join(PROJECT_ROOT, 'config.yaml'), 'r') as f:
    CONFIG = yaml.safe_load(f)

def load_features():
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, CONFIG['data']['processed'])
    X = np.load(os.path.join(PROCESSED_DIR, 'X.npy'))
    y = np.load(os.path.join(PROCESSED_DIR, 'y.npy'))
    
    print(f"Loaded features: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y

if __name__ == "__main__":
    load_features()