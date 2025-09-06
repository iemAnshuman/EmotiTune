import numpy as np
import os

PROCESSED_DATA_DIR = 'data/processed/'

def load_features():
    X = np.load(os.path.join(PROCESSED_DATA_DIR, 'X.npy'))
    y = np.load(os.path.join(PROCESSED_DATA_DIR, 'y.npy'))
    
    print(f"Loaded features: X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y

if __name__ == "__main__":
    load_features()
