import numpy as np
import os

# Import the preprocessing functions
from src.audio.preprocess import load_all_datasets

# Define where to save the processed features
PROCESSED_DATA_DIR = 'data/processed/'

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def save_features():
    X, y = load_all_datasets()
    
    # Save the feature arrays
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y.npy'), y)

    print(f"Saved features to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    save_features()
