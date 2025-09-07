import numpy as np
import os
import sys

# Define the project's root directory
# This makes the script runnable from any location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Import the preprocessing functions
from src.audio.preprocess import load_all_datasets

# Define where to save the processed features
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

def save_features():
    # Create the directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    X, y = load_all_datasets()
    
    # Save the feature arrays
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X.npy'), X)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y.npy'), y)

    print(f"Saved features to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    save_features()