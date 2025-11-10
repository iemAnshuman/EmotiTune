# Emotitune: Speech Emotion Recognition with CNNs

A deep learning pipeline that detects human emotions from audio recordings. It utilizes `librosa` for extracting rich acoustic features (MFCC, Chroma, Spectral Contrast, Tonnetz) and a Convolutional Neural Network (CNN) built in PyTorch for classification.

## 🚀 Features
* **Robust Preprocessing:** Standardized feature extraction pipeline handling varying audio lengths.
* **CNN Architecture:** Custom 2D CNN optimized for stacked audio feature maps.
* **Professional Workflow:** Includes logging, unit tests, and YAML-based configuration.
* **Inference Ready:** Simple command-line script to predict emotions on new audio files.

## 🛠️ Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Configure paths in `config.yaml` if using custom datasets.

## 🏃‍♂️ Usage

### 1. Data Processing
Extract features from raw audio datasets (RAVDESS, CREMA-D, etc.):
```bash
python src/audio/save_features.py