# Emotitune: Speech Emotion Recognition with CNNs

A deep learning pipeline that detects human emotions from audio recordings. It utilizes `librosa` for extracting rich acoustic features (MFCC, Chroma, Spectral Contrast, Tonnetz) and a Convolutional Neural Network (CNN) built in PyTorch for classification.

## 🚀 Features
* **Robust Preprocessing:** Standardized feature extraction pipeline handling varying audio lengths.
* **CNN Architecture:** Custom 2D CNN optimized for stacked audio feature maps.
* **Professional Workflow:** Includes logging, unit tests, and YAML-based configuration.
* **Inference Ready:** Simple command-line script to predict emotions on new audio files.
* **Automated Pipeline:** Complete end-to-end execution with report generation via `run.py`.

## 🛠️ Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Configure paths in `config.yaml` if using custom datasets.

## 🏃‍♂️ Usage

### Automated Pipeline (Recommended)
Run the entire process (processing, training, reporting) with a single command:
```bash
python run.py