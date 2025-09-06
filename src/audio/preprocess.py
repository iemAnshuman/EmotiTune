import os
import librosa
import numpy as np

from librosa.feature import chroma_stft, spectral_contrast, tonnetz, mfcc
from librosa.effects import harmonic


# Define paths to your datasets
RAVDESS_SPEECH_PATH = 'data/ravdess/speech/'
RAVDESS_SONG_PATH = 'data/ravdess/song/'
CREMA_PATH = 'data/crema-d/'
DEAM_PATH = 'data/deam/audio/'

# Define emotion mappings
ravdess_emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

crema_emotions = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fearful',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}

# Utility functions
def extract_mfcc(file_path, n_mfcc=40, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_audio_features(file_path, n_mfcc=40, max_pad_len=174):
    try:
        y, sr = librosa.load(file_path, res_type='kaiser_fast')

        # MFCC
        mfcc_feat = mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Chroma
        chroma_feat = chroma_stft(y=y, sr=sr)
        # Spectral Contrast
        contrast_feat = spectral_contrast(y=y, sr=sr)
        # Tonnetz
        tonnetz_feat = tonnetz(y=harmonic(y), sr=sr)

        # Pad all to same time axis (second dimension)
        def pad_or_trim(feature):
            if feature.shape[1] < max_pad_len:
                pad_width = max_pad_len - feature.shape[1]
                return np.pad(feature, ((0,0),(0,pad_width)), mode='constant')
            else:
                return feature[:, :max_pad_len]

        mfcc_feat = pad_or_trim(mfcc_feat)
        chroma_feat = pad_or_trim(chroma_feat)
        contrast_feat = pad_or_trim(contrast_feat)
        tonnetz_feat = pad_or_trim(tonnetz_feat)

        # Stack all features vertically
        full_feature = np.vstack([mfcc_feat, chroma_feat, contrast_feat, tonnetz_feat])

        return full_feature
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def parse_ravdess_filename(filename):
    parts = filename.split('-')
    emotion_code = parts[2]
    return ravdess_emotions.get(emotion_code)

def parse_crema_filename(filename):
    parts = filename.split('_')
    emotion_code = parts[2]
    return crema_emotions.get(emotion_code)

def load_ravdess(path):
    X, y = [], []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.wav'):
                label = parse_ravdess_filename(file)
                if label:
                    mfcc = extract_audio_features(os.path.join(root, file))
                    if mfcc is not None:
                        X.append(mfcc)
                        y.append(label)
    return X, y

def load_crema():
    X, y = [], []
    for file in os.listdir(CREMA_PATH):
        if file.endswith('.wav'):
            label = parse_crema_filename(file)
            if label:
                mfcc = extract_audio_features(os.path.join(CREMA_PATH, file))
                if mfcc is not None:
                    X.append(mfcc)
                    y.append(label)
    return X, y

def load_deam():
    X, y = [], []
    for file in os.listdir(DEAM_PATH):
        if file.endswith('.mp3'):
            mfcc = extract_audio_features(os.path.join(DEAM_PATH, file))
            if mfcc is not None:
                X.append(mfcc)
                y.append('unknown')  # Placeholder
    return X, y

def load_all_datasets():
    print("Loading RAVDESS Speech...")
    X_ravdess_speech, y_ravdess_speech = load_ravdess(RAVDESS_SPEECH_PATH)
    
    print("Loading RAVDESS Song...")
    X_ravdess_song, y_ravdess_song = load_ravdess(RAVDESS_SONG_PATH)
    
    print("Loading CREMA-D...")
    X_crema, y_crema = load_crema()
    
    print("Loading DEAM...")
    X_deam, y_deam = load_deam()

    # Combine all
    X = X_ravdess_speech + X_ravdess_song + X_crema + X_deam
    y = y_ravdess_speech + y_ravdess_song + y_crema + y_deam

    print(f"Total samples loaded: {len(X)}")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = load_all_datasets()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels example: {y[:10]}")
