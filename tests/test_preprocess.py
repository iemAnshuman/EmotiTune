import os
import sys
import numpy as np
import pytest
import soundfile as sf

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.audio.preprocess import extract_audio_features, CONFIG

@pytest.fixture
def dummy_audio_file(tmp_path):
    """Creates a temporary dummy .wav file for testing"""
    file_path = tmp_path / "silence.wav"
    sr = 22050
    duration_sec = 3.0
    # Create 3 seconds of silence (zeros)
    y = np.zeros(int(sr * duration_sec))
    sf.write(file_path, y, sr)
    return str(file_path)

def test_extract_audio_features_shape(dummy_audio_file):
    """Ensure the extractor returns the correct shape defined in config"""
    features = extract_audio_features(dummy_audio_file)
    
    assert features is not None, "Feature extraction failed on dummy file"
    
    # Check if the output is 2D
    assert len(features.shape) == 2
    
    # Check if the time axis (column count) matches our config padding
    expected_time_frames = CONFIG['audio']['max_pad_len']
    assert features.shape[1] == expected_time_frames, \
        f"Expected {expected_time_frames} time frames, got {features.shape[1]}"

def test_invalid_file_path():
    """Ensure it handles missing files gracefully (returns None, doesn't crash)"""
    features = extract_audio_features("non_existent_ghost_file.wav")
    assert features is None