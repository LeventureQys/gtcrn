#!/usr/bin/env python3
"""
Create a test file to check C STFT/ISTFT roundtrip.
"""

import os
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

# Create a simple test signal (sine wave)
sr = 16000
duration = 0.5  # 500ms
freq = 440  # A4

t = np.arange(int(sr * duration)) / sr
test_signal = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)

# Save test signal
test_path = os.path.join(gtcrn_dir, "test_wavs/test_sine_440.wav")
sf.write(test_path, test_signal, sr)
print(f"Created test signal: {test_path}")
print(f"Length: {len(test_signal)} samples")
print(f"RMS: {np.sqrt(np.mean(test_signal**2)):.6f}")
