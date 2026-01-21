#!/usr/bin/env python3
"""
Verify: What should C's time-domain output be given its spectrum output?
This will tell us if the issue is in the C NN or in the ISTFT.
"""

import os
import sys
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

# From test_stream_debug output:
# Frame 100, C output spectrum:
#   Real sum:  0.3626 (Python: -0.3100)
#   Imag sum:  -10.0191 (Python: -5.5179)
#   Mag sum:   69.5701 (Python: 73.3360)

# Load C streaming output
c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
c_stream, _ = sf.read(c_stream_path)

n_fft = 512
hop = 256
win = 512

# sqrt-Hann window
window = np.zeros(win, dtype=np.float32)
for i in range(win):
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / win))
    window[i] = np.sqrt(hann)

# Check if the C time-domain output is consistent with its spectrum
# Frame 100's output samples (from test output)
c_frame_100 = c_stream[100 * hop:101 * hop]

# What is the RMS?
c_rms = np.sqrt(np.mean(c_frame_100**2))
print(f"C frame 100 time-domain RMS: {c_rms:.6f}")

# For a perfect ISTFT with sqrt-Hann window:
# The time-domain energy should be related to spectrum energy
# E_time = sum(|istft_frame|^2) ≈ (1/N) * sum(|X|^2)  for each frame
# But with OLA, we accumulate two half-frames

# Let's compute expected RMS from spectrum
# If mag_sum = 69.57, and there are 257 bins:
# Average magnitude ≈ 69.57 / 257 ≈ 0.27
# This doesn't directly give us time-domain RMS, but we can compare

# Better approach: Load Python's output and see what the spectrum/time ratio is
print("\n=== Checking Python's spectrum to time-domain ratio ===")

# From save_py_spectrums.py output:
# Frame 100: Out Real Sum: -0.3100, Out Imag Sum: -5.5179, Out Mag Sum: 73.3360

# Load Python simulation output
py_sim_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_sim_c.wav")
py_sim, _ = sf.read(py_sim_path)

py_frame_100 = py_sim[100 * hop:101 * hop]
py_rms = np.sqrt(np.mean(py_frame_100**2))
print(f"Python frame 100 time-domain RMS: {py_rms:.6f}")

# Ratio of RMS
time_ratio = c_rms / py_rms
print(f"Time-domain ratio (C/Py): {time_ratio:.4f}")

# Ratio of spectrum magnitude sums
spec_ratio = 69.5701 / 73.3360
print(f"Spectrum magnitude ratio (C/Py): {spec_ratio:.4f}")

print("\n=== Analysis ===")
if abs(time_ratio - spec_ratio) < 0.05:
    print("Time ratio ≈ Spectrum ratio")
    print("=> C ISTFT is correct, issue is in C neural network")
else:
    print(f"Time ratio ({time_ratio:.4f}) ≠ Spectrum ratio ({spec_ratio:.4f})")
    print("=> C ISTFT might have issues, or the spectrum comparison is misaligned")

# Let's also check if there's a frame alignment issue
print("\n=== Checking frame alignment ===")
print("With 1-frame offset:")
c_frame_101 = c_stream[101 * hop:102 * hop]
print(f"C frame 101 RMS: {np.sqrt(np.mean(c_frame_101**2)):.6f}")
print(f"Ratio to Py frame 100: {np.sqrt(np.mean(c_frame_101**2)) / py_rms:.4f}")

# Correlation between C frame 101 and Py frame 100
corr = np.corrcoef(c_frame_101, py_frame_100)[0, 1]
print(f"Correlation (C frame 101 vs Py frame 100): {corr:.4f}")

# Let's also compare shapes - maybe the issue is the C NN is outputting differently scaled values
print("\n=== Comparing the SHAPES (normalized) ===")
c_norm = c_frame_100 / np.max(np.abs(c_frame_100))
py_norm = py_frame_100 / np.max(np.abs(py_frame_100))
corr_norm = np.corrcoef(c_norm, py_norm)[0, 1]
print(f"Correlation of normalized shapes: {corr_norm:.4f}")

# If correlation is high, shape is same, just different scaling
# If correlation is low, shape is different - suggesting different NN outputs
