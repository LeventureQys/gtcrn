#!/usr/bin/env python3
"""
Verify: If C neural network output mag sum is 69.57 and Python is 73.34 (ratio 0.95),
but time-domain ratio is ~0.59, where is the energy loss?

Let's compute what the expected time-domain RMS should be from the spectrum.
"""

import os
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

# Load C streaming output
c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
c_stream, _ = sf.read(c_stream_path)

# Load Python simulation output (using same C algorithm in Python)
py_sim_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_sim_c.wav")
py_sim, _ = sf.read(py_sim_path)

hop = 256

print("=== Energy comparison ===")
print("Frame | C Stream RMS | Py Sim RMS   | Ratio | Expected from debug")
print("-" * 70)

# From test_stream_debug output, the C NN mag sum was 69.57 vs Python 73.34
# Ratio = 0.9486
# But the time-domain ratio is ~0.59

for frame in range(100, 105):
    # Account for 1-frame offset: C frame 100 = Python frame 101
    c_start = (frame + 1) * hop  # C has 1-frame offset
    c_seg = c_stream[c_start:c_start + hop]

    py_start = frame * hop
    py_seg = py_sim[py_start:py_start + hop]

    c_rms = np.sqrt(np.mean(c_seg**2))
    py_rms = np.sqrt(np.mean(py_seg**2))
    ratio = c_rms / py_rms if py_rms > 1e-8 else 0

    print(f"{frame:5d} | {c_rms:12.6f} | {py_rms:12.6f} | {ratio:.4f} |")

print("\n=== Hypothesis: The problem might be frame index mismatch ===")
print("\nLet me also check WITHOUT the offset:")
print("-" * 70)

for frame in range(100, 105):
    c_start = frame * hop  # No offset
    c_seg = c_stream[c_start:c_start + hop]

    py_start = frame * hop
    py_seg = py_sim[py_start:py_start + hop]

    c_rms = np.sqrt(np.mean(c_seg**2))
    py_rms = np.sqrt(np.mean(py_seg**2))
    ratio = c_rms / py_rms if py_rms > 1e-8 else 0

    print(f"{frame:5d} | {c_rms:12.6f} | {py_rms:12.6f} | {ratio:.4f} |")

print("\n=== Now check spectrum vs time domain scaling ===")
print("\nFrom Parseval's theorem, sum(|X|^2) = N * sum(|x|^2)")
print("So if mag_sum ≈ 69.57 for 257 bins, we can estimate time-domain energy")

# Assuming the output spectrum has approximately these values
c_mag_sum = 69.57
py_mag_sum = 73.34

# For a frame of 512 samples with sqrt-Hann window:
# The energy ratio should be similar
expected_ratio = c_mag_sum / py_mag_sum
print(f"\nExpected spectrum ratio: {expected_ratio:.4f}")
print("But observed time-domain ratio is ~0.59")
print("\nThis suggests the issue is NOT in the neural network")
print("but in how the ISTFT output is being combined/written")
