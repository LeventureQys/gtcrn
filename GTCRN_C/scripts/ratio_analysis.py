#!/usr/bin/env python3
"""
Detailed analysis of the ratio between C and Python simulation.
"""

import os
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

# Load outputs
c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
py_sim_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_sim_c.wav")

c_stream, _ = sf.read(c_stream_path)
py_sim, _ = sf.read(py_sim_path)

hop = 256

print("=== Frame-by-frame energy ratio analysis ===")
print("Frame | C RMS      | Py RMS     | Ratio")
print("-" * 50)

for frame in range(100, 110):
    start = frame * hop
    end = start + hop

    c_seg = c_stream[start:end]
    py_seg = py_sim[start:end]

    c_rms = np.sqrt(np.mean(c_seg**2))
    py_rms = np.sqrt(np.mean(py_seg**2))

    if py_rms > 1e-8:
        ratio = c_rms / py_rms
        print(f"{frame:5d} | {c_rms:10.6f} | {py_rms:10.6f} | {ratio:.4f}")

print("\n=== Sample-by-sample ratio (where |Py| > 0.01) ===")
start = 100 * hop
end = 200 * hop
c_seg = c_stream[start:end]
py_seg = py_sim[start:end]

mask = np.abs(py_seg) > 0.01
ratios = c_seg[mask] / py_seg[mask]

print(f"Number of samples: {np.sum(mask)}")
print(f"Mean ratio:   {np.mean(ratios):.4f}")
print(f"Median ratio: {np.median(ratios):.4f}")
print(f"Std ratio:    {np.std(ratios):.4f}")
print(f"Min ratio:    {np.min(ratios):.4f}")
print(f"Max ratio:    {np.max(ratios):.4f}")

# Check if the ratio is constant (simple scaling)
cv = np.std(ratios) / np.abs(np.mean(ratios))
print(f"Coefficient of variation: {cv:.4f}")

if cv < 0.1:
    print("-> Simple scaling issue!")
else:
    print("-> Not a simple scaling issue")

# Check if there's a pattern related to sign
print("\n=== Sign analysis ===")
positive_mask = py_seg > 0.01
negative_mask = py_seg < -0.01

if np.any(positive_mask):
    pos_ratios = c_seg[positive_mask] / py_seg[positive_mask]
    print(f"Positive samples mean ratio: {np.mean(pos_ratios):.4f}")

if np.any(negative_mask):
    neg_ratios = c_seg[negative_mask] / py_seg[negative_mask]
    print(f"Negative samples mean ratio: {np.mean(neg_ratios):.4f}")

# Check if there's a position-dependent pattern
print("\n=== Position-dependent analysis (within hop) ===")
c_seg_reshaped = c_seg.reshape(-1, hop)
py_seg_reshaped = py_seg.reshape(-1, hop)

for pos in [0, 64, 128, 192, 255]:
    c_vals = c_seg_reshaped[:, pos]
    py_vals = py_seg_reshaped[:, pos]
    mask = np.abs(py_vals) > 0.01
    if np.sum(mask) > 10:
        ratios = c_vals[mask] / py_vals[mask]
        print(f"Position {pos:3d}: mean ratio = {np.mean(ratios):.4f}, std = {np.std(ratios):.4f}")
