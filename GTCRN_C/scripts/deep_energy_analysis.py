#!/usr/bin/env python3
"""
Deep analysis of the energy ratio issue.
"""

import numpy as np
import os
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

# Also compare C complete output
c_complete_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c.wav")
c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
py_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_stream_nocenter.wav")

c_complete, _ = sf.read(c_complete_path)
c_stream, _ = sf.read(c_stream_path)
py_stream, _ = sf.read(py_stream_path)

hop = 256
start = 100 * hop
end = 200 * hop

print("=== RMS comparison ===")
print(f"C complete RMS:   {np.sqrt(np.mean(c_complete[start:end]**2)):.6f}")
print(f"C streaming RMS:  {np.sqrt(np.mean(c_stream[start:end]**2)):.6f}")
print(f"Python stream RMS: {np.sqrt(np.mean(py_stream[start-hop:end-hop]**2)):.6f}")

print(f"\nC stream / C complete ratio: {np.sqrt(np.mean(c_stream[start:end]**2)) / np.sqrt(np.mean(c_complete[start:end]**2)):.4f}")
print(f"C stream / Py stream ratio:  {np.sqrt(np.mean(c_stream[start:end]**2)) / np.sqrt(np.mean(py_stream[start-hop:end-hop]**2)):.4f}")
print(f"C complete / Py stream ratio: {np.sqrt(np.mean(c_complete[start:end]**2)) / np.sqrt(np.mean(py_stream[start-hop:end-hop]**2)):.4f}")

print("\n=== Correlation comparison ===")
min_len = min(len(c_complete), len(c_stream), len(py_stream))

# C complete vs C stream (same offset)
corr_c_c = np.corrcoef(c_complete[start:end], c_stream[start:end])[0, 1]
print(f"C complete vs C stream (no shift): {corr_c_c:.4f}")

# C complete vs C stream (with offset)
corr_c_c_offset = np.corrcoef(c_complete[start:end], c_stream[start+hop:end+hop])[0, 1]
print(f"C complete vs C stream (C shifted +1 frame): {corr_c_c_offset:.4f}")

# C complete vs Py stream
corr_c_py = np.corrcoef(c_complete[start:end], py_stream[start-hop:end-hop])[0, 1]
print(f"C complete vs Py stream (Py shifted -1 frame): {corr_c_py:.4f}")

print("\n=== Checking if C stream is just scaled version ===")
# If C stream is a scaled version of Python, we should see constant ratio
c_seg = c_stream[start:end]
py_seg = py_stream[start-hop:end-hop]

mask = np.abs(py_seg) > 0.01
ratios = c_seg[mask] / py_seg[mask]

print(f"Sample ratio statistics (C/Py where |Py| > 0.01):")
print(f"  Count: {np.sum(mask)}")
print(f"  Mean:  {np.mean(ratios):.4f}")
print(f"  Std:   {np.std(ratios):.4f}")
print(f"  Min:   {np.min(ratios):.4f}")
print(f"  Max:   {np.max(ratios):.4f}")

# If std is small relative to mean, it's a simple scaling
cv = np.std(ratios) / np.abs(np.mean(ratios))
print(f"  Coefficient of variation: {cv:.4f}")

if cv < 0.1:
    print("  -> Likely a simple scaling issue!")
    print(f"  -> Scale factor: {np.mean(ratios):.4f}")
else:
    print("  -> Not a simple scaling issue, more complex difference")
