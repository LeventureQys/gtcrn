#!/usr/bin/env python3
"""
Compare neural network spectrum outputs between C and Python.
This requires running the C code with debug output first.
For now, let's compare the final OLA outputs more carefully.
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
py_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_stream_nocenter.wav")
c_complete_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c.wav")

c_stream, _ = sf.read(c_stream_path)
py_sim, _ = sf.read(py_sim_path)
py_stream, _ = sf.read(py_stream_path)
c_complete, _ = sf.read(c_complete_path)

hop = 256
start = 100 * hop
end = 200 * hop

print("=== Comprehensive comparison ===\n")

print("1. C streaming vs Python simulation (same algorithm):")
print(f"   Correlation: {np.corrcoef(c_stream[start:end], py_sim[start:end])[0,1]:.6f}")
print(f"   C RMS: {np.sqrt(np.mean(c_stream[start:end]**2)):.6f}")
print(f"   Py RMS: {np.sqrt(np.mean(py_sim[start:end]**2)):.6f}")
print(f"   Ratio: {np.sqrt(np.mean(c_stream[start:end]**2))/np.sqrt(np.mean(py_sim[start:end]**2)):.4f}")

print("\n2. C streaming vs Python stream (with normalization):")
# Account for 1-frame offset
c_seg = c_stream[start+hop:end+hop]
py_seg = py_stream[start:end]
min_len = min(len(c_seg), len(py_seg))
c_seg = c_seg[:min_len]
py_seg = py_seg[:min_len]
print(f"   Correlation: {np.corrcoef(c_seg, py_seg)[0,1]:.6f}")
print(f"   C RMS: {np.sqrt(np.mean(c_seg**2)):.6f}")
print(f"   Py RMS: {np.sqrt(np.mean(py_seg**2)):.6f}")
print(f"   Ratio: {np.sqrt(np.mean(c_seg**2))/np.sqrt(np.mean(py_seg**2)):.4f}")

print("\n3. C complete vs Python stream (with 1-frame offset):")
c_seg = c_complete[start:end]
py_seg = py_stream[start-hop:end-hop] if start >= hop else py_stream[:end-hop]
min_len = min(len(c_seg), len(py_seg))
c_seg = c_seg[:min_len]
py_seg = py_seg[:min_len]
print(f"   Correlation: {np.corrcoef(c_seg, py_seg)[0,1]:.6f}")
print(f"   C RMS: {np.sqrt(np.mean(c_seg**2)):.6f}")
print(f"   Py RMS: {np.sqrt(np.mean(py_seg**2)):.6f}")
print(f"   Ratio: {np.sqrt(np.mean(c_seg**2))/np.sqrt(np.mean(py_seg**2)):.4f}")

print("\n4. C streaming vs C complete (with 1-frame offset):")
c_stream_seg = c_stream[start+hop:end+hop]
c_complete_seg = c_complete[start:end]
min_len = min(len(c_stream_seg), len(c_complete_seg))
print(f"   Correlation: {np.corrcoef(c_stream_seg[:min_len], c_complete_seg[:min_len])[0,1]:.6f}")
print(f"   C stream RMS: {np.sqrt(np.mean(c_stream_seg[:min_len]**2)):.6f}")
print(f"   C complete RMS: {np.sqrt(np.mean(c_complete_seg[:min_len]**2)):.6f}")
print(f"   Ratio: {np.sqrt(np.mean(c_stream_seg[:min_len]**2))/np.sqrt(np.mean(c_complete_seg[:min_len]**2)):.4f}")

print("\n5. Python simulation vs Python stream (same neural network):")
# Python sim uses C-style STFT (1-frame offset), Python stream uses PyTorch STFT
py_sim_seg = py_sim[start:end]
py_stream_seg = py_stream[start-hop:end-hop] if start >= hop else py_stream[:end-hop]
min_len = min(len(py_sim_seg), len(py_stream_seg))
print(f"   Correlation: {np.corrcoef(py_sim_seg[:min_len], py_stream_seg[:min_len])[0,1]:.6f}")
print(f"   Py sim RMS: {np.sqrt(np.mean(py_sim_seg[:min_len]**2)):.6f}")
print(f"   Py stream RMS: {np.sqrt(np.mean(py_stream_seg[:min_len]**2)):.6f}")
print(f"   Ratio: {np.sqrt(np.mean(py_sim_seg[:min_len]**2))/np.sqrt(np.mean(py_stream_seg[:min_len]**2)):.4f}")

print("\n=== Key observations ===")
print("If C streaming has lower energy than Python simulation,")
print("the bug is in the C neural network implementation,")
print("not in the STFT/ISTFT/OLA code.")
print("")
print("If Python simulation matches Python stream,")
print("then the C-style STFT/ISTFT is equivalent to Python's.")
