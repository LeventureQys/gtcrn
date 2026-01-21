"""
Compare C streaming output with Python proper streaming output.
"""
import os
import numpy as np
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))

# Load outputs
c_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_c_stream.wav')
py_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_py_stream_proper.wav')

c_out, sr = sf.read(c_output_path)
py_out, _ = sf.read(py_output_path)

print(f"C output: {len(c_out)} samples")
print(f"Python output: {len(py_out)} samples")

min_len = min(len(c_out), len(py_out))
c_out = c_out[:min_len]
py_out = py_out[:min_len]

# Compare
correlation = np.corrcoef(c_out, py_out)[0, 1]
mse = np.mean((c_out - py_out)**2)
max_diff = np.max(np.abs(c_out - py_out))
energy_ratio = np.sum(c_out**2) / (np.sum(py_out**2) + 1e-10)

print(f"\n=== C vs Python Proper Comparison ===")
print(f"Correlation: {correlation:.6f}")
print(f"MSE: {mse:.8f}")
print(f"Max difference: {max_diff:.6f}")
print(f"Energy ratio (C/Py): {energy_ratio:.6f}")

print(f"\nFirst 10 samples (C):  {c_out[:10]}")
print(f"First 10 samples (Py): {py_out[:10]}")

# Find where they start to differ
for i in range(min(100, min_len)):
    if abs(c_out[i] - py_out[i]) > 0.0001:
        print(f"\nFirst significant difference at sample {i}")
        print(f"  C[{i}] = {c_out[i]:.6f}")
        print(f"  Py[{i}] = {py_out[i]:.6f}")
        break

# Check mid section
mid = min_len // 2
print(f"\nMiddle samples (C):  {c_out[mid:mid+10]}")
print(f"Middle samples (Py): {py_out[mid:mid+10]}")

# Statistics
print(f"\nC mean: {np.mean(c_out):.6f}, std: {np.std(c_out):.6f}")
print(f"Py mean: {np.mean(py_out):.6f}, std: {np.std(py_out):.6f}")
