"""
Quick check: apply 1-frame offset to C output and compare.
"""
import os
import numpy as np
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))

# Load outputs
c_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_c_stream.wav')
py_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_py_stream_proper.wav')  # Python with center=True

c_out, sr = sf.read(c_output_path)
py_out, _ = sf.read(py_output_path)

print(f"C output: {len(c_out)} samples")
print(f"Python output: {len(py_out)} samples")

# Apply +256 offset (skip first 256 samples of C)
offset = 256
c_shifted = c_out[offset:]
py_aligned = py_out[:len(c_shifted)]

min_len = min(len(c_shifted), len(py_aligned))
c_s = c_shifted[:min_len]
p_a = py_aligned[:min_len]

correlation = np.corrcoef(c_s, p_a)[0, 1]
mse = np.mean((c_s - p_a)**2)
max_diff = np.max(np.abs(c_s - p_a))

print(f"\n=== After shifting C by +256 samples ===")
print(f"Correlation: {correlation:.6f}")
print(f"MSE: {mse:.8f}")
print(f"Max difference: {max_diff:.6f}")

print(f"\nFirst 10 samples (C shifted): {c_s[:10]}")
print(f"First 10 samples (Py):        {p_a[:10]}")

# If they match, the C output needs to have its first 256 samples removed
# or we need to prepend 256 samples of initial output

# Let's also check if they match sample-by-sample
diffs = np.abs(c_s - p_a)
print(f"\nMean abs diff: {np.mean(diffs):.8f}")
print(f"Samples with diff > 0.001: {np.sum(diffs > 0.001)}")
print(f"Samples with diff > 0.0001: {np.sum(diffs > 0.0001)}")
