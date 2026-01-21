"""
Detailed comparison after offset.
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

# Apply +256 offset
offset = 256
c_shifted = c_out[offset:]

# Check different alignment positions
print("Checking different alignments:")
for start_offset in range(0, 300, 50):
    c_s = c_shifted[start_offset:]
    p_a = py_out[start_offset:start_offset + len(c_s)]

    min_len = min(len(c_s), len(p_a))
    if min_len < 1000:
        continue

    c_s = c_s[:min_len]
    p_a = p_a[:min_len]

    correlation = np.corrcoef(c_s, p_a)[0, 1]
    max_diff = np.max(np.abs(c_s - p_a))
    print(f"  C[{offset+start_offset}:] vs Py[{start_offset}:]: corr={correlation:.6f}, max_diff={max_diff:.6f}")

# Now check if it's a boundary issue - look at middle of file
print("\nMiddle of file comparison:")
mid = len(c_out) // 2
c_mid = c_out[mid:mid+20]
py_mid = py_out[mid-256:mid-256+20]  # Offset adjusted
print(f"C[{mid}:{mid+20}]:      {c_mid}")
print(f"Py[{mid-256}:{mid-256+20}]: {py_mid}")
print(f"Max diff: {np.max(np.abs(c_mid - py_mid)):.8f}")

# Precise sample-by-sample match
print("\nSample-by-sample match (middle region):")
start = len(c_out) // 2
for i in range(5):
    c_idx = start + i
    py_idx = start - 256 + i
    print(f"  C[{c_idx}]={c_out[c_idx]:.8f}, Py[{py_idx}]={py_out[py_idx]:.8f}, diff={abs(c_out[c_idx]-py_out[py_idx]):.8f}")
