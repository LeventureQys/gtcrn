# -*- coding: utf-8 -*-
"""
Compare C and Python complete inference outputs
"""
import numpy as np
import soundfile as sf

# Load outputs
c_complete, sr1 = sf.read('test_wavs/output_c/complete_test.wav')
py_complete, sr2 = sf.read('test_wavs/output_c/py_complete_test.wav')

print("=" * 50)
print("C vs Python Complete Inference Comparison")
print("=" * 50)

min_len = min(len(c_complete), len(py_complete))
c_complete = c_complete[:min_len]
py_complete = py_complete[:min_len]

diff = np.abs(c_complete - py_complete)
max_error = diff.max()
mean_error = diff.mean()

c_energy = np.sqrt((c_complete ** 2).mean())
py_energy = np.sqrt((py_complete ** 2).mean())
energy_ratio = c_energy / py_energy

correlation = np.corrcoef(c_complete, py_complete)[0, 1]

print(f"\nC Complete vs Python Complete:")
print(f"  Max absolute error: {max_error:.6f}")
print(f"  Mean absolute error: {mean_error:.6f}")
print(f"  Energy ratio (C/Py): {energy_ratio:.4f}")
print(f"  Correlation: {correlation:.6f}")

# Now compare C stream with Python stream
c_stream, _ = sf.read('test_wavs/output_c/stream_fixed_test.wav')
py_stream, _ = sf.read('test_wavs/output_c/py_stream_test.wav')

min_len2 = min(len(c_stream), len(py_stream))
c_stream = c_stream[:min_len2]
py_stream = py_stream[:min_len2]

diff2 = np.abs(c_stream - py_stream)
max_error2 = diff2.max()
mean_error2 = diff2.mean()

c_stream_energy = np.sqrt((c_stream ** 2).mean())
py_stream_energy = np.sqrt((py_stream ** 2).mean())
energy_ratio2 = c_stream_energy / py_stream_energy

correlation2 = np.corrcoef(c_stream, py_stream)[0, 1]

print(f"\nC Stream vs Python Stream:")
print(f"  Max absolute error: {max_error2:.6f}")
print(f"  Mean absolute error: {mean_error2:.6f}")
print(f"  Energy ratio (C/Py): {energy_ratio2:.4f}")
print(f"  Correlation: {correlation2:.6f}")

print("\n" + "=" * 50)
if correlation > 0.99:
    print("Complete inference: [OK] C and Python are consistent")
else:
    print("Complete inference: [ISSUE] C and Python differ")

if correlation2 > 0.99:
    print("Stream inference: [OK] C and Python are consistent")
else:
    print(f"Stream inference: [ISSUE] C stream has issues (corr={correlation2:.4f})")
print("=" * 50)
