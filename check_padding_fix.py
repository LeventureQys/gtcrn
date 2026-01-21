# -*- coding: utf-8 -*-
"""
Compare C stream output (after padding fix) with Python stream
"""
import numpy as np
import soundfile as sf

# Load outputs
c_stream, _ = sf.read('test_wavs/output_c/stream_fixed_test3.wav')
py_stream, _ = sf.read('test_wavs/output_c/py_stream_test.wav')

min_len = min(len(c_stream), len(py_stream))
c_stream = c_stream[:min_len]
py_stream = py_stream[:min_len]

diff = np.abs(c_stream - py_stream)
max_error = diff.max()
mean_error = diff.mean()

c_energy = np.sqrt((c_stream ** 2).mean())
py_energy = np.sqrt((py_stream ** 2).mean())
energy_ratio = c_energy / py_energy

correlation = np.corrcoef(c_stream, py_stream)[0, 1]

print("=" * 50)
print("C Stream vs Python Stream (after padding fix)")
print("=" * 50)
print(f"\nMax absolute error: {max_error:.6f}")
print(f"Mean absolute error: {mean_error:.6f}")
print(f"Energy ratio (C/Py): {energy_ratio:.4f}")
print(f"Correlation: {correlation:.6f}")

if correlation > 0.99:
    print("\n[OK] C and Python stream outputs are consistent!")
elif correlation > 0.9:
    print(f"\n[WARN] Some difference, correlation = {correlation:.4f}")
else:
    print(f"\n[ISSUE] Significant difference, correlation = {correlation:.4f}")
