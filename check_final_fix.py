# -*- coding: utf-8 -*-
"""
Compare C stream output (after all fixes) with Python stream
"""
import numpy as np
import soundfile as sf

# Load outputs
c_stream, _ = sf.read('test_wavs/output_c/stream_fixed_final.wav')
py_stream, _ = sf.read('test_wavs/output_c/py_stream_test.wav')
c_complete, _ = sf.read('test_wavs/output_c/complete_test.wav')

min_len = min(len(c_stream), len(py_stream))
c_stream = c_stream[:min_len]
py_stream = py_stream[:min_len]
c_complete = c_complete[:min_len]

diff = np.abs(c_stream - py_stream)
max_error = diff.max()
mean_error = diff.mean()

c_energy = np.sqrt((c_stream ** 2).mean())
py_energy = np.sqrt((py_stream ** 2).mean())
energy_ratio = c_energy / py_energy

correlation = np.corrcoef(c_stream, py_stream)[0, 1]
corr_complete = np.corrcoef(c_complete, py_stream)[0, 1]

print("=" * 60)
print("Final Comparison (after all fixes)")
print("=" * 60)
print(f"\nC Stream vs Python Stream:")
print(f"  Max absolute error: {max_error:.6f}")
print(f"  Mean absolute error: {mean_error:.6f}")
print(f"  Energy ratio (C/Py): {energy_ratio:.4f}")
print(f"  Correlation: {correlation:.6f}")

print(f"\nC Complete vs Python Stream:")
print(f"  Correlation: {corr_complete:.6f}")

print("\n" + "=" * 60)
if correlation > 0.99:
    print("✅ SUCCESS! C stream matches Python stream!")
    print("   All fixes applied successfully.")
elif correlation > 0.95:
    print("✅ GOOD! C stream is very close to Python stream.")
    print(f"   Correlation: {correlation:.4f}")
else:
    print(f"❌ Still has issues. Correlation: {correlation:.4f}")

print("=" * 60)
