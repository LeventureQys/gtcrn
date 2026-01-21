# -*- coding: utf-8 -*-
"""
Compare audio files statistically
"""
import numpy as np
import soundfile as sf

# Load all versions
noisy, _ = sf.read('test_wavs/noisy_16k/00001_1_fan_noise_level1_snr-5dB_noisy.wav')
c_complete, _ = sf.read('test_wavs/output_c/complete_test.wav')
c_stream, _ = sf.read('test_wavs/output_c/stream_fixed_test3.wav')
py_stream, _ = sf.read('test_wavs/output_c/py_stream_test.wav')

print("=" * 60)
print("Audio Comparison Summary")
print("=" * 60)

min_len = min(len(noisy), len(c_complete), len(c_stream), len(py_stream))
noisy = noisy[:min_len]
c_complete = c_complete[:min_len]
c_stream = c_stream[:min_len]
py_stream = py_stream[:min_len]

print(f"\n{'Version':<20} {'RMS':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
print("-" * 60)
print(f"{'Noisy (input)':<20} {np.sqrt((noisy**2).mean()):>10.4f} {noisy.min():>10.4f} {noisy.max():>10.4f} {noisy.std():>10.4f}")
print(f"{'C Complete':<20} {np.sqrt((c_complete**2).mean()):>10.4f} {c_complete.min():>10.4f} {c_complete.max():>10.4f} {c_complete.std():>10.4f}")
print(f"{'C Stream':<20} {np.sqrt((c_stream**2).mean()):>10.4f} {c_stream.min():>10.4f} {c_stream.max():>10.4f} {c_stream.std():>10.4f}")
print(f"{'Python Stream':<20} {np.sqrt((py_stream**2).mean()):>10.4f} {py_stream.min():>10.4f} {py_stream.max():>10.4f} {py_stream.std():>10.4f}")

print("\nCorrelation Matrix:")
print("-" * 60)
print(f"{'':20} {'C Complete':>12} {'C Stream':>12} {'Py Stream':>12}")
print(f"{'C Complete':<20} {1.0:>12.4f} {np.corrcoef(c_complete, c_stream)[0,1]:>12.4f} {np.corrcoef(c_complete, py_stream)[0,1]:>12.4f}")
print(f"{'C Stream':<20} {np.corrcoef(c_stream, c_complete)[0,1]:>12.4f} {1.0:>12.4f} {np.corrcoef(c_stream, py_stream)[0,1]:>12.4f}")
print(f"{'Py Stream':<20} {np.corrcoef(py_stream, c_complete)[0,1]:>12.4f} {np.corrcoef(py_stream, c_stream)[0,1]:>12.4f} {1.0:>12.4f}")

print("\nKey Observations:")
print("-" * 60)
corr_c_py = np.corrcoef(c_complete, py_stream)[0,1]
corr_c_stream_py = np.corrcoef(c_stream, py_stream)[0,1]

if corr_c_py > 0.99:
    print("[OK] C complete matches Python stream (they should be same)")
else:
    print(f"[INFO] C complete vs Python stream: {corr_c_py:.4f}")

if corr_c_stream_py > 0.99:
    print("[OK] C stream matches Python stream")
else:
    print(f"[ISSUE] C stream vs Python stream: {corr_c_stream_py:.4f} - significant mismatch!")

# Check if C stream is just scaled version
# Try to find optimal scale
scale_factor = np.dot(py_stream, c_stream) / np.dot(c_stream, c_stream)
c_stream_scaled = c_stream * scale_factor
corr_scaled = np.corrcoef(c_stream_scaled, py_stream)[0,1]
error_scaled = np.abs(c_stream_scaled - py_stream).max()

print(f"\nScale analysis:")
print(f"  Optimal scale factor for C stream: {scale_factor:.4f}")
print(f"  Correlation after scaling: {corr_scaled:.4f}")
print(f"  Max error after scaling: {error_scaled:.4f}")
