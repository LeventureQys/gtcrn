# -*- coding: utf-8 -*-
"""
Compare streaming and complete inference outputs
"""
import numpy as np
import soundfile as sf

# Load outputs
stream_audio, sr1 = sf.read('test_wavs/output_c/stream_fixed_test.wav')
complete_audio, sr2 = sf.read('test_wavs/output_c/complete_test.wav')

print("=" * 50)
print("Stream vs Complete Inference Comparison")
print("=" * 50)

print(f"\nSample rate: stream={sr1}Hz, complete={sr2}Hz")
print(f"Length: stream={len(stream_audio)}, complete={len(complete_audio)}")

# Align lengths
min_len = min(len(stream_audio), len(complete_audio))
stream_audio = stream_audio[:min_len]
complete_audio = complete_audio[:min_len]

# Calculate error
diff = np.abs(stream_audio - complete_audio)
max_error = diff.max()
mean_error = diff.mean()
rms_error = np.sqrt((diff ** 2).mean())

print(f"\nError Analysis:")
print(f"  Max absolute error: {max_error:.6f}")
print(f"  Mean absolute error: {mean_error:.6f}")
print(f"  RMS error: {rms_error:.6f}")

# Signal statistics
print(f"\nSignal Statistics:")
print(f"  Stream output - min: {stream_audio.min():.4f}, max: {stream_audio.max():.4f}, std: {stream_audio.std():.4f}")
print(f"  Complete output - min: {complete_audio.min():.4f}, max: {complete_audio.max():.4f}, std: {complete_audio.std():.4f}")

# Check if streaming output looks reasonable
stream_energy = np.sqrt((stream_audio ** 2).mean())
complete_energy = np.sqrt((complete_audio ** 2).mean())
energy_ratio = stream_energy / complete_energy if complete_energy > 0 else 0

print(f"\nEnergy Comparison:")
print(f"  Stream RMS energy: {stream_energy:.6f}")
print(f"  Complete RMS energy: {complete_energy:.6f}")
print(f"  Energy ratio: {energy_ratio:.4f}")

# Check correlation
correlation = np.corrcoef(stream_audio, complete_audio)[0, 1]
print(f"\nCorrelation: {correlation:.6f}")

# Overall assessment
print("\n" + "=" * 50)
if max_error < 0.01 and energy_ratio > 0.8 and energy_ratio < 1.2:
    print("Assessment: [OK] Stream and complete outputs are highly consistent")
elif max_error < 0.1 and correlation > 0.9:
    print("Assessment: [WARN] Some difference, but good correlation")
else:
    print("Assessment: [ISSUE] Significant difference between stream and complete outputs")
    print(f"  - Energy ratio {energy_ratio:.2f} suggests stream output might be scaled differently")
    print(f"  - Correlation {correlation:.4f}")
print("=" * 50)

# Find where largest errors occur
top_error_indices = np.argsort(diff)[-10:]
print("\nTop 10 error locations:")
for idx in top_error_indices:
    print(f"  Sample {idx}: stream={stream_audio[idx]:.4f}, complete={complete_audio[idx]:.4f}, diff={diff[idx]:.4f}")
