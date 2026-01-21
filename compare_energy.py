# -*- coding: utf-8 -*-
"""
Compare C streaming output with Python streaming output to analyze energy difference
"""
import numpy as np
import soundfile as sf
import os

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(gtcrn_dir, 'test_wavs/output_c/')

# List recent files
files = [
    ('c_stream_latest.wav', 'C Stream (stream weights)'),
    ('c_offline_correct.wav', 'C Offline (offline weights)'),
    ('py_stream_test.wav', 'Python Stream'),
]

print("=" * 60)
print("Audio Energy Comparison")
print("=" * 60)

# Load all files
data = {}
for fname, label in files:
    path = os.path.join(output_dir, fname)
    if os.path.exists(path):
        audio, sr = sf.read(path)
        data[label] = audio
        print(f"Loaded {label}: {len(audio)} samples")
    else:
        print(f"File not found: {path}")

if len(data) < 2:
    print("Not enough files to compare!")
    exit(1)

# Align lengths
min_len = min(len(v) for v in data.values())
for k in data:
    data[k] = data[k][:min_len]

print(f"\nAligned length: {min_len} samples")

# Stats
print(f"\n{'Version':<20} {'Energy':>12} {'RMS':>10} {'Max':>10}")
print("-" * 60)
for label, audio in data.items():
    energy = np.sum(audio ** 2)
    rms = np.sqrt(np.mean(audio ** 2))
    maxval = np.max(np.abs(audio))
    print(f"{label:<20} {energy:>12.4f} {rms:>10.6f} {maxval:>10.4f}")

# Energy ratios
print("\nEnergy Ratios:")
print("-" * 60)
labels = list(data.keys())
for i, l1 in enumerate(labels):
    for l2 in labels[i+1:]:
        e1 = np.sum(data[l1] ** 2)
        e2 = np.sum(data[l2] ** 2)
        ratio = e1 / e2 if e2 > 0 else float('inf')
        rms_ratio = np.sqrt(ratio)
        print(f"{l1} / {l2}: Energy={ratio:.4f}, RMS={rms_ratio:.4f}")

# Correlations
print("\nCorrelations:")
print("-" * 60)
for i, l1 in enumerate(labels):
    for l2 in labels[i+1:]:
        corr = np.corrcoef(data[l1], data[l2])[0, 1]
        print(f"{l1} vs {l2}: {corr:.6f}")

# Check with offset
print("\nCorrelations with 1-frame (256 sample) offset:")
print("-" * 60)
offset = 256
for i, l1 in enumerate(labels):
    for l2 in labels[i+1:]:
        a1 = data[l1][:-offset]
        a2 = data[l2][offset:]
        min_len2 = min(len(a1), len(a2))
        a1 = a1[:min_len2]
        a2 = a2[:min_len2]
        corr = np.corrcoef(a1, a2)[0, 1]
        e1 = np.sum(a1 ** 2)
        e2 = np.sum(a2 ** 2)
        print(f"{l1}[:-{offset}] vs {l2}[{offset}:]: corr={corr:.6f}, energy_ratio={e1/e2:.4f}")
