#!/usr/bin/env python3
"""
Trace overlap-add to understand the energy ratio issue.
"""

import numpy as np

win_length = 512
hop = 256

# sqrt-Hann window
window = np.sqrt(0.5 * (1 - np.cos(2 * np.pi * np.arange(win_length) / win_length)))

print("=== Window properties ===")
print(f"Window[0]: {window[0]:.6f}")
print(f"Window[255]: {window[255]:.6f}")
print(f"Window[256]: {window[256]:.6f}")
print(f"Window[511]: {window[511]:.6f}")

# Check: window[i]^2 + window[i+256]^2 for overlap positions
print("\n=== Overlap window sum ===")
for i in [0, 128, 255]:
    w_sum = window[i]**2 + window[i+256]**2
    print(f"Position {i}: w[{i}]^2 + w[{i+256}]^2 = {w_sum:.6f}")

# Simulate the streaming OLA
print("\n=== Streaming OLA simulation ===")
print("Using a constant signal of 1.0 to check normalization")

# Simulate 5 frames of constant input
n_frames = 5
signal = np.ones(n_frames * hop + win_length)

# Do STFT -> ISTFT simulation
frames_output = []
ola_buffer = np.zeros(hop)
first_frame = True

for frame_idx in range(n_frames):
    start = frame_idx * hop
    # Simulate ISTFT: window * signal (assume no frequency processing)
    istft_frame = window * signal[start:start+win_length]

    if first_frame:
        # Output first half directly
        output = istft_frame[:hop].copy()
        first_frame = False
    else:
        # Overlap-add
        output = ola_buffer + istft_frame[:hop]

    # Save second half for next frame
    ola_buffer = istft_frame[hop:].copy()

    frames_output.append(output)

    # Check output values
    print(f"\nFrame {frame_idx}:")
    print(f"  Output[0]: {output[0]:.6f}, Output[255]: {output[-1]:.6f}")
    print(f"  Mean output: {np.mean(output):.6f}")
    if frame_idx > 0:
        print(f"  Expected (if normalized): 1.0")
        print(f"  Ratio: {np.mean(output):.6f}")

# Full output
full_output = np.concatenate(frames_output)
print(f"\n=== Full output ===")
print(f"Length: {len(full_output)}")
print(f"Mean of samples 256-1024: {np.mean(full_output[hop:4*hop]):.6f}")

# Compare with proper normalization
print("\n=== With proper normalization ===")
# Properly normalized output
frames_output_norm = []
ola_buffer = np.zeros(hop)
window_sum_buffer = np.zeros(hop)  # Track window sum at each position
first_frame = True

for frame_idx in range(n_frames):
    start = frame_idx * hop
    istft_frame = window * signal[start:start+win_length]
    istft_window_sq = window ** 2

    if first_frame:
        output_unnorm = istft_frame[:hop].copy()
        window_sum = istft_window_sq[:hop].copy()
        first_frame = False
    else:
        output_unnorm = ola_buffer + istft_frame[:hop]
        window_sum = window_sum_buffer + istft_window_sq[:hop]

    # Normalize by window sum
    output = output_unnorm / np.maximum(window_sum, 1e-8)

    ola_buffer = istft_frame[hop:].copy()
    window_sum_buffer = istft_window_sq[hop:].copy()

    frames_output_norm.append(output)
    print(f"Frame {frame_idx}: mean normalized output = {np.mean(output):.6f}")

full_output_norm = np.concatenate(frames_output_norm)
print(f"\nMean of normalized samples 256-1024: {np.mean(full_output_norm[hop:4*hop]):.6f}")
