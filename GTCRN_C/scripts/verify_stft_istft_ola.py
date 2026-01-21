#!/usr/bin/env python3
"""
Verify C's STFT->ISTFT->OLA cycle in Python.
The key insight is that ISTFT applies the window twice (once in STFT, once in ISTFT),
so the total effect is window^2. OLA sums overlapping window^2 to get normalization.
"""

import numpy as np

n_fft = 512
hop = 256
win = 512

# sqrt-Hann window
window = np.zeros(win, dtype=np.float32)
for i in range(win):
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / win))
    window[i] = np.sqrt(hann)

print("=== Window analysis ===")
print(f"Window sum: {np.sum(window):.4f}")
print(f"Window^2 sum: {np.sum(window**2):.4f}")

# For 50% overlap, at steady state, each sample is covered by two windows
# The normalization factor at steady state is window[i]^2 + window[i+256]^2
# But wait, for streaming OLA, we're NOT dividing by the normalization!

# In C streaming code:
# 1. STFT: input -> input * window -> FFT -> spectrum
# 2. Neural network: spectrum -> output_spectrum
# 3. ISTFT: output_spectrum -> IFFT -> output_time * window
# 4. OLA: output = ola_buffer[0:256] + istft_frame[0:256]
#         ola_buffer = istft_frame[256:512]

# For a signal that passes through the neural network unchanged:
# - STFT: x * window -> X
# - ISTFT: X -> x * window * window (because IFFT(X) = x*window, then multiply by window again)
# - OLA: combines two consecutive frames

# Let's verify this with a simple test signal
print("\n=== Simulating C STFT->ISTFT->OLA ===")

# Create a simple sinusoidal signal
test_signal = np.sin(2 * np.pi * 1000 * np.arange(2048) / 16000).astype(np.float32)

# Process multiple frames
outputs = []
ola_buffer = np.zeros(hop, dtype=np.float32)
stft_input_buffer = np.zeros(hop, dtype=np.float32)
first_frame = True

for frame_idx in range(6):
    start = frame_idx * hop
    current_frame = test_signal[start:start + hop]

    # Build STFT window
    stft_window = np.concatenate([stft_input_buffer, current_frame])
    stft_input_buffer = current_frame.copy()

    # STFT
    windowed = stft_window * window
    fft_out = np.fft.fft(windowed, n_fft)
    spec_real = fft_out[:n_fft//2+1].real.astype(np.float32)
    spec_imag = fft_out[:n_fft//2+1].imag.astype(np.float32)

    # "Neural network" - just pass through (identity)
    out_spec_real = spec_real
    out_spec_imag = spec_imag

    # ISTFT (C style - apply window again)
    full_spec = np.zeros(n_fft, dtype=np.complex64)
    full_spec[:n_fft//2+1] = out_spec_real + 1j * out_spec_imag
    for i in range(1, n_fft // 2):
        full_spec[n_fft - i] = np.conj(full_spec[i])
    frame_time = np.fft.ifft(full_spec).real.astype(np.float32)
    istft_frame = frame_time * window  # Second window application!

    # OLA (C style)
    if first_frame:
        output = istft_frame[:hop].copy()
        first_frame = False
    else:
        output = ola_buffer + istft_frame[:hop]

    ola_buffer = istft_frame[hop:].copy()
    outputs.append(output)

    # Analysis
    if frame_idx >= 2:
        # At steady state (after first 2 frames), compare output with expected
        # Expected: input * window^2 (summed via OLA)
        # For 50% overlap with sqrt-Hann: window[i]^2 + window[i+256]^2 = 1.0

        # The input to this frame's STFT was [prev_frame, curr_frame]
        # After processing, output should be approximately prev_frame (delayed by 1 frame)
        expected = test_signal[(frame_idx - 1) * hop:(frame_idx) * hop]

        correlation = np.corrcoef(output, expected)[0, 1]
        ratio = np.sqrt(np.mean(output**2)) / np.sqrt(np.mean(expected**2))

        print(f"Frame {frame_idx}: correlation = {correlation:.4f}, energy ratio = {ratio:.4f}")

print("\n=== Analysis ===")
print("For sqrt-Hann window with 50% overlap:")
print("  window[i]^2 + window[i+256]^2 = ?")
for i in [0, 64, 128, 192, 255]:
    norm = window[i]**2 + window[i + hop]**2
    print(f"  i={i}: window[{i}]^2 + window[{i+256}]^2 = {norm:.4f}")

print("\n=== Key insight ===")
print("The STFT applies window once, ISTFT applies window again.")
print("Total effect: signal * window^2")
print("OLA sums overlapping window^2 contributions.")
print("For sqrt-Hann: window^2[i] + window^2[i+256] = 1.0 (perfect reconstruction!)")
