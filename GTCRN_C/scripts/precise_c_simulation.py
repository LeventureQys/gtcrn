#!/usr/bin/env python3
"""
Precisely simulate the C streaming ISTFT OLA to find the energy bug.
"""

import numpy as np

win = 512
hop = 256
n_fft = 512
n_freqs = n_fft // 2 + 1

# sqrt-Hann window
window = np.zeros(win, dtype=np.float32)
for i in range(win):
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / win))
    window[i] = np.sqrt(hann)

# Create test signal: constant 1.0
n_frames = 20
test_signal = np.ones(n_frames * hop + win, dtype=np.float32)

print("=== Simulating C streaming exactly ===")

# First, do STFT like C does (using stft_input_buffer)
stft_input_buffer = np.zeros(hop, dtype=np.float32)
spectrums_real = []
spectrums_imag = []

for frame_idx in range(n_frames):
    # Build 512-sample window
    stft_window = np.concatenate([stft_input_buffer, test_signal[frame_idx*hop:(frame_idx+1)*hop]])

    # Update buffer for next frame
    stft_input_buffer = test_signal[frame_idx*hop:(frame_idx+1)*hop].copy()

    # Apply window and FFT
    windowed = stft_window * window
    fft_out = np.fft.fft(windowed, n_fft)

    spectrums_real.append(fft_out[:n_freqs].real.astype(np.float32))
    spectrums_imag.append(fft_out[:n_freqs].imag.astype(np.float32))

print(f"Generated {len(spectrums_real)} frames of STFT")

# Now do streaming ISTFT like C does
ola_buffer = np.zeros(hop, dtype=np.float32)
first_frame = True
c_style_output = []

for frame_idx in range(n_frames):
    spec_real = spectrums_real[frame_idx]
    spec_imag = spectrums_imag[frame_idx]

    # Reconstruct full spectrum (C style)
    full_real = np.zeros(n_fft, dtype=np.float32)
    full_imag = np.zeros(n_fft, dtype=np.float32)

    full_real[:n_freqs] = spec_real
    full_imag[:n_freqs] = spec_imag

    # Conjugate symmetry
    for i in range(1, n_fft // 2):
        full_real[n_fft - i] = full_real[i]
        full_imag[n_fft - i] = -full_imag[i]

    # IFFT (C style with conjugate trick)
    full_imag_neg = -full_imag  # conjugate
    full_complex = full_real + 1j * full_imag_neg
    fft_out = np.fft.fft(full_complex)
    fft_out_conj = np.conj(fft_out) / n_fft  # conjugate and scale
    frame_time = fft_out_conj.real.astype(np.float32)

    # Apply window (gtcrn_istft_frame line 244)
    istft_frame = frame_time * window

    # OLA (gtcrn_process_frame lines 370-384)
    if first_frame:
        output = istft_frame[:hop].copy()
        first_frame = False
    else:
        output = ola_buffer + istft_frame[:hop]

    ola_buffer = istft_frame[hop:].copy()
    c_style_output.append(output)

c_style_output = np.concatenate(c_style_output)
print(f"C-style output length: {len(c_style_output)}")

# Skip first 5 frames for steady state
start = 5 * hop
end = 15 * hop
print(f"\nSteady state (samples {start}-{end}):")
print(f"  C-style output mean: {np.mean(c_style_output[start:end]):.6f} (expected: 1.0)")
print(f"  C-style output RMS:  {np.sqrt(np.mean(c_style_output[start:end]**2)):.6f} (expected: 1.0)")

# Now do proper ISTFT with normalization
print("\n=== Proper ISTFT with normalization ===")
proper_output = np.zeros(n_frames * hop + win, dtype=np.float32)
window_sum = np.zeros(n_frames * hop + win, dtype=np.float32)

for frame_idx in range(n_frames):
    spec_real = spectrums_real[frame_idx]
    spec_imag = spectrums_imag[frame_idx]

    # Full spectrum
    full_real = np.zeros(n_fft, dtype=np.float32)
    full_imag = np.zeros(n_fft, dtype=np.float32)
    full_real[:n_freqs] = spec_real
    full_imag[:n_freqs] = spec_imag
    for i in range(1, n_fft // 2):
        full_real[n_fft - i] = full_real[i]
        full_imag[n_fft - i] = -full_imag[i]

    # IFFT
    full_imag_neg = -full_imag
    full_complex = full_real + 1j * full_imag_neg
    fft_out = np.fft.fft(full_complex)
    fft_out_conj = np.conj(fft_out) / n_fft
    frame_time = fft_out_conj.real.astype(np.float32)

    # Apply window
    istft_frame = frame_time * window

    # OLA
    pos = frame_idx * hop
    proper_output[pos:pos+win] += istft_frame
    window_sum[pos:pos+win] += window ** 2

# Normalize
proper_output /= np.maximum(window_sum, 1e-8)

print(f"Proper output mean (samples {start}-{end}): {np.mean(proper_output[start:end]):.6f}")
print(f"Proper output RMS:  {np.sqrt(np.mean(proper_output[start:end]**2)):.6f}")

# Compare
print(f"\nRatio (C-style / proper): {np.mean(c_style_output[start:end]) / np.mean(proper_output[start:end]):.6f}")

# The key insight: in the C streaming code, we're not adding the FULL overlap!
# Let me trace through more carefully:
print("\n=== Detailed frame-by-frame trace ===")
print("For constant input signal = 1.0:")
print("")

# Reset and trace
stft_input_buffer = np.zeros(hop, dtype=np.float32)
ola_buffer = np.zeros(hop, dtype=np.float32)
first_frame = True

for frame_idx in range(5):
    # Build STFT window
    stft_window = np.concatenate([stft_input_buffer, np.ones(hop)])
    stft_input_buffer = np.ones(hop)  # update for next frame

    print(f"Frame {frame_idx}:")
    print(f"  STFT input first 5: {stft_window[:5]}")
    print(f"  STFT input last 5:  {stft_window[-5:]}")

    # Windowed input
    windowed = stft_window * window
    print(f"  Windowed first 5:   {windowed[:5]}")
    print(f"  Windowed last 5:    {windowed[-5:]}")

    # FFT -> IFFT (for simplicity, assume perfect reconstruction)
    # IFFT(FFT(windowed)) = windowed
    frame_time = windowed  # simplified

    # Apply window again (this is what C does in ISTFT)
    istft_frame = frame_time * window
    print(f"  ISTFT frame first 5: {istft_frame[:5]}")
    print(f"  ISTFT frame last 5:  {istft_frame[-5:]}")

    # OLA
    if first_frame:
        output = istft_frame[:hop].copy()
        first_frame = False
    else:
        output = ola_buffer + istft_frame[:hop]

    print(f"  OLA buffer first 5:  {ola_buffer[:5]}")
    print(f"  Output first 5:      {output[:5]}")
    print(f"  Output mean:         {np.mean(output):.6f}")

    ola_buffer = istft_frame[hop:].copy()
    print("")
