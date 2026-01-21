#!/usr/bin/env python3
"""
Check PyTorch STFT behavior vs C STFT.
"""

import torch
import numpy as np

# Create test signal
n_fft = 512
hop = 256
win_length = 512

# sqrt-Hann window
window = torch.sqrt(torch.hann_window(win_length))
window_np = window.numpy()

# Test signal: constant 1.0
test_sig = torch.ones(2048)

# PyTorch STFT
stft_out = torch.stft(test_sig, n_fft=n_fft, hop_length=hop, win_length=win_length,
                      window=window, center=False, return_complex=True)
print(f"STFT output shape: {stft_out.shape}")
print(f"STFT frame 4 DC component: {stft_out[0, 4]:.4f}")

# What should the DC component be for a constant signal?
# If signal is 1.0 and window is applied, then:
# windowed = 1.0 * window = window
# FFT of window: sum(window) at DC bin
window_sum_fft = np.sum(window_np)
print(f"Sum of window (expected DC for windowed constant): {window_sum_fft:.4f}")

# Now do IFFT and check
spec = stft_out[:, 4]  # Frame 4
full_spec = torch.zeros(n_fft, dtype=torch.complex64)
full_spec[:n_fft//2+1] = spec
full_spec[n_fft//2+1:] = torch.conj(torch.flip(spec[1:-1], [0]))
frame_time = torch.fft.ifft(full_spec).real

print(f"\nIFFT of frame 4 (no synthesis window):")
print(f"  First 5: {frame_time[:5].numpy()}")
print(f"  Middle: {frame_time[256:261].numpy()}")

# The IFFT should give us back the windowed signal
# Since input was constant 1.0, IFFT should give us back window
print(f"\nExpected (window values):")
print(f"  First 5: {window_np[:5]}")
print(f"  Middle: {window_np[256:261]}")

# Apply synthesis window
frame_windowed = frame_time.numpy() * window_np
print(f"\nAfter synthesis window (window * IFFT):")
print(f"  First 5: {frame_windowed[:5]}")
print(f"  Middle: {frame_windowed[256:261]}")
print(f"  Sum: {np.sum(frame_windowed):.4f}")

# Compare with window^2
print(f"\nExpected (window^2):")
print(f"  First 5: {window_np[:5]**2}")
print(f"  Middle: {window_np[256:261]**2}")
print(f"  Sum: {np.sum(window_np**2):.4f}")

# So after STFT -> IFFT -> synthesis window, we get window^2, not window!
# The overlap-add of window^2 at each position gives window_sum,
# which we divide by to get the original signal.

# For C streaming:
# - STFT applies window: input * window -> FFT
# - ISTFT applies window: IFFT -> result * window
# - OLA: sum of (IFFT * window) = sum of window^2 for constant input
# - No division by window_sum in C streaming!

# So C should output window^2 values, not 1.0
# And window_sum = 1.0 in steady state for sqrt-Hann
# So after OLA (without division), output = window^2 at each position
# But wait, window^2 sums to 1.0 across overlapping frames!

# Let me trace through more carefully:
print("\n=== Tracing OLA for constant input ===")
# Frame N at position hop*N:
#   istft_frame = IFFT(STFT(1 * window)) * window = window * window = window^2
# OLA at position i (within hop):
#   output[i] = prev_istft[i+hop] + curr_istft[i]
#             = window[i+hop]^2 + window[i]^2
#             = 1.0 (for sqrt-Hann!)

print("For constant input = 1.0:")
print("  STFT windowed input: 1.0 * window = window")
print("  IFFT output: window (since FFT->IFFT is identity)")
print("  Synthesis windowed: window * window = window^2")
print("  OLA: window[i]^2 + window[i+256]^2 = 1.0 for all i")
print("")
print("So C streaming SHOULD produce 1.0 for constant input!")
print("The Python simulation confirmed this.")
print("")
print("The issue must be elsewhere - possibly in the neural network or STFT input construction.")
