#!/usr/bin/env python3
"""
Debug STFT centering behavior.
"""

import torch
import numpy as np

# Create a simple test signal
signal = torch.arange(1024).float()

n_fft = 512
hop_length = 256
win_length = 512
window = torch.sqrt(torch.hann_window(win_length))

# Default STFT (center=True)
spec_centered = torch.stft(signal, n_fft=n_fft, hop_length=hop_length,
                           win_length=win_length, window=window, return_complex=False,
                           center=True)
print(f"Centered STFT shape: {spec_centered.shape}")  # (257, frames, 2)

# Non-centered STFT (center=False)
spec_not_centered = torch.stft(signal, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window, return_complex=False,
                                center=False)
print(f"Non-centered STFT shape: {spec_not_centered.shape}")

print("\nWith center=True (default):")
print(f"  Number of frames: {spec_centered.shape[1]}")
print(f"  Signal length: {len(signal)}")
print(f"  Expected frames with center=True: signal is padded by n_fft//2 on each side")

print("\nWith center=False:")
print(f"  Number of frames: {spec_not_centered.shape[1]}")
print(f"  Formula: (signal_len - n_fft) // hop_length + 1 = (1024 - 512) // 256 + 1 = 3")
