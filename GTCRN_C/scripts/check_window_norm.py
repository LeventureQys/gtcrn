#!/usr/bin/env python3
"""
Check sqrt-Hann window normalization.
"""

import numpy as np
import torch

win_length = 512
hop = 256

# sqrt-Hann window
window = torch.sqrt(torch.hann_window(win_length)).numpy()

# Check window sum for overlap-add
signal_len = 10 * hop
window_sum = np.zeros(signal_len)

for i in range(10):
    start = i * hop
    end = start + win_length
    if end <= signal_len:
        window_sum[start:end] += window ** 2

print("sqrt-Hann window squared sum for 50% overlap:")
print(f"  First 256 samples: {window_sum[:hop]}")
print(f"  Samples 256-512:   {window_sum[hop:2*hop]}")
print(f"  Middle samples:    {window_sum[2*hop:3*hop]}")
print(f"  Last 256 samples:  {window_sum[-hop:]}")

print("\nExpected: middle samples should sum to 1.0")
print(f"Actual middle value: {window_sum[2*hop]:.6f}")

# For streaming ISTFT, we need to normalize
# For proper overlap-add, after normalization output = sum(windowed_frames) / window_sum
