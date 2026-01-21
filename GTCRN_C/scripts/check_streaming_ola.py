#!/usr/bin/env python3
"""
Check what the streaming OLA window_sum actually is.
"""

import numpy as np

win = 512
hop = 256

# sqrt-Hann window (same formula as C)
window = np.zeros(win, dtype=np.float32)
for i in range(win):
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / win))
    window[i] = np.sqrt(hann)

print("=== Window properties ===")
print(f"Window energy: {np.sum(window**2):.6f}")
print(f"Window mean: {np.mean(window):.6f}")
print(f"Window max: {np.max(window):.6f}")

# In streaming OLA, for output position i (within a hop):
# - Current frame contributes: window[i]^2
# - Previous frame contributes: window[i + hop]^2
# Total: window[i]^2 + window[i + hop]^2

print("\n=== Streaming OLA window_sum ===")
window_sum = np.zeros(hop)
for i in range(hop):
    window_sum[i] = window[i]**2 + window[i + hop]**2

print(f"window_sum min: {np.min(window_sum):.6f}")
print(f"window_sum max: {np.max(window_sum):.6f}")
print(f"window_sum mean: {np.mean(window_sum):.6f}")

# So window_sum is indeed ~1.0. The issue is not in the normalization.

# Let me think about this differently.
# The C streaming outputs ~60% of the expected energy.
# 0.6 ≈ 1.0 - 0.4 ≈ 0.6
# 0.6 ≈ sqrt(0.36) ≈ 0.6
# 0.6 ≈ 0.707^2 * 1.2 ≈ 0.6

# Wait! 0.6 is close to the mean of the window!
print(f"\n=== Hypothesis testing ===")
print(f"Mean of window: {np.mean(window):.6f}")
print(f"Mean of window^2: {np.mean(window**2):.6f}")

# Hmm, mean of window is ~0.63, which is close to our ratio!
# This suggests that maybe we're multiplying by window when we shouldn't be,
# or we're missing one window application somewhere.

# In STFT/ISTFT with sqrt-Hann:
# - STFT: x * window -> FFT -> spectrum
# - ISTFT: spectrum -> IFFT -> x' * window
# - OLA: sum(x' * window) / sum(window^2)
#
# If we correctly apply window in both STFT and ISTFT, and sum(window^2) = 1 at each position,
# then: output = sum((IFFT(FFT(x * window))) * window) / 1.0 = x
#
# But if we're somehow applying window only once (either STFT or ISTFT but not both),
# then we'd get output = x * window (on average)

# Let's verify: if output = x * window (on average), then:
# RMS(output) / RMS(x) = mean(window) ≈ 0.63
# This matches our observation!

# So the issue is: either the C STFT or ISTFT is NOT applying the window correctly.

print("\n=== Window sum for different positions ===")
for pos in [0, 64, 128, 192, 255]:
    ws = window[pos]**2 + window[pos + hop]**2
    print(f"Position {pos}: window_sum = {ws:.6f}")

# Let me also check: what if the C code is applying window TWICE in STFT but not in ISTFT?
# Then STFT output = FFT(x * window^2), and ISTFT output = IFFT(STFT) = x * window^2
# But that would give lower energy, not match what we see.

# What if C code is applying window in STFT but NOT in ISTFT?
# Then STFT output = FFT(x * window), and ISTFT output = IFFT(STFT) = x * window
# OLA of (x * window) without dividing by window^2 would give... let's see

print("\n=== Simulating no-window ISTFT ===")
# If ISTFT doesn't apply window, what happens?
# STFT: x * window -> FFT -> spec
# ISTFT: spec -> IFFT -> x * window (no additional window)
# OLA: sum(x * window) / ?
#
# Wait, in the current C code, ISTFT DOES apply window:
# output[i] = real[pad_left + i] * stft->window[i];  (line 244 in gtcrn_fft.c)
#
# So the window IS applied in both STFT and ISTFT. The issue must be elsewhere.

# Let me check if maybe the issue is that we're building the 512-sample window incorrectly
# in streaming mode...

print("\n=== Checking streaming STFT input construction ===")
# In gtcrn_process_frame:
# stft_window = [stft_input_buffer (prev 256)] + [input_frame (current 256)]
# This gives 512 samples, which should be correct.

# Hmm, let me check if the issue is in the OLA buffer handling.
# Frame 0:
#   STFT window = [zeros(256)] + [input[0:256]] (since stft_input_buffer is initialized to zeros)
#   ISTFT gives istft_frame[0:512]
#   Output = istft_frame[0:256] (no OLA yet)
#   OLA buffer = istft_frame[256:512]
#
# Frame 1:
#   STFT window = [input[0:256]] + [input[256:512]]
#   ISTFT gives istft_frame[0:512]
#   Output = OLA_buffer + istft_frame[0:256]
#          = prev_istft[256:512] + curr_istft[0:256]
#   OLA buffer = istft_frame[256:512]
#
# This seems correct for 50% overlap OLA...

# Wait! The first frame has zeros in the first half of the STFT window!
# This means the first frame's STFT is processing [0, 0, ..., 0, input[0], input[1], ...]
# But Python center=False would process [input[0], input[1], ..., input[511]]!

print("\n=== First frame mismatch! ===")
print("C streaming first frame STFT input: [zeros(256)] + [input[0:256]]")
print("Python center=False first frame: [input[0:512]]")
print("")
print("This causes a 1-frame delay in C streaming, which we've already identified.")
print("But it shouldn't affect the steady-state energy ratio...")

# Let me verify by computing what the ratio should be if there's a 256-sample shift
# in the STFT analysis:
print("\n=== Effect of shifted STFT window ===")
# If C analyzes [prev_hop, current_hop] while Python analyzes [current_hop, next_hop],
# the spectrums would be different but the energy should be similar after neural network processing.
# The neural network doesn't change the energy ratio significantly.

# I think the issue might be more fundamental. Let me check if the C code's overlap-add
# is actually accumulating correctly...
