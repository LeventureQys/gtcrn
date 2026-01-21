#!/usr/bin/env python3
"""
Simulate the exact C streaming ISTFT to find the bug.
"""

import numpy as np
import os
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

# Load Python complete processing for reference spectrum
c_complete_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c.wav")
c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")

c_complete, sr = sf.read(c_complete_path)
c_stream, _ = sf.read(c_stream_path)

hop = 256
win = 512
n_fft = 512
n_freqs = n_fft // 2 + 1

# sqrt-Hann window (same as C)
window = np.zeros(win, dtype=np.float32)
for i in range(win):
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / win))
    window[i] = np.sqrt(hann)

print("=== Simulating C ISTFT approaches ===")

# Use a simple test: synthesize a known signal
test_len = 10000
test_sig = np.sin(2 * np.pi * 440 * np.arange(test_len) / 16000).astype(np.float32)

# STFT (matching C behavior)
# center=False: frame starts at sample 0, not -256
n_frames = (test_len - n_fft) // hop + 1

stft_real = np.zeros((n_frames, n_freqs), dtype=np.float32)
stft_imag = np.zeros((n_frames, n_freqs), dtype=np.float32)

for t in range(n_frames):
    start = t * hop
    frame = test_sig[start:start+win] * window

    # Zero-pad if needed (n_fft == win here, so no padding needed)
    fft_out = np.fft.fft(frame, n_fft)
    stft_real[t] = fft_out[:n_freqs].real
    stft_imag[t] = fft_out[:n_freqs].imag

print(f"STFT: {n_frames} frames")

# Method 1: Batch ISTFT (like gtcrn_istft in C - with window normalization)
batch_output = np.zeros(test_len, dtype=np.float32)
window_sum = np.zeros(test_len, dtype=np.float32)

for t in range(n_frames):
    # Reconstruct full spectrum
    full_spec = np.zeros(n_fft, dtype=np.complex64)
    full_spec[:n_freqs] = stft_real[t] + 1j * stft_imag[t]
    full_spec[n_freqs:] = np.conj(full_spec[1:n_fft//2][::-1])

    # IFFT
    frame_time = np.fft.ifft(full_spec).real.astype(np.float32)

    # Apply window
    frame_windowed = frame_time * window

    # Overlap-add
    start = t * hop
    batch_output[start:start+win] += frame_windowed
    window_sum[start:start+win] += window ** 2

# Normalize
batch_output /= np.maximum(window_sum, 1e-8)

# Compare mid-section
mid_start = 5 * hop
mid_end = test_len - 5 * hop
error_batch = np.sqrt(np.mean((test_sig[mid_start:mid_end] - batch_output[mid_start:mid_end])**2))
print(f"Batch ISTFT (with normalization) error: {error_batch:.8f}")

# Method 2: Streaming ISTFT (exactly like C code)
stream_output = []
ola_buffer = np.zeros(hop, dtype=np.float32)
first_frame = True

for t in range(n_frames):
    # Reconstruct full spectrum
    full_spec = np.zeros(n_fft, dtype=np.complex64)
    full_spec[:n_freqs] = stft_real[t] + 1j * stft_imag[t]
    full_spec[n_freqs:] = np.conj(full_spec[1:n_fft//2][::-1])

    # IFFT
    frame_time = np.fft.ifft(full_spec).real.astype(np.float32)

    # Apply window
    istft_frame = frame_time * window

    # C streaming logic
    if first_frame:
        output = istft_frame[:hop].copy()
        first_frame = False
    else:
        output = ola_buffer + istft_frame[:hop]

    ola_buffer = istft_frame[hop:].copy()
    stream_output.append(output)

stream_output = np.concatenate(stream_output)
print(f"Stream output length: {len(stream_output)}")

# Compare with batch (need to align)
# Stream output starts at sample 0, batch also starts at 0
# But batch has proper window normalization, stream doesn't
compare_len = min(len(stream_output), len(batch_output))
stream_rms = np.sqrt(np.mean(stream_output[mid_start:mid_end]**2))
batch_rms = np.sqrt(np.mean(batch_output[mid_start:mid_end]**2))
print(f"\nStream RMS: {stream_rms:.6f}")
print(f"Batch RMS:  {batch_rms:.6f}")
print(f"Stream / Batch ratio: {stream_rms / batch_rms:.4f}")

# Method 3: Streaming with normalization
# Since window_sum in steady state = 1.0, no division needed in theory
# But what's the actual value?

# For position i in output hop [start, start+256):
#   - Contribution from frame t:   window[i-start] * sample
#   - Contribution from frame t-1: window[i-start+256] * sample (from previous frame's second half)
#   - Total window_sum = window[i-start]^2 + window[i-start+256]^2 = 1.0 for sqrt-Hann!

# So in theory, the streaming overlap-add should give correct amplitude WITHOUT normalization
# But our stream_output is different...

# Wait! The issue is that without normalization, we're summing windowed values, not dividing
# Let me check what the steady-state sum actually looks like

print("\n=== Analyzing steady-state window sum ===")
# At position 256 (start of second hop):
# - Frame 1 contributes window[0] * x = 0
# - Frame 0 contributes window[256] * x = x
# No! This is wrong thinking.

# Actually for OLA:
# output[256] = ola_buffer[0] + istft_frame[0]
#             = prev_frame_windowed[256] + curr_frame_windowed[0]
#             = window[256] * x_prev + window[0] * x_curr
# where x_prev and x_curr are the time-domain IFFT outputs

# If window[256] = 1.0 and window[0] = 0, then output[256] = x_prev * 1.0 + x_curr * 0

# Wait, but the window is applied to BOTH frames' IFFT outputs
# So if the original signal was constant 1.0:
# IFFT of STFT(1.0) = 1.0 (if window energy is normalized)
# Then windowed IFFT = window * 1.0 = window
# And OLA at position k in hop:
#   = window[k+256] * 1.0 + window[k] * 1.0 = window[k+256] + window[k]

# Let me verify
print("Window[0] + Window[256] =", window[0] + window[256])
print("Window[127] + Window[383] =", window[127] + window[383])
print("Window[255] + Window[511] =", window[255] + window[511])

# These should sum to about sqrt(2) ≈ 1.414 because window^2 sums to 1, not window itself!
# sqrt-Hann: w[i] + w[i+256] ≈ sqrt(2) in middle, varies at edges

# SO THE FIX: We need to divide by the appropriate normalization factor!
# For WOLA: divide by sum of window^2 = 1.0
# But we're summing window * signal, not window^2 * signal!

# The correct WOLA formula is:
# output = sum(window * istft_frame) / sum(window^2)
# Since sum(window^2) = 1.0 for sqrt-Hann at 50% overlap, we just need to make sure
# we're getting the right values before normalization

# Actually I think I've been confusing myself. Let me trace through more carefully:
#
# STFT analysis: x * window -> FFT
# ISTFT synthesis: IFFT -> y * window
#
# For perfect reconstruction: output = x
# This requires: sum_t(window[n-t*hop]^2) = 1 (COLA condition)
#
# With sqrt-Hann and 50% overlap, window[n]^2 + window[n+256]^2 = 1
# So the OLA of (IFFT * window) should give back the original signal!
#
# But we're not getting that. Let me check if there's a scale factor in the FFT

# Check FFT scaling
test_frame = np.ones(n_fft, dtype=np.float32)
fft_out = np.fft.fft(test_frame)
ifft_out = np.fft.ifft(fft_out).real
print(f"\nFFT->IFFT of ones: {ifft_out[0]:.6f} (should be 1.0)")

# Check with window
windowed = test_frame * window
fft_out = np.fft.fft(windowed)
ifft_out = np.fft.ifft(fft_out).real
print(f"FFT->IFFT of windowed ones: {ifft_out[0]:.6f} (should be window[0]={window[0]:.6f})")

# The FFT/IFFT is correct. The issue must be in how we're doing overlap-add
# Let me check what the C code is actually outputting vs what it should output

print("\n=== Correct streaming OLA ===")
# For correct reconstruction, after OLA we should get the original signal
# But the C code does OLA without dividing by window_sum
# And window_sum = 1.0 in steady state
# So the issue is... hmm

# Let me just compute what the output SHOULD be and compare
correct_stream = []
ola_buffer = np.zeros(hop, dtype=np.float32)
first_frame = True

for t in range(n_frames):
    # Reconstruct full spectrum
    full_spec = np.zeros(n_fft, dtype=np.complex64)
    full_spec[:n_freqs] = stft_real[t] + 1j * stft_imag[t]
    full_spec[n_freqs:] = np.conj(full_spec[1:n_fft//2][::-1])

    # IFFT
    frame_time = np.fft.ifft(full_spec).real.astype(np.float32)

    # Apply window
    istft_frame = frame_time * window

    # Correct OLA: always overlap-add, even on first frame (but first frame's first half has no prior contribution)
    if first_frame:
        # First frame: output is just istft_frame[0:256], no overlap yet
        output = istft_frame[:hop].copy()
        first_frame = False
    else:
        # Subsequent frames: overlap-add
        output = ola_buffer + istft_frame[:hop]

    # IMPORTANT: Divide by window_sum!
    # At position i in hop: window_sum[i] = window[i]^2 + window[i+256]^2 = 1.0 (for sqrt-Hann)
    # But on first frame, window_sum[i] = window[i]^2 only!
    # So we should NOT normalize first frame the same way

    # Wait, but we're outputting hop-sized chunks, not the full signal
    # The window_sum for each position in the output hop is always 1.0 in steady state
    # because output[i] = ola_buffer[i] + istft_frame[i]
    #                   = prev_window[i+256] * x + curr_window[i] * x
    # and window[i]^2 + window[i+256]^2 = 1.0

    ola_buffer = istft_frame[hop:].copy()
    correct_stream.append(output)

correct_stream = np.concatenate(correct_stream)

# Compare
print(f"Correct stream RMS: {np.sqrt(np.mean(correct_stream[mid_start:mid_end]**2)):.6f}")
print(f"Original signal RMS: {np.sqrt(np.mean(test_sig[mid_start:mid_end]**2)):.6f}")

# Hmm, they should be the same...
# Let me check more directly
corr = np.corrcoef(test_sig[mid_start:mid_end], correct_stream[mid_start:mid_end])[0, 1]
print(f"Correlation: {corr:.6f}")

ratio = np.sqrt(np.mean(correct_stream[mid_start:mid_end]**2)) / np.sqrt(np.mean(test_sig[mid_start:mid_end]**2))
print(f"Energy ratio: {ratio:.6f}")
