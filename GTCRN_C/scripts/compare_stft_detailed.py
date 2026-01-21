#!/usr/bin/env python3
"""
Check if C streaming STFT matches Python STFT with center=False.
"""

import numpy as np
import torch
import os
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

# Load test audio
test_wav = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
audio, sr = sf.read(test_wav)
audio = audio.astype(np.float32)

n_fft = 512
hop = 256
win = 512

# sqrt-Hann window (same formula as C)
window = np.zeros(win, dtype=np.float32)
for i in range(win):
    hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / win))
    window[i] = np.sqrt(hann)

# Python STFT with center=False
audio_tensor = torch.from_numpy(audio)
window_tensor = torch.from_numpy(window)
py_stft = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop, win_length=win,
                     window=window_tensor, center=False, return_complex=True)
print(f"Python STFT shape: {py_stft.shape}")  # (freq, time)
num_frames = py_stft.shape[1]

# Simulate C streaming STFT
print("\nSimulating C streaming STFT...")
c_stft_real = []
c_stft_imag = []
stft_input_buffer = np.zeros(hop, dtype=np.float32)

for frame_idx in range(num_frames + 1):  # +1 to match frame count
    # Build 512-sample window (C style)
    if frame_idx * hop + hop <= len(audio):
        current_frame = audio[frame_idx * hop : frame_idx * hop + hop]
    else:
        current_frame = np.zeros(hop, dtype=np.float32)

    stft_window = np.concatenate([stft_input_buffer, current_frame])
    stft_input_buffer = current_frame.copy()

    # Apply window and FFT
    windowed = stft_window * window
    fft_out = np.fft.fft(windowed, n_fft)

    c_stft_real.append(fft_out[:n_fft//2+1].real.astype(np.float32))
    c_stft_imag.append(fft_out[:n_fft//2+1].imag.astype(np.float32))

c_stft_real = np.array(c_stft_real).T  # (freq, time)
c_stft_imag = np.array(c_stft_imag).T

print(f"C STFT shape: {c_stft_real.shape}")

# Compare frames
print("\n=== Comparing STFT frames ===")
print(f"Python has {num_frames} frames")
print(f"C has {c_stft_real.shape[1]} frames")

# Check frame alignment
# Python center=False: frame 0 uses samples [0:512]
# C streaming: frame 0 uses samples [zeros(256), audio[0:256]] (different!)
#              frame 1 uses samples [audio[0:256], audio[256:512]] (matches Python frame 0!)

print("\nPython frame 0 vs C frame 1:")
py_frame0 = py_stft[:, 0].numpy()
c_frame1 = c_stft_real[:, 1] + 1j * c_stft_imag[:, 1]
corr = np.abs(np.corrcoef(np.abs(py_frame0), np.abs(c_frame1))[0, 1])
print(f"  Magnitude correlation: {corr:.6f}")
print(f"  Python frame 0 DC: {py_frame0[0]:.4f}")
print(f"  C frame 1 DC: {c_frame1[0]:.4f}")

print("\nPython frame 1 vs C frame 2:")
py_frame1 = py_stft[:, 1].numpy()
c_frame2 = c_stft_real[:, 2] + 1j * c_stft_imag[:, 2]
corr = np.abs(np.corrcoef(np.abs(py_frame1), np.abs(c_frame2))[0, 1])
print(f"  Magnitude correlation: {corr:.6f}")
print(f"  Python frame 1 DC: {py_frame1[0]:.4f}")
print(f"  C frame 2 DC: {c_frame2[0]:.4f}")

print("\n=== Frame alignment conclusion ===")
print("C streaming frame N corresponds to Python frame N-1")
print("This is the 1-frame offset we already identified.")

# What about the energy? Let's check
print("\n=== Energy comparison ===")
py_energy = np.mean(np.abs(py_stft.numpy())**2)
c_energy = np.mean(c_stft_real**2 + c_stft_imag**2)
print(f"Python STFT energy: {py_energy:.4f}")
print(f"C STFT energy: {c_energy:.4f}")
print(f"Ratio: {c_energy / py_energy:.4f}")

# Compare aligned frames
print("\n=== Aligned frame comparison (C[1:] vs Py[:-1]) ===")
min_frames = min(num_frames - 1, c_stft_real.shape[1] - 1)
c_aligned = c_stft_real[:, 1:min_frames+1] + 1j * c_stft_imag[:, 1:min_frames+1]
py_aligned = py_stft[:, :min_frames].numpy()

c_energy_aligned = np.mean(np.abs(c_aligned)**2)
py_energy_aligned = np.mean(np.abs(py_aligned)**2)
print(f"C aligned energy: {c_energy_aligned:.4f}")
print(f"Python aligned energy: {py_energy_aligned:.4f}")
print(f"Ratio: {c_energy_aligned / py_energy_aligned:.4f}")

# Element-wise comparison
diff = c_aligned - py_aligned
max_diff = np.max(np.abs(diff))
mean_diff = np.mean(np.abs(diff))
print(f"Max absolute difference: {max_diff:.6f}")
print(f"Mean absolute difference: {mean_diff:.6f}")
print(f"Relative difference: {mean_diff / np.mean(np.abs(py_aligned)):.6f}")
