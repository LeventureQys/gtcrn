#!/usr/bin/env python3
"""
Compare window application between Python ISTFT and C ISTFT.
"""

import os
import numpy as np
import soundfile as sf
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

def main():
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    py_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_stream_nocenter.wav")

    c_stream, _ = sf.read(c_stream_path)
    py_stream, _ = sf.read(py_stream_path)

    hop = 256
    win = 512

    # sqrt-Hann window
    window = torch.sqrt(torch.hann_window(win)).numpy()

    print("=== Window sum analysis ===")
    # At steady state (after overlap), sum of squared windows = 1.0
    # So if we apply window once in STFT and once in ISTFT, we get window^2 total

    # Check energy in different regions
    print("\n=== Energy analysis (samples 25600-51200, after warmup) ===")
    start = 100 * hop
    end = 200 * hop

    c_seg = c_stream[start:end]
    py_seg = py_stream[start-hop:end-hop]  # Shift Python left by 1 frame to align

    c_energy = np.mean(c_seg**2)
    py_energy = np.mean(py_seg**2)

    print(f"C energy:      {c_energy:.8f}")
    print(f"Python energy: {py_energy:.8f}")
    print(f"Energy ratio:  {c_energy/py_energy:.4f}")
    print(f"Amplitude ratio: {np.sqrt(c_energy/py_energy):.4f}")

    # Check if there's a constant scale factor
    print("\n=== Scale factor analysis ===")
    # Compare samples directly
    min_len = min(len(c_seg), len(py_seg))
    c_seg = c_seg[:min_len]
    py_seg = py_seg[:min_len]

    # Filter out near-zero samples
    mask = np.abs(py_seg) > 0.01
    if np.any(mask):
        ratios = c_seg[mask] / py_seg[mask]
        print(f"Sample ratios (C/Py):")
        print(f"  Mean:   {np.mean(ratios):.4f}")
        print(f"  Median: {np.median(ratios):.4f}")
        print(f"  Std:    {np.std(ratios):.4f}")

    # Check if the issue is a missing window normalization
    # If C divides by window_sum but Python doesn't (or vice versa), we'd see this
    print("\n=== Checking if issue is window normalization ===")
    # Expected window_sum in steady state = 1.0 for sqrt-Hann
    # But during overlap-add, the contribution is:
    # output[i] = w[i]*frame_n[i] + w[i+256]*frame_n-1[i+256] for aligned positions

    # The key question: does Python ISTFT normalize by window_sum?
    # Let's check the PyTorch STFT/ISTFT behavior

    # Create a simple test signal
    test_sig = np.sin(2 * np.pi * 100 * np.arange(16000) / 16000).astype(np.float32)
    test_tensor = torch.from_numpy(test_sig)

    # STFT -> ISTFT round trip with center=False
    window_t = torch.sqrt(torch.hann_window(512))
    stft = torch.stft(test_tensor, n_fft=512, hop_length=256, win_length=512,
                      window=window_t, center=False, return_complex=True)

    # Manual ISTFT
    n_frames = stft.shape[1]
    expected_len = (n_frames - 1) * 256 + 512
    output = np.zeros(expected_len, dtype=np.float32)
    window_sum = np.zeros(expected_len, dtype=np.float32)
    window_np = window_t.numpy()

    for i in range(n_frames):
        # Reconstruct full spectrum
        frame_spec = stft[:, i]
        full_spec = torch.zeros(512, dtype=torch.complex64)
        full_spec[:257] = frame_spec
        full_spec[257:] = torch.conj(torch.flip(frame_spec[1:-1], [0]))
        frame_time = torch.fft.ifft(full_spec).real.numpy()

        # Apply synthesis window
        frame_windowed = frame_time * window_np

        # Overlap-add
        start = i * 256
        output[start:start+512] += frame_windowed
        window_sum[start:start+512] += window_np ** 2

    # Normalize
    output_norm = output / np.maximum(window_sum, 1e-8)

    # Compare with original
    min_len = min(len(test_sig), len(output_norm))
    # Skip first and last frames for fair comparison
    mid_start = 2 * 256
    mid_end = min_len - 2 * 256
    reconstruction_error = np.mean((test_sig[mid_start:mid_end] - output_norm[mid_start:mid_end])**2)
    print(f"Python manual ISTFT reconstruction error (mid section): {reconstruction_error:.10f}")

    # Without normalization
    output_no_norm = output / 1.0  # Just overlap-add
    reconstruction_error_no_norm = np.mean((test_sig[mid_start:mid_end] - output_no_norm[mid_start:mid_end])**2)
    print(f"Without normalization reconstruction error: {reconstruction_error_no_norm:.10f}")

    # What's the ratio without normalization?
    ratio_no_norm = np.mean(output_no_norm[mid_start:mid_end]) / np.mean(np.abs(test_sig[mid_start:mid_end]))
    print(f"Ratio without normalization: {ratio_no_norm:.4f}")


if __name__ == "__main__":
    main()
