#!/usr/bin/env python3
"""
Verify C implementation output against PyTorch reference.
"""

import os
import sys
import numpy as np
import torch
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gtcrn import GTCRN

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    gtcrn_dir = os.path.dirname(project_dir)

    input_wav = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
    c_output_wav = os.path.join(gtcrn_dir, "test_wavs/output_c/test_output.wav")
    pytorch_output_wav = os.path.join(gtcrn_dir, "test_wavs/output_c/pytorch_output.wav")
    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")

    print("=== GTCRN C vs PyTorch Verification ===\n")

    # Check if C output exists
    if not os.path.exists(c_output_wav):
        print(f"Error: C output file not found: {c_output_wav}")
        print("Please run gtcrn_demo first.")
        return 1

    # Load input audio
    print(f"Loading input: {input_wav}")
    audio_in, sr = sf.read(input_wav, dtype='float32')
    assert sr == 16000, f"Expected 16kHz, got {sr}Hz"
    print(f"  Samples: {len(audio_in)}, Duration: {len(audio_in)/sr:.2f}s")

    # Load C output
    print(f"\nLoading C output: {c_output_wav}")
    c_output, _ = sf.read(c_output_wav, dtype='float32')
    print(f"  Samples: {len(c_output)}")

    # Run PyTorch inference
    print(f"\nRunning PyTorch inference...")
    device = torch.device("cpu")
    model = GTCRN().to(device).eval()

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # STFT
    window = torch.hann_window(512).pow(0.5)
    x = torch.from_numpy(audio_in)
    x_stft = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]  # (1, F, T, 2)

    with torch.no_grad():
        y_stft = model(x_stft)  # (1, F, T, 2)

    # ISTFT
    y_complex = torch.complex(y_stft[0, :, :, 0], y_stft[0, :, :, 1])
    pytorch_output = torch.istft(y_complex, 512, 256, 512, window).detach().cpu().numpy()

    print(f"  PyTorch output samples: {len(pytorch_output)}")

    # Save PyTorch output for comparison
    sf.write(pytorch_output_wav, pytorch_output, 16000)
    print(f"  Saved to: {pytorch_output_wav}")

    # Compare outputs
    print("\n=== Comparison Results ===")

    # Truncate to same length
    min_len = min(len(c_output), len(pytorch_output))
    c_out = c_output[:min_len]
    py_out = pytorch_output[:min_len]

    # Calculate metrics
    abs_diff = np.abs(c_out - py_out)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    # Signal-to-Noise Ratio between outputs
    signal_power = np.mean(py_out ** 2)
    noise_power = np.mean((c_out - py_out) ** 2)
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')

    # Correlation
    correlation = np.corrcoef(c_out.flatten(), py_out.flatten())[0, 1]

    print(f"  Length (C output):       {len(c_output)}")
    print(f"  Length (PyTorch output): {len(pytorch_output)}")
    print(f"  Comparison length:       {min_len}")
    print(f"  Max absolute diff:       {max_diff:.6f}")
    print(f"  Mean absolute diff:      {mean_diff:.6f}")
    print(f"  SNR (C vs PyTorch):      {snr:.2f} dB")
    print(f"  Correlation:             {correlation:.6f}")

    # Verdict
    print("\n=== Verdict ===")
    if max_diff < 0.01 and correlation > 0.99:
        print("  PASS: C implementation matches PyTorch closely!")
        return 0
    elif max_diff < 0.1 and correlation > 0.9:
        print("  WARNING: Minor differences detected, but outputs are similar")
        return 0
    else:
        print("  FAIL: Significant differences between C and PyTorch outputs")
        return 1

if __name__ == "__main__":
    sys.exit(main())
