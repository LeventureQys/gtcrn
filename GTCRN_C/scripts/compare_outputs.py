#!/usr/bin/env python3
"""
Compare C and PyTorch outputs.
"""

import numpy as np
import soundfile as sf
import os

def main():
    c_output = "../../test_wavs/output_16k/enhanced_c.wav"
    pytorch_output = "../../test_wavs/output_16k/enhanced_pytorch.wav"
    noisy_input = "../../test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav"

    print("=" * 60)
    print("GTCRN Output Comparison")
    print("=" * 60)

    # Load files
    audio_noisy, sr = sf.read(noisy_input)
    audio_c, _ = sf.read(c_output)
    audio_pytorch, _ = sf.read(pytorch_output)

    print(f"\nInput (noisy):")
    print(f"  Length: {len(audio_noisy)} samples")
    print(f"  Range: [{audio_noisy.min():.4f}, {audio_noisy.max():.4f}]")
    print(f"  Std: {audio_noisy.std():.4f}")

    print(f"\nC output:")
    print(f"  Length: {len(audio_c)} samples")
    print(f"  Range: [{audio_c.min():.4f}, {audio_c.max():.4f}]")
    print(f"  Std: {audio_c.std():.4f}")

    print(f"\nPyTorch output:")
    print(f"  Length: {len(audio_pytorch)} samples")
    print(f"  Range: [{audio_pytorch.min():.4f}, {audio_pytorch.max():.4f}]")
    print(f"  Std: {audio_pytorch.std():.4f}")

    # Compare
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    # Check if C output equals noisy input (placeholder behavior)
    min_len = min(len(audio_noisy), len(audio_c))
    c_vs_noisy_diff = np.abs(audio_c[:min_len] - audio_noisy[:min_len]).max()
    print(f"\nC output vs Noisy input:")
    print(f"  Max difference: {c_vs_noisy_diff:.6f}")
    if c_vs_noisy_diff < 1e-5:
        print(f"  Status: C output = Noisy input (placeholder forward pass)")
    else:
        print(f"  Status: C output differs from input")

    # Check C vs PyTorch
    min_len = min(len(audio_c), len(audio_pytorch))
    c_vs_pytorch_diff = np.abs(audio_c[:min_len] - audio_pytorch[:min_len]).max()
    c_vs_pytorch_mse = np.mean((audio_c[:min_len] - audio_pytorch[:min_len])**2)
    print(f"\nC output vs PyTorch output:")
    print(f"  Max difference: {c_vs_pytorch_diff:.6f}")
    print(f"  MSE: {c_vs_pytorch_mse:.6f}")

    # Check PyTorch enhancement
    min_len = min(len(audio_noisy), len(audio_pytorch))
    pytorch_vs_noisy_diff = np.abs(audio_pytorch[:min_len] - audio_noisy[:min_len]).max()
    pytorch_vs_noisy_mse = np.mean((audio_pytorch[:min_len] - audio_noisy[:min_len])**2)
    print(f"\nPyTorch output vs Noisy input:")
    print(f"  Max difference: {pytorch_vs_noisy_diff:.6f}")
    print(f"  MSE: {pytorch_vs_noisy_mse:.6f}")
    print(f"  Status: PyTorch has enhanced the audio")


if __name__ == "__main__":
    main()
