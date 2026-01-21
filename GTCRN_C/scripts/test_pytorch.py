#!/usr/bin/env python3
"""
GTCRN Test Script - Compare PyTorch output for testing

This script runs the PyTorch GTCRN model on a test file and saves the output.
It can be used to compare with the C implementation.
"""

import os
import sys
import time
import numpy as np
import torch
import soundfile as sf

# Add gtcrn directory for imports
gtcrn_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, gtcrn_dir)
from gtcrn import GTCRN


def process_audio(model, audio, sample_rate=16000):
    """Process audio through GTCRN model."""
    assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate}Hz"

    # Convert to tensor
    x = torch.from_numpy(audio).float()

    # STFT
    window = torch.hann_window(512).pow(0.5)
    spec = torch.stft(x, 512, 256, 512, window, return_complex=False)
    spec = spec.unsqueeze(0)  # Add batch dimension: (1, F, T, 2)

    # Model inference
    with torch.no_grad():
        start_time = time.perf_counter()
        enhanced_spec = model(spec)
        inference_time = time.perf_counter() - start_time

    # ISTFT
    enhanced_spec = enhanced_spec.squeeze(0)  # Remove batch: (F, T, 2)
    enhanced_complex = torch.complex(enhanced_spec[:, :, 0], enhanced_spec[:, :, 1])
    enhanced_audio = torch.istft(enhanced_complex, 512, 256, 512, window)

    return enhanced_audio.numpy(), inference_time


def main():
    # Paths - using paths relative to gtcrn directory
    model_path = "../../checkpoints/model_trained_on_dns3.tar"
    input_path = "../../test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav"
    output_path = "../../test_wavs/output_16k/enhanced_pytorch.wav"

    print("=" * 50)
    print("GTCRN PyTorch Inference Test")
    print("=" * 50)

    # Load model
    print(f"\n[1/4] Loading model from {model_path}...")
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,} ({total_params * 4 / 1024:.1f} KB)")

    # Load audio
    print(f"\n[2/4] Loading audio from {input_path}...")
    audio, sr = sf.read(input_path, dtype='float32')
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio) / sr:.2f} seconds ({len(audio)} samples)")

    # Process
    print(f"\n[3/4] Processing audio...")
    enhanced_audio, inference_time = process_audio(model, audio, sr)

    audio_duration = len(audio) / sr
    rtf = inference_time / audio_duration

    print(f"  Inference time: {inference_time * 1000:.2f} ms")
    print(f"  Audio duration: {audio_duration * 1000:.2f} ms")
    print(f"  Real-time factor (RTF): {rtf:.4f}")
    if rtf < 1.0:
        print(f"  Status: REAL-TIME ({1.0/rtf:.1f}x faster than real-time)")
    else:
        print(f"  Status: NOT REAL-TIME ({rtf:.1f}x slower)")

    # Save output
    print(f"\n[4/4] Saving enhanced audio to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, enhanced_audio, sr)
    print(f"  Output saved successfully!")

    # Statistics
    print("\n" + "=" * 50)
    print("Audio Statistics")
    print("=" * 50)
    print(f"  Input  - min: {audio.min():.4f}, max: {audio.max():.4f}, std: {audio.std():.4f}")
    print(f"  Output - min: {enhanced_audio.min():.4f}, max: {enhanced_audio.max():.4f}, std: {enhanced_audio.std():.4f}")

    return enhanced_audio


if __name__ == "__main__":
    main()
