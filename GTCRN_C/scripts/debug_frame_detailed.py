#!/usr/bin/env python3
"""
Debug script to compare C and Python streaming outputs at the first frame.
Saves intermediate tensors for detailed comparison.
"""

import os
import sys
import struct
import numpy as np
import torch
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream
from gtcrn import GTCRN


def load_model():
    """Load the streaming model."""
    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")

    # Load offline model
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])

    # Create streaming model and convert
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    return stream_model


def detailed_forward(model, spec_frame, conv_cache, tra_cache, inter_cache, frame_idx=0):
    """Run forward pass with detailed intermediate outputs."""
    outputs = {}

    with torch.no_grad():
        spec_ref = spec_frame  # (B,F,T,2)

        spec_real = spec_frame[..., 0].permute(0,2,1)
        spec_imag = spec_frame[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)
        outputs['feat'] = feat.clone()

        feat_erb = model.erb.bm(feat)  # (B,3,T,129)
        outputs['feat_erb'] = feat_erb.clone()

        feat_sfe = model.sfe(feat_erb)  # (B,9,T,129)
        outputs['feat_sfe'] = feat_sfe.clone()

        # Use original model forward to correctly handle cache
        # Just run the model and save outputs
        spec_enh, conv_cache, tra_cache, inter_cache = model(
            spec_frame, conv_cache, tra_cache, inter_cache)

        outputs['output'] = spec_enh.clone()

    return outputs, conv_cache, tra_cache, inter_cache


def main():
    print("=== Detailed Stream Debug ===\n")

    # Load test audio
    test_wav = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
    if not os.path.exists(test_wav):
        print(f"Test file not found: {test_wav}")
        return

    audio, sr = sf.read(test_wav)
    print(f"Loaded audio: {len(audio)} samples, sr={sr}")

    # Load streaming model
    print("\nLoading streaming model...")
    model = load_model()

    # STFT
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))
    audio_tensor = torch.from_numpy(audio).float()
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False)
    spec = spec.unsqueeze(0)  # (1, freq, time, 2)
    print(f"Spec shape: {spec.shape}")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process first 10 frames and save outputs
    output_dir = os.path.join(gtcrn_dir, "test_wavs/output_debug")
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in range(10):
        frame = spec[:, :, frame_idx:frame_idx+1, :]  # (1, 257, 1, 2)

        outputs, conv_cache, tra_cache, inter_cache = detailed_forward(
            model, frame, conv_cache, tra_cache, inter_cache, frame_idx)

        if frame_idx < 5:
            print(f"\n=== Frame {frame_idx} ===")
            for name, tensor in outputs.items():
                print(f"  {name}: shape={list(tensor.shape)}, sum={tensor.sum():.6f}, "
                      f"min={tensor.min():.6f}, max={tensor.max():.6f}")

            # Save tensors for frame 0
            if frame_idx == 0:
                for name, tensor in outputs.items():
                    np.save(os.path.join(output_dir, f"py_frame0_{name}.npy"),
                            tensor.numpy())
                print(f"\nSaved frame 0 tensors to {output_dir}/py_frame0_*.npy")


if __name__ == "__main__":
    main()
