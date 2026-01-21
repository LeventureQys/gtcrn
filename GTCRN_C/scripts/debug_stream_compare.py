#!/usr/bin/env python3
"""
Debug script to compare C streaming with Python streaming frame by frame.
Outputs intermediate tensors for comparison.
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

def process_audio_streaming(model, audio, sr=16000):
    """Process audio using streaming model."""
    # STFT parameters
    n_fft = 512
    hop_length = 256
    win_length = 512

    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio
    audio_tensor = torch.from_numpy(audio).float()

    # STFT
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False)
    # spec: (freq, time, 2)
    spec = spec.unsqueeze(0)  # (1, freq, time, 2)

    num_frames = spec.shape[2]
    print(f"Total frames: {num_frames}")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)   # encoder/decoder conv cache
    tra_cache = torch.zeros(2, 3, 1, 1, 16)      # encoder/decoder TRA cache
    inter_cache = torch.zeros(2, 1, 33, 16)      # DPGRNN inter-frame cache

    outputs = []
    debug_outputs = []

    with torch.no_grad():
        for i in range(num_frames):
            frame = spec[:, :, i:i+1, :]  # (1, 257, 1, 2)

            # Run streaming model
            out_frame, conv_cache, tra_cache, inter_cache = model(
                frame, conv_cache, tra_cache, inter_cache
            )

            outputs.append(out_frame)

            # Save debug info for first few frames
            if i < 5:
                debug_outputs.append({
                    'frame': i,
                    'input': frame.clone(),
                    'output': out_frame.clone(),
                    'conv_cache': conv_cache.clone(),
                    'tra_cache': tra_cache.clone(),
                    'inter_cache': inter_cache.clone()
                })
                print(f"Frame {i}: input sum={frame.sum():.6f}, output sum={out_frame.sum():.6f}")

    # Concatenate outputs
    output_spec = torch.cat(outputs, dim=2)

    # ISTFT
    output_spec_complex = torch.complex(output_spec[0, :, :, 0], output_spec[0, :, :, 1])
    output_audio = torch.istft(output_spec_complex, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length, window=window, length=len(audio))

    return output_audio.numpy(), debug_outputs

def main():
    print("=== Debug Stream Compare ===\n")

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

    # Process with Python streaming
    print("\nProcessing with Python streaming...")
    py_output, debug_info = process_audio_streaming(model, audio, sr)

    print(f"\nPython output: {len(py_output)} samples")
    print(f"  RMS: {np.sqrt(np.mean(py_output**2)):.6f}")
    print(f"  Range: [{py_output.min():.4f}, {py_output.max():.4f}]")

    # Save Python streaming output
    output_dir = os.path.join(gtcrn_dir, "test_wavs/output_debug")
    os.makedirs(output_dir, exist_ok=True)

    py_stream_path = os.path.join(output_dir, "enhanced_py_stream.wav")
    sf.write(py_stream_path, py_output, sr)
    print(f"\nSaved Python stream output to: {py_stream_path}")

    # Load C streaming output if exists
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    if os.path.exists(c_stream_path):
        c_output, _ = sf.read(c_stream_path)
        print(f"\nC streaming output: {len(c_output)} samples")
        print(f"  RMS: {np.sqrt(np.mean(c_output**2)):.6f}")
        print(f"  Range: [{c_output.min():.4f}, {c_output.max():.4f}]")

        # Compare
        min_len = min(len(py_output), len(c_output))
        correlation = np.corrcoef(py_output[:min_len], c_output[:min_len])[0, 1]
        print(f"\nCorrelation: {correlation:.6f}")

        # Compare different segments
        segments = [(0, 5000), (5000, 10000), (10000, 20000), (50000, 55000)]
        for start, end in segments:
            if end <= min_len:
                corr = np.corrcoef(py_output[start:end], c_output[start:end])[0, 1]
                print(f"  Segment [{start}:{end}]: {corr:.6f}")
    else:
        print(f"\nC streaming output not found: {c_stream_path}")
        print("Run the C streaming demo first to generate this file.")

if __name__ == "__main__":
    main()
