#!/usr/bin/env python3
"""
Process audio using Python streaming with center=False to match C streaming.
"""

import os
import sys
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

def main():
    print("=== Python Streaming (center=False) ===\n")

    # Load test audio
    test_wav = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
    audio, sr = sf.read(test_wav)
    print(f"Loaded audio: {len(audio)} samples, sr={sr}")

    # Load streaming model
    print("Loading streaming model...")
    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # STFT parameters
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio for center=False: need to start from sample 0 with full window
    # With center=False, first frame uses samples [0:512], etc.
    audio_tensor = torch.from_numpy(audio).float()

    # STFT with center=False
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False,
                      center=False)
    spec = spec.unsqueeze(0)  # (1, freq, time, 2)
    num_frames = spec.shape[2]
    print(f"Total frames (center=False): {num_frames}")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    outputs = []
    with torch.no_grad():
        for i in range(num_frames):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            outputs.append(out_frame)

    # Concatenate and ISTFT manually (overlap-add)
    output_spec = torch.cat(outputs, dim=2)
    output_spec_complex = torch.complex(output_spec[0, :, :, 0], output_spec[0, :, :, 1])

    # Manual ISTFT with overlap-add
    expected_length = (num_frames - 1) * hop_length + win_length
    py_output = np.zeros(expected_length, dtype=np.float32)
    window_sum = np.zeros(expected_length, dtype=np.float32)

    window_np = window.numpy()

    for i in range(num_frames):
        # IFFT this frame
        frame_spec = output_spec_complex[:, i]
        # Reconstruct full spectrum (conjugate symmetric)
        full_spec = torch.zeros(n_fft, dtype=torch.complex64)
        full_spec[:n_fft//2+1] = frame_spec
        full_spec[n_fft//2+1:] = torch.conj(torch.flip(frame_spec[1:-1], [0]))
        frame_time = torch.fft.ifft(full_spec).real.numpy()

        # Apply synthesis window
        frame_windowed = frame_time * window_np

        # Overlap-add
        start = i * hop_length
        py_output[start:start+win_length] += frame_windowed
        window_sum[start:start+win_length] += window_np ** 2

    # Normalize by window sum
    eps = 1e-8
    py_output = py_output / np.maximum(window_sum, eps)
    print(f"Python output: {len(py_output)} samples")
    print(f"  RMS: {np.sqrt(np.mean(py_output**2)):.6f}")

    # Save output
    output_dir = os.path.join(gtcrn_dir, "test_wavs/output_debug")
    os.makedirs(output_dir, exist_ok=True)
    py_stream_path = os.path.join(output_dir, "enhanced_py_stream_nocenter.wav")
    sf.write(py_stream_path, py_output, sr)
    print(f"Saved to: {py_stream_path}")

    # Compare with C streaming
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    if os.path.exists(c_stream_path):
        c_stream, _ = sf.read(c_stream_path)
        min_len = min(len(py_output), len(c_stream))
        correlation = np.corrcoef(py_output[:min_len], c_stream[:min_len])[0, 1]
        print(f"\nCorrelation with C streaming: {correlation:.6f}")

        print("\n=== First 20 samples ===")
        print("Sample | Python      | C           | Diff")
        print("-" * 50)
        for i in range(min(20, min_len)):
            print(f"{i:6d} | {py_output[i]:11.6f} | {c_stream[i]:11.6f} | {py_output[i]-c_stream[i]:11.6f}")


if __name__ == "__main__":
    main()
