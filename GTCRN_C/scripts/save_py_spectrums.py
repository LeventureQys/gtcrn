#!/usr/bin/env python3
"""
Debug: Compare the output spectrum (before ISTFT) between C and Python.
This script saves Python's output spectrum to binary files that we can compare
with C's output spectrum by modifying C code to dump spectrums.
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch
import struct

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream
from gtcrn import GTCRN

def main():
    # Load test audio
    test_wav = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
    audio, sr = sf.read(test_wav)
    audio = audio.astype(np.float32)

    n_fft = 512
    hop = 256
    win = 512

    # sqrt-Hann window
    window = np.zeros(win, dtype=np.float32)
    for i in range(win):
        hann = 0.5 * (1.0 - np.cos(2.0 * np.pi * i / win))
        window[i] = np.sqrt(hann)

    # Load streaming model
    print("Loading streaming model...")
    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    stft_input_buffer = np.zeros(hop, dtype=np.float32)

    # Process multiple frames and collect spectrums
    num_frames = 120
    input_specs = []
    output_specs = []

    print(f"Processing {num_frames} frames...")

    with torch.no_grad():
        for frame_idx in range(num_frames):
            start = frame_idx * hop
            if start + hop <= len(audio):
                current_frame = audio[start:start + hop]
            else:
                current_frame = np.zeros(hop, dtype=np.float32)

            stft_window = np.concatenate([stft_input_buffer, current_frame])
            stft_input_buffer = current_frame.copy()

            windowed = stft_window * window
            fft_out = np.fft.fft(windowed, n_fft)
            spec_real = fft_out[:n_fft//2+1].real.astype(np.float32)
            spec_imag = fft_out[:n_fft//2+1].imag.astype(np.float32)

            input_specs.append((spec_real.copy(), spec_imag.copy()))

            spec_tensor = torch.zeros(1, n_fft//2+1, 1, 2)
            spec_tensor[0, :, 0, 0] = torch.from_numpy(spec_real)
            spec_tensor[0, :, 0, 1] = torch.from_numpy(spec_imag)

            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                spec_tensor, conv_cache, tra_cache, inter_cache
            )

            out_spec_real = out_frame[0, :, 0, 0].numpy()
            out_spec_imag = out_frame[0, :, 0, 1].numpy()

            output_specs.append((out_spec_real.copy(), out_spec_imag.copy()))

    # Save spectrums to binary file for C comparison
    output_file = os.path.join(script_dir, "py_spectrums.bin")
    with open(output_file, 'wb') as f:
        # Write number of frames
        f.write(struct.pack('i', num_frames))
        # Write n_freq
        f.write(struct.pack('i', n_fft//2+1))

        # Write all input and output spectrums
        for i in range(num_frames):
            in_real, in_imag = input_specs[i]
            out_real, out_imag = output_specs[i]

            for val in in_real:
                f.write(struct.pack('f', val))
            for val in in_imag:
                f.write(struct.pack('f', val))
            for val in out_real:
                f.write(struct.pack('f', val))
            for val in out_imag:
                f.write(struct.pack('f', val))

    print(f"Saved Python spectrums to: {output_file}")

    # Print stats for frames 100-110
    print("\n=== Output spectrum statistics (frames 100-110) ===")
    print("Frame | Out Real Sum | Out Imag Sum | Out Mag Sum")
    print("-" * 60)
    for i in range(100, min(110, num_frames)):
        out_real, out_imag = output_specs[i]
        mag = np.sqrt(out_real**2 + out_imag**2)
        print(f"{i:5d} | {np.sum(out_real):12.4f} | {np.sum(out_imag):12.4f} | {np.sum(mag):11.4f}")

    # Also compute the mask values directly to check
    print("\n=== Check mask range (should be [-1, 1] due to tanh) ===")
    for i in [100, 105, 110]:
        if i >= num_frames:
            continue
        in_real, in_imag = input_specs[i]
        out_real, out_imag = output_specs[i]

        # Compute mask: out = mask * in  (complex multiplication)
        # mask_real = (out_real * in_real + out_imag * in_imag) / (in_real^2 + in_imag^2)
        # mask_imag = (out_imag * in_real - out_real * in_imag) / (in_real^2 + in_imag^2)
        in_mag_sq = in_real**2 + in_imag**2 + 1e-12
        mask_real = (out_real * in_real + out_imag * in_imag) / in_mag_sq
        mask_imag = (out_imag * in_real - out_real * in_imag) / in_mag_sq

        print(f"Frame {i}: mask_real range [{mask_real.min():.3f}, {mask_real.max():.3f}], "
              f"mask_imag range [{mask_imag.min():.3f}, {mask_imag.max():.3f}]")

if __name__ == "__main__":
    main()
