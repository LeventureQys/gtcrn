#!/usr/bin/env python3
"""
Compare C streaming output spectrum with Python streaming output spectrum.
This test saves Python's output spectrums to a file that we can load in C to compare.
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
    ola_buffer = np.zeros(hop, dtype=np.float32)
    first_frame = True

    # Process frames and track spectrums
    num_frames = 120
    py_time_outputs = []
    py_mag_sums = []

    print(f"Processing {num_frames} frames with Python...")

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

            spec_tensor = torch.zeros(1, n_fft//2+1, 1, 2)
            spec_tensor[0, :, 0, 0] = torch.from_numpy(spec_real)
            spec_tensor[0, :, 0, 1] = torch.from_numpy(spec_imag)

            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                spec_tensor, conv_cache, tra_cache, inter_cache
            )

            out_spec_real = out_frame[0, :, 0, 0].numpy()
            out_spec_imag = out_frame[0, :, 0, 1].numpy()

            # Track magnitude sum
            mag = np.sqrt(out_spec_real**2 + out_spec_imag**2)
            py_mag_sums.append(np.sum(mag))

            # ISTFT (same as C)
            full_spec = np.zeros(n_fft, dtype=np.complex64)
            full_spec[:n_fft//2+1] = out_spec_real + 1j * out_spec_imag
            for i in range(1, n_fft // 2):
                full_spec[n_fft - i] = np.conj(full_spec[i])
            frame_time = np.fft.ifft(full_spec).real.astype(np.float32)
            istft_frame = frame_time * window

            # OLA (same as C)
            if first_frame:
                output = istft_frame[:hop].copy()
                first_frame = False
            else:
                output = ola_buffer + istft_frame[:hop]

            ola_buffer = istft_frame[hop:].copy()
            py_time_outputs.append(output)

    # Load C streaming output
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    c_stream, _ = sf.read(c_stream_path)

    # Compare frame by frame
    print("\n=== Frame-by-frame comparison ===")
    print("Frame | Py Time RMS | C Time RMS  | Time Ratio | Py Mag Sum | Expected C Ratio")
    print("-" * 85)

    # From the test_stream_debug, we know:
    # Frame 100: C mag sum = 69.57, Python mag sum = 73.34, spectrum ratio = 0.9486

    for frame_idx in range(100, 110):
        py_time = py_time_outputs[frame_idx]
        c_time = c_stream[frame_idx * hop:(frame_idx + 1) * hop]

        py_time_rms = np.sqrt(np.mean(py_time**2))
        c_time_rms = np.sqrt(np.mean(c_time**2))
        time_ratio = c_time_rms / py_time_rms if py_time_rms > 1e-8 else 0

        py_mag = py_mag_sums[frame_idx]
        # Expected: if spectrum ratio is ~0.95, time ratio should also be ~0.95
        # But we're seeing ~0.59-0.72

        print(f"{frame_idx:5d} | {py_time_rms:11.6f} | {c_time_rms:11.6f} | {time_ratio:10.4f} | {py_mag:10.2f} |")

    # Compute what the time-domain RMS SHOULD be from the spectrum
    print("\n=== Energy analysis ===")
    print("For sqrt-Hann window with 50% overlap:")
    print("  - At steady state, sum(window^2) = 1.0 over overlapping region")
    print("  - So time-domain energy = spectrum energy / N")
    print("")

    # Check the first few samples of frame 100
    print("=== Sample-level comparison for frame 100 ===")
    py_frame = py_time_outputs[100]
    c_frame = c_stream[100 * hop:101 * hop]

    print("Sample | Py Value    | C Value     | Ratio")
    print("-" * 50)
    for i in range(min(10, len(py_frame))):
        ratio = c_frame[i] / py_frame[i] if abs(py_frame[i]) > 1e-8 else 0
        print(f"{i:6d} | {py_frame[i]:11.6f} | {c_frame[i]:11.6f} | {ratio:6.3f}")

if __name__ == "__main__":
    main()
