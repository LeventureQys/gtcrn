#!/usr/bin/env python3
"""
Direct comparison: C streaming output vs Python simulation of exact C algorithm.
"""

import os
import sys
import numpy as np
import soundfile as sf
import torch

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

    # Load C streaming output
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    c_stream, _ = sf.read(c_stream_path)

    # Parameters
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

    # Simulate C streaming pipeline EXACTLY
    print("Simulating C streaming pipeline...")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    stft_input_buffer = np.zeros(hop, dtype=np.float32)
    ola_buffer = np.zeros(hop, dtype=np.float32)
    first_frame = True

    py_sim_output = []
    window_torch = torch.from_numpy(window)

    # Process frame by frame
    num_frames = (len(audio) - win) // hop + 1
    print(f"Processing {num_frames} frames...")

    with torch.no_grad():
        for frame_idx in range(num_frames + 1):  # +1 to account for C's extra frame
            # Get current audio frame
            start = frame_idx * hop
            if start + hop <= len(audio):
                current_frame = audio[start:start + hop]
            else:
                current_frame = np.zeros(hop, dtype=np.float32)

            # Build 512-sample STFT window (C style)
            stft_window = np.concatenate([stft_input_buffer, current_frame])
            stft_input_buffer = current_frame.copy()

            # STFT
            windowed = stft_window * window
            fft_out = np.fft.fft(windowed, n_fft)
            spec_real = fft_out[:n_fft//2+1].real.astype(np.float32)
            spec_imag = fft_out[:n_fft//2+1].imag.astype(np.float32)

            # Convert to tensor for model (1, freq, 1, 2)
            spec_tensor = torch.zeros(1, n_fft//2+1, 1, 2)
            spec_tensor[0, :, 0, 0] = torch.from_numpy(spec_real)
            spec_tensor[0, :, 0, 1] = torch.from_numpy(spec_imag)

            # Neural network processing
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                spec_tensor, conv_cache, tra_cache, inter_cache
            )

            # Get output spectrum
            out_spec_real = out_frame[0, :, 0, 0].numpy()
            out_spec_imag = out_frame[0, :, 0, 1].numpy()

            # ISTFT (C style)
            full_spec = np.zeros(n_fft, dtype=np.complex64)
            full_spec[:n_fft//2+1] = out_spec_real + 1j * out_spec_imag
            for i in range(1, n_fft // 2):
                full_spec[n_fft - i] = np.conj(full_spec[i])

            frame_time = np.fft.ifft(full_spec).real.astype(np.float32)
            istft_frame = frame_time * window

            # OLA (C style)
            if first_frame:
                output = istft_frame[:hop].copy()
                first_frame = False
            else:
                output = ola_buffer + istft_frame[:hop]

            ola_buffer = istft_frame[hop:].copy()
            py_sim_output.append(output)

    py_sim_output = np.concatenate(py_sim_output)
    print(f"Python simulation output length: {len(py_sim_output)}")
    print(f"C streaming output length: {len(c_stream)}")

    # Compare
    print("\n=== Comparison: Python simulation vs C streaming ===")

    # They should match exactly (up to floating point precision)
    min_len = min(len(py_sim_output), len(c_stream))

    # Overall comparison
    corr = np.corrcoef(py_sim_output[:min_len], c_stream[:min_len])[0, 1]
    print(f"Correlation: {corr:.6f}")

    py_rms = np.sqrt(np.mean(py_sim_output[:min_len]**2))
    c_rms = np.sqrt(np.mean(c_stream[:min_len]**2))
    print(f"Python sim RMS: {py_rms:.6f}")
    print(f"C streaming RMS: {c_rms:.6f}")
    print(f"Ratio (C/Py): {c_rms / py_rms:.4f}")

    # Sample comparison
    print("\n=== First 20 samples ===")
    print("Sample | Python sim  | C stream    | Diff")
    print("-" * 55)
    for i in range(min(20, min_len)):
        print(f"{i:6d} | {py_sim_output[i]:11.6f} | {c_stream[i]:11.6f} | {py_sim_output[i]-c_stream[i]:11.6f}")

    # After warmup
    print("\n=== Samples 25600-25620 (after warmup) ===")
    start = 25600
    for i in range(start, min(start + 20, min_len)):
        diff = py_sim_output[i] - c_stream[i]
        print(f"{i:6d} | {py_sim_output[i]:11.6f} | {c_stream[i]:11.6f} | {diff:11.6f}")

    # Save Python simulation for further comparison
    output_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_sim_c.wav")
    sf.write(output_path, py_sim_output, sr)
    print(f"\nSaved Python simulation to: {output_path}")


if __name__ == "__main__":
    main()
