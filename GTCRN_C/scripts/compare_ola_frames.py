#!/usr/bin/env python3
"""
Check if the OLA buffer is being accumulated correctly.
Compare C streaming output with what we'd expect from Python.
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

    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    c_stream, _ = sf.read(c_stream_path)

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

    # Process frames and compare OLA outputs
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    ola_buffer = np.zeros(hop, dtype=np.float32)
    first_frame = True
    py_outputs = []

    print("\n=== Frame-by-frame comparison (after OLA) ===")
    print("Frame | Python RMS | C RMS      | Ratio    | First 3 samples")
    print("-" * 80)

    with torch.no_grad():
        for frame_idx in range(110):
            start = frame_idx * hop
            if start >= hop:
                prev = audio[start - hop:start]
            else:
                prev = np.zeros(hop, dtype=np.float32)
            curr = audio[start:start + hop] if start + hop <= len(audio) else np.zeros(hop, dtype=np.float32)
            stft_window = np.concatenate([prev, curr])

            # STFT
            windowed = stft_window * window
            fft_out = np.fft.fft(windowed, n_fft)
            spec_real = fft_out[:n_fft//2+1].real.astype(np.float32)
            spec_imag = fft_out[:n_fft//2+1].imag.astype(np.float32)

            # Neural network
            spec_tensor = torch.zeros(1, n_fft//2+1, 1, 2)
            spec_tensor[0, :, 0, 0] = torch.from_numpy(spec_real)
            spec_tensor[0, :, 0, 1] = torch.from_numpy(spec_imag)

            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                spec_tensor, conv_cache, tra_cache, inter_cache
            )

            out_spec_real = out_frame[0, :, 0, 0].numpy()
            out_spec_imag = out_frame[0, :, 0, 1].numpy()

            # ISTFT
            full_spec = np.zeros(n_fft, dtype=np.complex64)
            full_spec[:n_fft//2+1] = out_spec_real + 1j * out_spec_imag
            for i in range(1, n_fft // 2):
                full_spec[n_fft - i] = np.conj(full_spec[i])
            frame_time = np.fft.ifft(full_spec).real.astype(np.float32)
            istft_frame = frame_time * window

            # OLA
            if first_frame:
                output = istft_frame[:hop].copy()
                first_frame = False
            else:
                output = ola_buffer + istft_frame[:hop]

            ola_buffer = istft_frame[hop:].copy()
            py_outputs.append(output)

            # Compare with C
            c_output = c_stream[frame_idx * hop:(frame_idx + 1) * hop]

            if frame_idx >= 100 and frame_idx < 110:
                py_rms = np.sqrt(np.mean(output**2))
                c_rms = np.sqrt(np.mean(c_output**2))
                ratio = c_rms / py_rms if py_rms > 1e-8 else 0
                print(f"{frame_idx:5d} | {py_rms:10.6f} | {c_rms:10.6f} | {ratio:8.4f} | py:{output[:3]}")
                print(f"      |            |            |          | c: {c_output[:3]}")

    # Overall comparison
    py_full = np.concatenate(py_outputs)
    min_len = min(len(py_full), len(c_stream))

    print(f"\n=== Overall comparison ===")
    print(f"Python RMS: {np.sqrt(np.mean(py_full[:min_len]**2)):.6f}")
    print(f"C RMS: {np.sqrt(np.mean(c_stream[:min_len]**2)):.6f}")
    print(f"Correlation: {np.corrcoef(py_full[:min_len], c_stream[:min_len])[0,1]:.6f}")


if __name__ == "__main__":
    main()
