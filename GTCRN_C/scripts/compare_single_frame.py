#!/usr/bin/env python3
"""
Compare single frame neural network output between C streaming and Python.
This creates a test input and compares the output spectrum.
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

    # Process a specific frame to compare
    frame_idx = 100  # After warmup

    # Build STFT input (C style)
    start = frame_idx * hop
    prev_frame = audio[start - hop:start]
    curr_frame = audio[start:start + hop]
    stft_window = np.concatenate([prev_frame, curr_frame])

    # STFT
    windowed = stft_window * window
    fft_out = np.fft.fft(windowed, n_fft)
    spec_real = fft_out[:n_fft//2+1].real.astype(np.float32)
    spec_imag = fft_out[:n_fft//2+1].imag.astype(np.float32)

    print(f"\n=== Frame {frame_idx} STFT spectrum ===")
    print(f"Spec real sum: {np.sum(spec_real):.6f}")
    print(f"Spec imag sum: {np.sum(spec_imag):.6f}")
    print(f"Spec magnitude sum: {np.sum(np.sqrt(spec_real**2 + spec_imag**2)):.6f}")

    # Initialize caches for frame 0..99
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    stft_input_buffer = np.zeros(hop, dtype=np.float32)

    # Process frames 0..99 to warm up
    print("\nWarming up caches (frames 0-99)...")
    with torch.no_grad():
        for fi in range(frame_idx):
            start_fi = fi * hop
            if start_fi >= hop:
                prev_fi = audio[start_fi - hop:start_fi]
            else:
                prev_fi = np.zeros(hop, dtype=np.float32)
            curr_fi = audio[start_fi:start_fi + hop]
            stft_window_fi = np.concatenate([prev_fi, curr_fi])

            windowed_fi = stft_window_fi * window
            fft_out_fi = np.fft.fft(windowed_fi, n_fft)
            spec_real_fi = fft_out_fi[:n_fft//2+1].real.astype(np.float32)
            spec_imag_fi = fft_out_fi[:n_fft//2+1].imag.astype(np.float32)

            spec_tensor = torch.zeros(1, n_fft//2+1, 1, 2)
            spec_tensor[0, :, 0, 0] = torch.from_numpy(spec_real_fi)
            spec_tensor[0, :, 0, 1] = torch.from_numpy(spec_imag_fi)

            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                spec_tensor, conv_cache, tra_cache, inter_cache
            )

    # Now process frame 100
    print(f"\nProcessing frame {frame_idx}...")
    spec_tensor = torch.zeros(1, n_fft//2+1, 1, 2)
    spec_tensor[0, :, 0, 0] = torch.from_numpy(spec_real)
    spec_tensor[0, :, 0, 1] = torch.from_numpy(spec_imag)

    with torch.no_grad():
        out_frame, _, _, _ = stream_model(
            spec_tensor, conv_cache, tra_cache, inter_cache
        )

    out_spec_real = out_frame[0, :, 0, 0].numpy()
    out_spec_imag = out_frame[0, :, 0, 1].numpy()

    print(f"\n=== Frame {frame_idx} output spectrum (Python) ===")
    print(f"Out real sum: {np.sum(out_spec_real):.6f}")
    print(f"Out imag sum: {np.sum(out_spec_imag):.6f}")
    print(f"Out magnitude sum: {np.sum(np.sqrt(out_spec_real**2 + out_spec_imag**2)):.6f}")
    print(f"Out real[0:5]: {out_spec_real[:5]}")
    print(f"Out imag[0:5]: {out_spec_imag[:5]}")

    # Save these values for comparison with C
    np.save(os.path.join(project_dir, "scripts/py_out_spec_real_100.npy"), out_spec_real)
    np.save(os.path.join(project_dir, "scripts/py_out_spec_imag_100.npy"), out_spec_imag)
    print(f"\nSaved Python output spectrum to npy files")

    # Now let's check what the C output should look like
    # Load C streaming output and extract frame 100
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    c_stream, _ = sf.read(c_stream_path)

    # Frame 100 output samples
    c_frame_100 = c_stream[(frame_idx + 1) * hop:(frame_idx + 2) * hop]  # +1 for offset
    print(f"\n=== C streaming frame {frame_idx} output (time domain) ===")
    print(f"RMS: {np.sqrt(np.mean(c_frame_100**2)):.6f}")
    print(f"First 5 samples: {c_frame_100[:5]}")

    # Python ISTFT for comparison
    full_spec = np.zeros(n_fft, dtype=np.complex64)
    full_spec[:n_fft//2+1] = out_spec_real + 1j * out_spec_imag
    for i in range(1, n_fft // 2):
        full_spec[n_fft - i] = np.conj(full_spec[i])
    frame_time = np.fft.ifft(full_spec).real.astype(np.float32)
    istft_frame = frame_time * window

    print(f"\n=== Python ISTFT frame {frame_idx} (before OLA) ===")
    print(f"First 5 samples: {istft_frame[:5]}")
    print(f"Samples 256-261: {istft_frame[256:261]}")
    print(f"RMS of first half: {np.sqrt(np.mean(istft_frame[:hop]**2)):.6f}")


if __name__ == "__main__":
    main()
