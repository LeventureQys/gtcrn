#!/usr/bin/env python3
"""
Compare intermediate neural network outputs between C streaming and Python streaming.
This will identify which layer is causing the energy loss.
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

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Warm up for 100 frames
    stft_input_buffer = np.zeros(hop, dtype=np.float32)

    print("Warming up (100 frames)...")
    with torch.no_grad():
        for frame_idx in range(100):
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

            _, conv_cache, tra_cache, inter_cache = stream_model(
                spec_tensor, conv_cache, tra_cache, inter_cache
            )

    # Process frame 100 and inspect intermediate outputs
    frame_idx = 100
    start = frame_idx * hop
    current_frame = audio[start:start + hop]

    stft_window = np.concatenate([stft_input_buffer, current_frame])

    windowed = stft_window * window
    fft_out = np.fft.fft(windowed, n_fft)
    spec_real = fft_out[:n_fft//2+1].real.astype(np.float32)
    spec_imag = fft_out[:n_fft//2+1].imag.astype(np.float32)

    print(f"\n=== Frame {frame_idx} input spectrum ===")
    print(f"Input spec_real sum: {np.sum(spec_real):.6f}")
    print(f"Input spec_imag sum: {np.sum(spec_imag):.6f}")
    print(f"Input magnitude sum: {np.sum(np.sqrt(spec_real**2 + spec_imag**2)):.6f}")

    # Save for C comparison
    np.save(os.path.join(script_dir, "frame100_spec_real.npy"), spec_real)
    np.save(os.path.join(script_dir, "frame100_spec_imag.npy"), spec_imag)

    spec_tensor = torch.zeros(1, n_fft//2+1, 1, 2)
    spec_tensor[0, :, 0, 0] = torch.from_numpy(spec_real)
    spec_tensor[0, :, 0, 1] = torch.from_numpy(spec_imag)

    # Process with instrumented output
    with torch.no_grad():
        out_frame, _, _, _ = stream_model(
            spec_tensor, conv_cache, tra_cache, inter_cache
        )

    out_spec_real = out_frame[0, :, 0, 0].numpy()
    out_spec_imag = out_frame[0, :, 0, 1].numpy()

    print(f"\n=== Frame {frame_idx} output spectrum (Python) ===")
    print(f"Output spec_real sum: {np.sum(out_spec_real):.6f}")
    print(f"Output spec_imag sum: {np.sum(out_spec_imag):.6f}")
    print(f"Output magnitude sum: {np.sum(np.sqrt(out_spec_real**2 + out_spec_imag**2)):.6f}")

    # Save for C comparison
    np.save(os.path.join(script_dir, "frame100_out_real.npy"), out_spec_real)
    np.save(os.path.join(script_dir, "frame100_out_imag.npy"), out_spec_imag)

    # Let's also trace through some intermediate values by looking at model internals
    print("\n=== Model component analysis ===")

    # Get ERB transformation parameters
    erb_weight = stream_model.erb.erb.weight.data.numpy()
    erb_bs_weight = stream_model.erb.erb_bs.weight.data.numpy()

    print(f"ERB weight shape: {erb_weight.shape}")
    print(f"ERB_bs weight shape: {erb_bs_weight.shape}")
    print(f"ERB weight sum: {np.sum(erb_weight):.6f}")
    print(f"ERB_bs weight sum: {np.sum(erb_bs_weight):.6f}")

    # Calculate mask manually to verify
    # First, compute input features (magnitude, real, imag)
    mag = np.sqrt(spec_real**2 + spec_imag**2)
    features = np.stack([mag, spec_real, spec_imag], axis=0)  # (3, 257)

    # ERB compression
    features_torch = torch.from_numpy(features).unsqueeze(0).unsqueeze(-1)  # (1, 3, 257, 1)
    with torch.no_grad():
        erb_out = stream_model.erb.erb(features_torch)  # (1, 3, 129, 1)

    print(f"\nERB output shape: {erb_out.shape}")
    print(f"ERB output mean: {erb_out.mean().item():.6f}")
    print(f"ERB output std: {erb_out.std().item():.6f}")

    # Check C output for comparison
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    c_stream, _ = sf.read(c_stream_path)

    # Frame 100+1 from C (1 frame offset)
    c_frame = c_stream[(frame_idx + 1) * hop:(frame_idx + 2) * hop]

    # Python output
    full_spec = np.zeros(n_fft, dtype=np.complex64)
    full_spec[:n_fft//2+1] = out_spec_real + 1j * out_spec_imag
    for i in range(1, n_fft // 2):
        full_spec[n_fft - i] = np.conj(full_spec[i])
    frame_time = np.fft.ifft(full_spec).real.astype(np.float32)

    print(f"\n=== Frame {frame_idx} time domain comparison ===")
    print(f"Python IFFT output RMS (full 512): {np.sqrt(np.mean(frame_time**2)):.6f}")
    print(f"C frame {frame_idx+1} RMS: {np.sqrt(np.mean(c_frame**2)):.6f}")
    print(f"Ratio: {np.sqrt(np.mean(c_frame**2)) / np.sqrt(np.mean(frame_time**2)):.4f}")

    print("\n=== Files saved for C comparison ===")
    print(f"  frame100_spec_real.npy - Input spectrum real part")
    print(f"  frame100_spec_imag.npy - Input spectrum imaginary part")
    print(f"  frame100_out_real.npy - Output spectrum real part")
    print(f"  frame100_out_imag.npy - Output spectrum imaginary part")

if __name__ == "__main__":
    main()
