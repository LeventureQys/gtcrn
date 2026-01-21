#!/usr/bin/env python3
"""
Compare C streaming with Python streaming at the spectrum level.
"""

import os
import numpy as np
import soundfile as sf
import torch
import sys

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

    # Load C outputs
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    c_complete_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c.wav")

    c_stream, _ = sf.read(c_stream_path)
    c_complete, _ = sf.read(c_complete_path)

    # Do Python streaming processing
    print("Loading streaming model...")
    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    audio_tensor = torch.from_numpy(audio).float()

    # STFT with center=False (matching C)
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False,
                      center=False)
    spec = spec.unsqueeze(0)  # (1, freq, time, 2)
    num_frames = spec.shape[2]
    print(f"Total frames: {num_frames}")

    # Process through model frame by frame
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    output_specs = []
    with torch.no_grad():
        for i in range(num_frames):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            output_specs.append(out_frame)

    # Stack output spectrums
    output_spec = torch.cat(output_specs, dim=2)  # (1, freq, time, 2)
    print(f"Output spec shape: {output_spec.shape}")

    # Now do streaming ISTFT matching C behavior
    # C does: for each frame:
    #   1. ISTFT to get 512 samples (windowed)
    #   2. OLA: output[i] = ola_buffer[i] + istft_frame[i] for i in [0, 256)
    #   3. Save istft_frame[256:512] to ola_buffer

    output_spec_np = output_spec[0].numpy()  # (freq, time, 2)
    n_freqs = n_fft // 2 + 1
    window_np = window.numpy()

    py_stream_ola = []
    ola_buffer = np.zeros(hop_length, dtype=np.float32)
    first_frame = True

    for t in range(num_frames):
        # Get spectrum for this frame
        spec_real = output_spec_np[:, t, 0]
        spec_imag = output_spec_np[:, t, 1]

        # Reconstruct full spectrum (conjugate symmetric)
        full_spec = np.zeros(n_fft, dtype=np.complex64)
        full_spec[:n_freqs] = spec_real + 1j * spec_imag
        full_spec[n_freqs:] = np.conj(full_spec[1:n_fft//2][::-1])

        # IFFT
        frame_time = np.fft.ifft(full_spec).real.astype(np.float32)

        # Apply synthesis window
        istft_frame = frame_time * window_np

        # OLA (matching C)
        if first_frame:
            output = istft_frame[:hop_length].copy()
            first_frame = False
        else:
            output = ola_buffer + istft_frame[:hop_length]

        ola_buffer = istft_frame[hop_length:].copy()
        py_stream_ola.append(output)

    py_stream_ola = np.concatenate(py_stream_ola)
    print(f"Python stream OLA output length: {len(py_stream_ola)}")

    # Compare Python stream OLA with C stream
    hop = hop_length
    start = 100 * hop
    end = 200 * hop

    # C stream is shifted by 1 frame relative to Python
    c_seg = c_stream[start+hop:end+hop]
    py_seg = py_stream_ola[start:end]

    min_len = min(len(c_seg), len(py_seg))
    c_seg = c_seg[:min_len]
    py_seg = py_seg[:min_len]

    corr = np.corrcoef(c_seg, py_seg)[0, 1]
    print(f"\n=== C stream vs Python stream OLA (with 1-frame shift) ===")
    print(f"Correlation: {corr:.6f}")

    c_rms = np.sqrt(np.mean(c_seg**2))
    py_rms = np.sqrt(np.mean(py_seg**2))
    print(f"C RMS:      {c_rms:.6f}")
    print(f"Python RMS: {py_rms:.6f}")
    print(f"Ratio:      {c_rms/py_rms:.4f}")

    # Also compare C stream with C complete (shifted)
    c_complete_seg = c_complete[start:end]
    c_stream_seg = c_stream[start+hop:end+hop]
    min_len2 = min(len(c_complete_seg), len(c_stream_seg))

    corr2 = np.corrcoef(c_complete_seg[:min_len2], c_stream_seg[:min_len2])[0, 1]
    print(f"\n=== C complete vs C stream (with 1-frame shift) ===")
    print(f"Correlation: {corr2:.6f}")
    print(f"C complete RMS: {np.sqrt(np.mean(c_complete_seg[:min_len2]**2)):.6f}")
    print(f"C stream RMS:   {np.sqrt(np.mean(c_stream_seg[:min_len2]**2)):.6f}")
    print(f"Ratio: {np.sqrt(np.mean(c_stream_seg[:min_len2]**2))/np.sqrt(np.mean(c_complete_seg[:min_len2]**2)):.4f}")

    # Check if the model outputs are similar
    print(f"\n=== Checking Python model output spectrum ===")
    # Compare spec magnitudes
    py_mag = np.sqrt(output_spec_np[:, 100, 0]**2 + output_spec_np[:, 100, 1]**2)
    print(f"Python output spec frame 100 energy: {np.sum(py_mag**2):.4f}")


if __name__ == "__main__":
    main()
