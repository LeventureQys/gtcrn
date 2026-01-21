"""
Debug C streaming by comparing frame 0 output with Python streaming.
"""
import sys
import os
import numpy as np

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

import torch
import scipy.io.wavfile as wav
from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream

def main():
    # Load test audio
    audio_path = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00001_1_fan_noise_level1_snr-5dB_noisy.wav")
    sr, audio = wav.read(audio_path)
    audio = audio.astype(np.float32) / 32768.0

    # Load weights
    ckpt = torch.load(os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar"), map_location='cpu')
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # STFT parameters
    n_fft = 512
    hop_len = 256
    win_len = 512
    window = torch.sqrt(torch.hann_window(win_len))

    # Create streaming model
    offline_model = GTCRN()
    offline_model.load_state_dict(state_dict)
    offline_model.eval()

    stream_model = StreamGTCRN()
    convert_to_stream(stream_model, offline_model)
    stream_model.eval()

    # STFT
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    spec = torch.stft(audio_tensor, n_fft, hop_len, win_len, window, return_complex=False)
    # spec shape: (1, 257, T, 2)

    # Initialize cache (all zeros)
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Get frame 0 input
    frame0_real = spec[0, :, 0, 0].numpy()
    frame0_imag = spec[0, :, 0, 1].numpy()

    print("=" * 60)
    print("Frame 0 Analysis (all caches are zero)")
    print("=" * 60)

    print(f"\nInput spectrum frame 0:")
    print(f"  Real sum: {np.sum(frame0_real):.4f}")
    print(f"  Imag sum: {np.sum(frame0_imag):.4f}")
    print(f"  Mag sum: {np.sum(np.sqrt(frame0_real**2 + frame0_imag**2)):.4f}")

    # Process frame 0 with Python streaming
    with torch.no_grad():
        frame_input = spec[:, :, 0:1, :]  # (1, 257, 1, 2)
        output, conv_cache, tra_cache, inter_cache = stream_model(
            frame_input, conv_cache, tra_cache, inter_cache
        )

    out_real = output[0, :, 0, 0].numpy()
    out_imag = output[0, :, 0, 1].numpy()
    out_mag = np.sqrt(out_real**2 + out_imag**2)

    print(f"\nPython streaming output frame 0:")
    print(f"  Real sum: {np.sum(out_real):.4f}")
    print(f"  Imag sum: {np.sum(out_imag):.4f}")
    print(f"  Mag sum: {np.sum(out_mag):.4f}")
    print(f"  Max mag: {np.max(out_mag):.4f}")

    print(f"\nFirst 10 output spectrum values (Python):")
    print("Bin | Real       | Imag       | Mag")
    print("----+------------+------------+--------")
    for i in range(10):
        print(f"{i:3d} | {out_real[i]:10.6f} | {out_imag[i]:10.6f} | {out_mag[i]:.4f}")

    # Save for comparison
    output_dir = os.path.join(gtcrn_dir, "test_wavs/output_c")
    np.save(os.path.join(output_dir, "py_frame0_out_real.npy"), out_real)
    np.save(os.path.join(output_dir, "py_frame0_out_imag.npy"), out_imag)
    np.save(os.path.join(output_dir, "py_frame0_in_real.npy"), frame0_real)
    np.save(os.path.join(output_dir, "py_frame0_in_imag.npy"), frame0_imag)

    print(f"\nSaved to {output_dir}")

if __name__ == "__main__":
    main()
