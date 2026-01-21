"""
Debug C streaming by comparing frame-by-frame output with Python streaming.
"""
import sys
import os

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

import torch
import numpy as np
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

    # Initialize cache
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process first 10 frames
    print("=" * 60)
    print("Frame-by-frame comparison (first 10 frames)")
    print("=" * 60)

    stream_outputs = []
    with torch.no_grad():
        for i in range(10):
            # Extract single frame: (1, 257, 1, 2)
            frame_input = spec[:, :, i:i+1, :]

            # Process
            output, conv_cache, tra_cache, inter_cache = stream_model(
                frame_input, conv_cache, tra_cache, inter_cache
            )

            # output: (1, 257, 1, 2)
            stream_outputs.append(output)

            # Get stats
            out_real = output[0, :, 0, 0].numpy()
            out_imag = output[0, :, 0, 1].numpy()
            out_mag = np.sqrt(out_real**2 + out_imag**2)

            print(f"Frame {i}: real_sum={np.sum(out_real):.4f}, imag_sum={np.sum(out_imag):.4f}, "
                  f"mag_sum={np.sum(out_mag):.4f}, max={np.max(out_mag):.4f}")

    # Save first 10 frames output for C comparison
    output_dir = os.path.join(gtcrn_dir, "test_wavs/output_c")

    # Stack and save
    stream_output = torch.cat(stream_outputs, dim=2)  # (1, 257, 10, 2)
    np.save(os.path.join(output_dir, "py_stream_first10_real.npy"),
            stream_output[0, :, :, 0].numpy())
    np.save(os.path.join(output_dir, "py_stream_first10_imag.npy"),
            stream_output[0, :, :, 1].numpy())

    print(f"\nSaved Python streaming first 10 frames to {output_dir}")

    # Also save input frames for C to use
    for i in range(10):
        in_real = spec[0, :, i, 0].numpy()
        in_imag = spec[0, :, i, 1].numpy()
        np.save(os.path.join(output_dir, f"py_input_frame{i}_real.npy"), in_real)
        np.save(os.path.join(output_dir, f"py_input_frame{i}_imag.npy"), in_imag)

    print(f"Saved Python input frames 0-9 to {output_dir}")

if __name__ == "__main__":
    main()
