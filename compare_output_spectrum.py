"""
Compare output spectrum (before ISTFT) between C and Python.
"""
import os
import sys
import numpy as np
import torch
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream

def main():
    print("=== Compare Output Spectrum (before ISTFT) ===\n")

    # Load model
    model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # Load noisy audio
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio (same as C)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # STFT with center=False (same as C)
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False,
                      center=False)
    spec = spec.unsqueeze(0)  # (1, 257, T, 2)
    n_frames = spec.shape[2]
    print(f"Number of frames: {n_frames}")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process specific frames and print output spectrum
    print("\n=== Frame-by-frame output spectrum comparison ===")
    print("(Compare with C debug output 'Spec input' at each frame)\n")

    # C Frame 6 debug shows:
    # [C Stream] Spec input:
    #     spec_real first 10: 0.135563 0.137168 -0.273219 0.159885 -0.130444 ...
    #     spec_imag first 10: 0.000000 -0.019988 -0.173454 0.057402 0.070443 ...
    # And:
    # [C Stream] Mask after ERB expansion (2, 257):
    #     mask_real sum: 8.864581
    #     mask_imag sum: 0.081881

    with torch.no_grad():
        for frame_idx in range(10):
            frame = spec[:, :, frame_idx:frame_idx+1, :]

            # Get output from stream model
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )

            # out_frame: (1, 257, 1, 2)
            out_real = out_frame[0, :, 0, 0].numpy()
            out_imag = out_frame[0, :, 0, 1].numpy()

            # Also get input spectrum for verification
            in_real = frame[0, :, 0, 0].numpy()
            in_imag = frame[0, :, 0, 1].numpy()

            # Print comparison
            c_frame = frame_idx + 1  # C uses 1-indexed frame counter
            print(f"Frame {frame_idx} (C Frame {c_frame}):")
            print(f"  Input:  real_sum={in_real.sum():.4f}, imag_sum={in_imag.sum():.4f}")
            print(f"  Output: real_sum={out_real.sum():.4f}, imag_sum={out_imag.sum():.4f}")
            print(f"  Out first 5 real: {out_real[:5]}")

    print("\n=== Expected from C Frame 6 (frame_idx=5) ===")
    print("Mask after ERB expansion: real_sum=8.864581, imag_sum=0.081881")
    print("Applied to spec: spec_real[0:10] = 0.135563 0.137168 -0.273219 ...")
    print("So output should be mask * spec via complex multiplication")

if __name__ == "__main__":
    main()
