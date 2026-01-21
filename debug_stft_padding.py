"""
Debug torch.stft padding behavior.
"""
import os
import sys
import numpy as np
import torch
import soundfile as sf

def main():
    print("=== Debug torch.stft padding ===\n")

    gtcrn_dir = os.path.dirname(os.path.abspath(__file__))

    # Load noisy audio
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Test 1: torch.stft with default padding (center=True)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    spec_center_true = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=window, return_complex=False,
                                   center=True)
    print(f"center=True (default): shape = {spec_center_true.shape}")
    print(f"  Frame 0: real_sum={spec_center_true[:, 0, 0].sum():.4f}, imag_sum={spec_center_true[:, 0, 1].sum():.4f}")
    print(f"  Frame 1: real_sum={spec_center_true[:, 1, 0].sum():.4f}, imag_sum={spec_center_true[:, 1, 1].sum():.4f}")
    print(f"  Frame 5: real_sum={spec_center_true[:, 5, 0].sum():.4f}, imag_sum={spec_center_true[:, 5, 1].sum():.4f}")
    print(f"  Frame 6: real_sum={spec_center_true[:, 6, 0].sum():.4f}, imag_sum={spec_center_true[:, 6, 1].sum():.4f}")

    # Test 2: torch.stft with center=False
    spec_center_false = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                                    win_length=win_length, window=window, return_complex=False,
                                    center=False)
    print(f"\ncenter=False: shape = {spec_center_false.shape}")
    print(f"  Frame 0: real_sum={spec_center_false[:, 0, 0].sum():.4f}, imag_sum={spec_center_false[:, 0, 1].sum():.4f}")
    print(f"  Frame 1: real_sum={spec_center_false[:, 1, 0].sum():.4f}, imag_sum={spec_center_false[:, 1, 1].sum():.4f}")
    print(f"  Frame 5: real_sum={spec_center_false[:, 5, 0].sum():.4f}, imag_sum={spec_center_false[:, 5, 1].sum():.4f}")

    # Manual computation (simulating C)
    def manual_stft_frame(frame_512):
        windowed = frame_512 * window.numpy()
        fft_result = np.fft.rfft(windowed)
        return fft_result.real, fft_result.imag

    print(f"\nManual (simulating C, center=False equivalent):")
    for frame_idx in [0, 1, 5, 6]:
        start = frame_idx * hop_length
        frame_512 = audio_padded[start:start + n_fft]
        manual_real, manual_imag = manual_stft_frame(frame_512)
        print(f"  Frame {frame_idx}: real_sum={manual_real.sum():.4f}, imag_sum={manual_imag.sum():.4f}")

    # Now let's check what Python reference code actually uses
    print("\n=== Checking Python reference code's STFT settings ===")

    # The debug_decoder.py and gtcrn_stream.py use:
    # spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
    #                   win_length=win_length, window=window, return_complex=False)
    # This uses center=True (default) !

    # So Python uses center=True but C uses center=False style (no centering)!
    # This creates a 1-frame offset!

    print("\nCONCLUSION:")
    print("Python uses center=True (default), which adds n_fft//2 zeros on each side")
    print("C uses center=False equivalent (no padding, direct frame extraction)")
    print("This causes a 1-frame offset between Python and C outputs!")

if __name__ == "__main__":
    main()
