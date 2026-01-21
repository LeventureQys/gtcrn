"""
Compare C and Python STFT output for frame 0.
"""
import sys
import os
import numpy as np

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

import torch
import scipy.io.wavfile as wav

def main():
    # Load test audio
    audio_path = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00001_1_fan_noise_level1_snr-5dB_noisy.wav")
    sr, audio = wav.read(audio_path)
    audio = audio.astype(np.float32) / 32768.0

    # STFT parameters
    n_fft = 512
    hop_len = 256
    win_len = 512

    # Create sqrt-Hann window (same as C version)
    window = torch.sqrt(torch.hann_window(win_len))

    print("=" * 60)
    print("STFT Comparison: C vs Python")
    print("=" * 60)

    # First, check the raw audio values for the first frame
    first_frame_audio = audio[:512]
    print(f"\nFirst frame audio stats:")
    print(f"  Sum: {np.sum(first_frame_audio):.6f}")
    print(f"  Mean: {np.mean(first_frame_audio):.6f}")
    print(f"  Max: {np.max(first_frame_audio):.6f}")
    print(f"  First 5: {first_frame_audio[:5]}")

    # Apply window
    windowed = first_frame_audio * window.numpy()
    print(f"\nWindowed audio stats:")
    print(f"  Sum: {np.sum(windowed):.6f}")
    print(f"  Mean: {np.mean(windowed):.6f}")
    print(f"  Max: {np.max(windowed):.6f}")

    # Do FFT
    fft_result = np.fft.rfft(windowed)  # 257 complex values
    print(f"\nFFT result (rfft):")
    print(f"  Real sum: {np.sum(fft_result.real):.4f}")
    print(f"  Imag sum: {np.sum(fft_result.imag):.4f}")
    print(f"  Mag sum: {np.sum(np.abs(fft_result)):.4f}")
    print(f"  First 10 real: {fft_result[:10].real}")
    print(f"  First 10 imag: {fft_result[:10].imag}")

    # Now compare with PyTorch STFT
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    spec = torch.stft(audio_tensor, n_fft, hop_len, win_len, window, return_complex=False)
    # spec shape: (1, 257, T, 2)

    frame0_real = spec[0, :, 0, 0].numpy()
    frame0_imag = spec[0, :, 0, 1].numpy()

    print(f"\nPyTorch STFT frame 0:")
    print(f"  Real sum: {np.sum(frame0_real):.4f}")
    print(f"  Imag sum: {np.sum(frame0_imag):.4f}")
    print(f"  Mag sum: {np.sum(np.sqrt(frame0_real**2 + frame0_imag**2)):.4f}")
    print(f"  First 10 real: {frame0_real[:10]}")
    print(f"  First 10 imag: {frame0_imag[:10]}")

    # Check if they match
    match = np.allclose(fft_result.real, frame0_real, atol=1e-6) and \
            np.allclose(fft_result.imag, frame0_imag, atol=1e-6)
    print(f"\nNumpy rfft matches PyTorch STFT: {match}")

    # Check with center=True (default for PyTorch)
    print("\n" + "=" * 60)
    print("Note: PyTorch STFT uses center=True by default!")
    print("This means it pads the signal with win_len//2 on each side.")
    print("=" * 60)

    # Compute what PyTorch is doing with center=True
    # The first frame takes input from [-win_len//2, win_len//2]
    # With center padding, this becomes [0:win_len]
    # But the padding is zeros on the left side

    # Let's compute without centering
    spec_no_center = torch.stft(audio_tensor, n_fft, hop_len, win_len, window,
                                 return_complex=False, center=False)
    print(f"\nPyTorch STFT without center (first frame = audio[0:512]):")
    print(f"  Real sum: {np.sum(spec_no_center[0, :, 0, 0].numpy()):.4f}")
    print(f"  Imag sum: {np.sum(spec_no_center[0, :, 0, 1].numpy()):.4f}")

    # Compare with numpy rfft
    print(f"\nNumpy rfft (audio[0:512]):")
    print(f"  Real sum: {np.sum(fft_result.real):.4f}")
    print(f"  Imag sum: {np.sum(fft_result.imag):.4f}")

if __name__ == "__main__":
    main()
