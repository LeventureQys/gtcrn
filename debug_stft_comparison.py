"""
Debug STFT comparison between Python torch.stft and C streaming STFT.
"""
import os
import sys
import numpy as np
import torch
import soundfile as sf

def main():
    print("=== Debug STFT Comparison ===\n")

    gtcrn_dir = os.path.dirname(os.path.abspath(__file__))

    # Load noisy audio
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)
    print(f"Audio: {len(audio)} samples at {sr} Hz")

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio (same as C) - add 256 zeros at beginning
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # torch.stft (Python)
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False)
    n_frames = spec.shape[1]
    print(f"Number of frames: {n_frames}")

    # Compare frame-by-frame using manual STFT (simulating C behavior)
    print("\n=== Frame-by-frame comparison ===")
    print("Simulating C streaming STFT and comparing to torch.stft\n")

    # Helper function for manual STFT frame
    def manual_stft_frame(frame_512):
        """Compute STFT for a single 512-sample frame with sqrt-Hann window."""
        # Apply window
        windowed = frame_512 * window.numpy()
        # FFT
        fft_result = np.fft.rfft(windowed)
        return fft_result.real, fft_result.imag

    # Initialize input buffer (same as C's stft_input_buffer)
    stft_input_buffer = np.zeros(256)

    for frame_idx in range(min(10, n_frames)):
        # C-style: build 512-sample window from [prev_buffer, current_frame]
        # What is "current_frame" for streaming?
        # In streaming C code: input_frame = audio samples for this frame
        # Frame 0: stft_input_buffer=zeros, current_frame = audio[0:256] (before padding!)

        # But wait - in Python we padded the audio first!
        # So for Python frame 0, it uses audio_padded[0:512] = [zeros[0:256], audio[0:256]]

        # For C streaming:
        # Frame 0: stft_window = [zeros, frame0]
        #   where frame0 = input audio samples 0-255
        # Frame 1: stft_window = [frame0, frame1]
        #   where frame1 = input audio samples 256-511

        # For Python with padded audio:
        # Frame 0: audio_padded[0:512] = [zeros, audio[0:256]]
        # Frame 1: audio_padded[256:768] = [audio[0:256], audio[256:512]]

        # These match! So what could be different?

        # Get C-style 512 samples
        start = frame_idx * hop_length
        frame_512 = audio_padded[start:start + n_fft]

        if len(frame_512) < n_fft:
            break

        # Manual STFT
        manual_real, manual_imag = manual_stft_frame(frame_512)

        # Compare to torch.stft result
        torch_real = spec[:, frame_idx, 0].numpy()
        torch_imag = spec[:, frame_idx, 1].numpy()

        # Compute sums
        manual_real_sum = manual_real.sum()
        manual_imag_sum = manual_imag.sum()
        torch_real_sum = torch_real.sum()
        torch_imag_sum = torch_imag.sum()

        match = np.allclose(manual_real, torch_real, rtol=1e-4, atol=1e-6) and \
                np.allclose(manual_imag, torch_imag, rtol=1e-4, atol=1e-6)

        print(f"Frame {frame_idx}: Manual real_sum={manual_real_sum:.4f}, imag_sum={manual_imag_sum:.4f}")
        print(f"          torch  real_sum={torch_real_sum:.4f}, imag_sum={torch_imag_sum:.4f}")
        print(f"          Match: {match}")
        print()

    # Now check what C actually computes for frame 6
    print("\n=== Exact values for Frame 5 (Python) / Frame 6 (C) ===")
    frame_idx = 5
    torch_real = spec[:, frame_idx, 0].numpy()
    torch_imag = spec[:, frame_idx, 1].numpy()
    torch_mag = np.sqrt(torch_real**2 + torch_imag**2 + 1e-12)

    print(f"Python torch.stft frame {frame_idx}:")
    print(f"  real sum: {torch_real.sum():.6f}")
    print(f"  imag sum: {torch_imag.sum():.6f}")
    print(f"  mag sum:  {torch_mag.sum():.6f}")
    print(f"  real first 5: {torch_real[:5]}")
    print(f"  imag first 5: {torch_imag[:5]}")

    # Simulate C for frame 5
    # In C streaming, frame 5 means we've processed 5 frames before
    # At frame 5:
    #   - Previous frames have filled stft_input_buffer
    #   - stft_window = [audio[4*256:5*256], audio[5*256:6*256]]  # but with initial zero padding...

    # Actually in C, the input is:
    # gtcrn_process_frame is called with input_frame = raw audio frames
    # The raw audio is NOT padded - the padding is implicit via stft_input_buffer starting at zero

    # So for C frame N (1-indexed), which corresponds to Python frame N-1:
    # C frame 1: stft_window = [zeros, audio[0:256]]
    # C frame 2: stft_window = [audio[0:256], audio[256:512]]
    # ...
    # C frame 6: stft_window = [audio[1024:1280], audio[1280:1536]]

    # But Python frame 5 (0-indexed) uses:
    # audio_padded[5*256:5*256+512] = audio_padded[1280:1792]
    # Since audio_padded = [zeros(256), audio], this is:
    # audio_padded[1280:1792] = audio[1024:1536]
    # Which equals [audio[1024:1280], audio[1280:1536]]

    # So they SHOULD match if C is doing the same thing!

    print("\n=== What C frame 6 should see ===")
    print("(C frame 6 = Python frame 5, both 0-indexed with initial padding)")

    # Build the same window C would build for "frame 6" (1-indexed)
    # After 5 previous calls:
    # stft_input_buffer = audio[1024:1280]
    # input_frame = audio[1280:1536]
    # stft_window = [stft_input_buffer, input_frame] = audio[1024:1536]

    c_frame_5_window = audio[1024:1536]  # 512 samples
    c_real, c_imag = manual_stft_frame(c_frame_5_window)
    c_mag = np.sqrt(c_real**2 + c_imag**2 + 1e-12)

    print(f"C frame 6 (simulated with np.fft):")
    print(f"  real sum: {c_real.sum():.6f}")
    print(f"  imag sum: {c_imag.sum():.6f}")
    print(f"  mag sum:  {c_mag.sum():.6f}")
    print(f"  real first 5: {c_real[:5]}")
    print(f"  imag first 5: {c_imag[:5]}")

    # Check if they match
    print("\n=== Difference ===")
    print(f"Real sum diff: {c_real.sum() - torch_real.sum():.6f}")
    print(f"Imag sum diff: {c_imag.sum() - torch_imag.sum():.6f}")
    print(f"Mag sum diff: {c_mag.sum() - torch_mag.sum():.6f}")

    # The issue might be the window mapping!
    # torch.stft center pads, but C might not?
    print("\n=== Window application check ===")
    print(f"C window len={len(c_frame_5_window)}")
    print(f"audio[1024:1030] = {audio[1024:1030]}")
    print(f"c_frame_5_window[0:6] = {c_frame_5_window[:6]}")

if __name__ == "__main__":
    main()
