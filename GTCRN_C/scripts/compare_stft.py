#!/usr/bin/env python3
"""
Compare STFT outputs between C streaming and Python batch processing.
"""

import os
import sys
import numpy as np
import torch
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

def main():
    print("=== STFT Comparison ===\n")

    # Load test audio
    test_wav = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
    audio, sr = sf.read(test_wav)
    print(f"Loaded audio: {len(audio)} samples, sr={sr}")

    # STFT parameters
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Python batch STFT
    audio_tensor = torch.from_numpy(audio).float()
    spec_batch = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, return_complex=False)
    print(f"Batch STFT shape: {spec_batch.shape}")  # (freq, time, 2)

    # Simulate C streaming STFT
    # For streaming: each frame uses [prev_256 + current_256] = 512 samples
    # Frame 0: [zeros(256) + audio[0:256]]
    # Frame 1: [audio[0:256] + audio[256:512]]
    # Frame 2: [audio[256:512] + audio[512:768]]
    # etc.

    num_frames = (len(audio) - n_fft) // hop_length + 1
    spec_stream = []

    prev_samples = np.zeros(hop_length, dtype=np.float32)

    for i in range(num_frames):
        # Get current 256 samples
        start = i * hop_length
        end = start + hop_length
        if end > len(audio):
            break
        current_samples = audio[start:end].astype(np.float32)

        # Build 512-sample window
        window_samples = np.concatenate([prev_samples, current_samples])

        # Compute STFT for this window
        window_tensor = torch.from_numpy(window_samples).float()
        frame_spec = torch.stft(window_tensor, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window, return_complex=False)
        # frame_spec: (freq, 1, 2) or (freq, 2, 2) depending on centering

        # Update prev_samples
        prev_samples = current_samples.copy()

        spec_stream.append(frame_spec)

    print(f"Stream frames computed: {len(spec_stream)}")
    if len(spec_stream) > 0:
        print(f"Frame 0 shape: {spec_stream[0].shape}")

    # Compare first few frames
    print("\n=== Frame Comparison ===")
    for i in range(min(5, len(spec_stream), spec_batch.shape[1])):
        batch_frame = spec_batch[:, i, :]  # (freq, 2)
        stream_frame = spec_stream[i][:, 0, :]  # (freq, 2) - take first time step

        diff = (batch_frame - stream_frame).abs()
        corr_real = torch.corrcoef(torch.stack([batch_frame[:, 0], stream_frame[:, 0]]))[0, 1]
        corr_imag = torch.corrcoef(torch.stack([batch_frame[:, 1], stream_frame[:, 1]]))[0, 1]

        print(f"Frame {i}:")
        print(f"  Batch sum: {batch_frame.sum():.6f}, Stream sum: {stream_frame.sum():.6f}")
        print(f"  Max diff: {diff.max():.6f}")
        print(f"  Corr (real): {corr_real:.6f}, Corr (imag): {corr_imag:.6f}")


if __name__ == "__main__":
    main()
