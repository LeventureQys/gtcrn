"""
Compare C vs Python streaming at the spectrum level (before ISTFT).
This avoids ISTFT implementation differences.
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
    print("=== C vs Python Streaming Spectrum Comparison ===\n")

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
    print(f"Input audio: {len(audio)} samples")

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio with 256 zeros (matches C streaming)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # Compute STFT
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

    # Process all frames
    py_spec_real = []
    py_spec_imag = []
    with torch.no_grad():
        for i in range(n_frames):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            # out_frame: (1, 257, 1, 2)
            py_spec_real.append(out_frame[0, :, 0, 0].numpy())
            py_spec_imag.append(out_frame[0, :, 0, 1].numpy())

    py_spec_real = np.array(py_spec_real)  # (T, 257)
    py_spec_imag = np.array(py_spec_imag)  # (T, 257)

    print(f"Python output spectrum shape: {py_spec_real.shape}")

    # Load C spectrum output (if saved) or compare via audio
    # For now, let's compute statistics on Python output
    # and compare with C's intermediate debug values

    # Check specific frame (frame 5 = C frame_count 6)
    frame_idx = 5
    print(f"\n=== Frame {frame_idx} (C frame_count={frame_idx+1}) ===")
    print(f"Python spec_real sum: {py_spec_real[frame_idx].sum():.6f}")
    print(f"Python spec_imag sum: {py_spec_imag[frame_idx].sum():.6f}")
    print(f"Python spec magnitude sum: {np.sqrt(py_spec_real[frame_idx]**2 + py_spec_imag[frame_idx]**2).sum():.6f}")

    # Overall statistics
    print(f"\n=== Overall Statistics ===")
    print(f"Python spec_real total abs sum: {np.abs(py_spec_real).sum():.6f}")
    print(f"Python spec_imag total abs sum: {np.abs(py_spec_imag).sum():.6f}")

    # Compare specific intermediate values with C
    print(f"\n=== Comparison with C Debug Output ===")
    print(f"Expected C EnGT2 sum (frame 6): ~114.750")
    print(f"Expected C EnGT3 sum (frame 6): ~64.33")
    print(f"Expected C EnGT4 sum (frame 6): ~35.52")

if __name__ == "__main__":
    main()
