"""
Compare C streaming output with Python using center=False (matching C behavior).
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
    print("=== Compare C vs Python (both using center=False equivalent) ===\n")

    # Load models
    model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # Load noisy audio
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)
    print(f"Input audio: {len(audio)} samples at {sr} Hz")

    n_fft = 512
    hop_length = 256
    window = torch.sqrt(torch.hann_window(n_fft))

    # Pad audio same as C implementation: 256 zeros at the start
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # STFT with center=False to match C behavior
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=n_fft, window=window, return_complex=False, center=False)
    spec = spec.unsqueeze(0)  # (1, 257, T, 2)
    n_frames = spec.shape[2]
    print(f"Number of frames: {n_frames}")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process frame by frame
    output_specs = []
    with torch.no_grad():
        for i in range(n_frames):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            output_specs.append(out_frame)

    # Stack outputs
    output_spec = torch.cat(output_specs, dim=2)  # (1, 257, T, 2)
    print(f"Output spec shape: {output_spec.shape}")

    # Manual ISTFT with overlap-add matching C behavior
    # Reconstruct full spectrum (conjugate symmetric)
    output_spec_np = output_spec[0].numpy()  # (257, T, 2)
    full_spec = np.zeros((n_fft, n_frames), dtype=np.complex64)
    full_spec[:257, :] = output_spec_np[:, :, 0] + 1j * output_spec_np[:, :, 1]
    for i in range(1, n_fft // 2):
        full_spec[n_fft - i, :] = np.conj(full_spec[i, :])

    # IFFT each frame and overlap-add (same as C)
    output_len = n_frames * hop_length + hop_length  # Account for OLA buffer
    output_audio = np.zeros(output_len)
    window_np = window.numpy()

    # First frame: output directly
    frame_freq = full_spec[:, 0]
    frame_time = np.fft.ifft(frame_freq).real
    frame_windowed = frame_time * window_np
    output_audio[0:hop_length] = frame_windowed[0:hop_length]
    ola_buffer = frame_windowed[hop_length:].copy()

    # Subsequent frames: overlap-add
    for t in range(1, n_frames):
        frame_freq = full_spec[:, t]
        frame_time = np.fft.ifft(frame_freq).real
        frame_windowed = frame_time * window_np

        start = t * hop_length
        output_audio[start:start + hop_length] = ola_buffer + frame_windowed[0:hop_length]
        ola_buffer = frame_windowed[hop_length:].copy()

    # Remove initial padding (256 samples)
    output_audio = output_audio[hop_length:]  # Skip first hop (from padding)

    # Trim to original length
    output_audio = output_audio[:len(audio)]

    print(f"Python output: {len(output_audio)} samples")

    # Save Python output
    py_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_py_stream_centerFalse.wav')
    sf.write(py_output_path, output_audio, sr)
    print(f"Saved to: {py_output_path}")

    # Load C output
    c_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_c_stream.wav')
    c_out, _ = sf.read(c_output_path)
    print(f"C output: {len(c_out)} samples")

    # Compare
    min_len = min(len(output_audio), len(c_out))
    py_out = output_audio[:min_len]
    c_out = c_out[:min_len]

    correlation = np.corrcoef(c_out, py_out)[0, 1]
    mse = np.mean((c_out - py_out)**2)
    max_diff = np.max(np.abs(c_out - py_out))

    print(f"\n=== C vs Python (center=False) ===")
    print(f"Correlation: {correlation:.6f}")
    print(f"MSE: {mse:.8f}")
    print(f"Max difference: {max_diff:.6f}")

    print(f"\nFirst 10 samples (C):  {c_out[:10]}")
    print(f"First 10 samples (Py): {py_out[:10]}")

    # Find where they differ
    diffs = np.abs(c_out - py_out)
    max_diff_idx = np.argmax(diffs)
    print(f"\nMax diff at sample {max_diff_idx}: C={c_out[max_diff_idx]:.6f}, Py={py_out[max_diff_idx]:.6f}")

    # Middle samples
    mid = min_len // 2
    print(f"\nMiddle samples (C):  {c_out[mid:mid+10]}")
    print(f"Middle samples (Py): {py_out[mid:mid+10]}")

if __name__ == "__main__":
    main()
