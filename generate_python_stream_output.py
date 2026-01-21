"""
Generate Python streaming output and compare with C output.
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
    print("=== Python Streaming Output Generation ===\n")

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
    print(f"Input audio: {len(audio)} samples at {sr} Hz")

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio (same as C)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # STFT - use center=False to match C behavior
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

    # Manual ISTFT to match C behavior exactly
    output_spec_complex = output_spec[0, :, :, 0] + 1j * output_spec[0, :, :, 1]  # (257, T)
    output_spec_complex = output_spec_complex.numpy()

    # Reconstruct full spectrum (conjugate symmetric)
    full_spec = np.zeros((n_fft, n_frames), dtype=np.complex64)
    full_spec[:257, :] = output_spec_complex
    for i in range(1, n_fft // 2):
        full_spec[n_fft - i, :] = np.conj(full_spec[i, :])

    # IFFT each frame
    output_len = (n_frames - 1) * hop_length + win_length
    output_audio = np.zeros(output_len)
    window_sum = np.zeros(output_len)
    window_np = window.numpy()

    for t in range(n_frames):
        frame_freq = full_spec[:, t]
        frame_time = np.fft.ifft(frame_freq).real
        # Apply window
        frame_windowed = frame_time * window_np
        # Overlap-add
        start = t * hop_length
        output_audio[start:start + win_length] += frame_windowed
        window_sum[start:start + win_length] += window_np ** 2

    # Normalize by window sum
    eps = 1e-8
    output_audio = output_audio / (window_sum + eps)

    # Remove initial padding
    output_audio = output_audio[256:]

    # Save output
    output_path = os.path.join(gtcrn_dir, 'test_wavs/output_py_stream.wav')
    sf.write(output_path, output_audio, sr)
    print(f"Output audio saved to: {output_path}")

    # Compare with C output
    c_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_c_stream.wav')
    c_output, _ = sf.read(c_output_path)
    print(f"\nC output: {len(c_output)} samples")
    print(f"Python output: {len(output_audio)} samples")

    # Align lengths
    min_len = min(len(c_output), len(output_audio))
    c_out = c_output[:min_len]
    py_out = output_audio[:min_len]

    # Compute metrics
    correlation = np.corrcoef(c_out, py_out)[0, 1]
    mse = np.mean((c_out - py_out)**2)
    max_diff = np.max(np.abs(c_out - py_out))
    energy_ratio = np.sum(c_out**2) / (np.sum(py_out**2) + 1e-10)

    print(f"\n=== Comparison Metrics ===")
    print(f"Correlation: {correlation:.6f}")
    print(f"MSE: {mse:.8f}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Energy ratio (C/Py): {energy_ratio:.6f}")

    # Check sample values
    print(f"\n=== Sample Values ===")
    print(f"First 10 samples (C):  {c_out[:10]}")
    print(f"First 10 samples (Py): {py_out[:10]}")

    # Check at different positions
    mid = min_len // 2
    print(f"\nMiddle samples (C):  {c_out[mid:mid+10]}")
    print(f"Middle samples (Py): {py_out[mid:mid+10]}")

    if correlation > 0.99:
        print(f"\n✓ SUCCESS: Correlation > 0.99 - C and Python outputs match!")
    elif correlation > 0.95:
        print(f"\n~ GOOD: Correlation > 0.95 - Minor differences")
    else:
        print(f"\n✗ MISMATCH: Correlation < 0.95 - Significant differences")

if __name__ == "__main__":
    main()
