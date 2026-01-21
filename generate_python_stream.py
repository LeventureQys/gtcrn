"""
Generate Python outputs and compare with C output.
Use offline model for reference and manual overlap-add for streaming.
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

def manual_istft(spec_real, spec_imag, window, hop_length):
    """Manual ISTFT with overlap-add, matching C implementation."""
    n_frames = spec_real.shape[0]
    n_fft = (spec_real.shape[1] - 1) * 2
    win_length = len(window)

    # Output length
    output_length = (n_frames - 1) * hop_length + win_length
    output = np.zeros(output_length)

    for i in range(n_frames):
        # Build full spectrum (conjugate symmetric)
        spec_frame = spec_real[i] + 1j * spec_imag[i]
        spec_full = np.zeros(n_fft, dtype=np.complex64)
        spec_full[:len(spec_frame)] = spec_frame
        spec_full[len(spec_frame):] = np.conj(spec_frame[1:-1][::-1])

        # IFFT
        frame = np.fft.ifft(spec_full).real

        # Window and overlap-add
        frame = frame[:win_length] * window
        start = i * hop_length
        output[start:start + win_length] += frame

    return output

def main():
    print("=== Python vs C Streaming Comparison ===\n")

    # Load model
    model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)
    print("Model loaded")

    # Load noisy audio
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)
    print(f"Input audio: {len(audio)} samples, {sr} Hz")
    print(f"Input energy: {np.sum(audio**2):.2f}")

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))
    window_np = window.numpy()

    # ========== Offline model (reference) ==========
    print("\n--- Offline Model ---")
    audio_tensor = torch.from_numpy(audio.astype(np.float32))

    # STFT for offline model input
    input_spec = torch.stft(audio_tensor, n_fft, hop_length, win_length, window, return_complex=False)
    print(f"Input spec shape: {input_spec.shape}")  # (257, T, 2)

    with torch.no_grad():
        output_spec = offline_model(input_spec.unsqueeze(0))[0]  # (257, T, 2)

    print(f"Output spec shape: {output_spec.shape}")

    # ISTFT
    output_complex = torch.complex(output_spec[..., 0], output_spec[..., 1])
    offline_audio = torch.istft(output_complex, n_fft, hop_length, win_length, window, return_complex=False)
    offline_audio = offline_audio.numpy()

    print(f"Offline output: {len(offline_audio)} samples")
    print(f"Offline energy: {np.sum(offline_audio**2):.2f}")

    offline_path = os.path.join(gtcrn_dir, 'GTCRN_C/build/Release/python_offline_out.wav')
    sf.write(offline_path, offline_audio, sr)
    print(f"Saved: {offline_path}")

    # ========== Streaming model ==========
    print("\n--- Streaming Model ---")

    # Pad audio (same as C)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # STFT
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False)
    spec = spec.unsqueeze(0)  # (1, 257, T, 2)
    n_frames = spec.shape[2]
    print(f"Number of frames: {n_frames}")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process frames
    spec_real_out = []
    spec_imag_out = []
    with torch.no_grad():
        for i in range(n_frames):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            spec_real_out.append(out_frame[0, :, 0, 0].numpy())
            spec_imag_out.append(out_frame[0, :, 0, 1].numpy())

    spec_real_out = np.array(spec_real_out)  # (T, 257)
    spec_imag_out = np.array(spec_imag_out)

    # Manual ISTFT
    stream_audio = manual_istft(spec_real_out, spec_imag_out, window_np, hop_length)

    # Remove padding and trim
    stream_audio = stream_audio[256:256+len(audio)]

    print(f"Stream output: {len(stream_audio)} samples")
    print(f"Stream energy: {np.sum(stream_audio**2):.2f}")

    stream_path = os.path.join(gtcrn_dir, 'GTCRN_C/build/Release/python_stream_out.wav')
    sf.write(stream_path, stream_audio, sr)
    print(f"Saved: {stream_path}")

    # ========== Compare with C output ==========
    print("\n--- Comparison with C Output ---")
    c_output_path = os.path.join(gtcrn_dir, 'GTCRN_C/build/Release/c_stream_out.wav')

    if os.path.exists(c_output_path):
        c_audio, _ = sf.read(c_output_path)
        print(f"C output: {len(c_audio)} samples")
        print(f"C energy: {np.sum(c_audio**2):.2f}")

        # Align lengths
        min_len = min(len(stream_audio), len(c_audio))
        py = stream_audio[:min_len]
        c = c_audio[:min_len]

        # Direct comparison
        corr = np.corrcoef(py, c)[0, 1]
        diff = py - c
        snr = 10 * np.log10(np.sum(py**2) / (np.sum(diff**2) + 1e-12))

        print(f"\nDirect comparison:")
        print(f"  Correlation: {corr:.6f}")
        print(f"  SNR: {snr:.2f} dB")

        # With offset (C may be delayed by 256 samples)
        print(f"\nWith 256-sample offset (C[256:] vs Py[:-256]):")
        if min_len > 256:
            c_shifted = c_audio[256:min_len]
            py_shifted = stream_audio[:min_len-256]
            corr2 = np.corrcoef(c_shifted, py_shifted)[0, 1]
            diff2 = c_shifted - py_shifted
            snr2 = 10 * np.log10(np.sum(py_shifted**2) / (np.sum(diff2**2) + 1e-12))
            print(f"  Correlation: {corr2:.6f}")
            print(f"  SNR: {snr2:.2f} dB")

        # Compare Python stream vs offline
        print(f"\nPython Stream vs Offline:")
        min_len2 = min(len(stream_audio), len(offline_audio))
        corr3 = np.corrcoef(stream_audio[:min_len2], offline_audio[:min_len2])[0, 1]
        print(f"  Correlation: {corr3:.6f}")

        # Check if C output looks correct
        print(f"\n--- Signal Characteristics ---")
        print(f"Input max amplitude: {np.max(np.abs(audio)):.4f}")
        print(f"Offline max amplitude: {np.max(np.abs(offline_audio)):.4f}")
        print(f"Py stream max amplitude: {np.max(np.abs(stream_audio)):.4f}")
        print(f"C stream max amplitude: {np.max(np.abs(c_audio)):.4f}")

        # Check first few samples
        print(f"\nFirst 10 samples:")
        print(f"  Input:   {audio[:10]}")
        print(f"  Offline: {offline_audio[:10]}")
        print(f"  C:       {c_audio[:10]}")

    else:
        print(f"C output not found: {c_output_path}")

if __name__ == "__main__":
    main()
