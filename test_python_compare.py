"""
Compare Python offline vs Python streaming output to establish baseline behavior.
"""
import sys
import os

# Add the gtcrn directory to path
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
    audio_path = os.path.join(gtcrn_dir, "test_wavs/noisy/00003_1_fan_noise_level1_snr+5dB_noisy.wav")
    sr, audio = wav.read(audio_path)
    audio = audio.astype(np.float32) / 32768.0

    # Limit length for testing
    audio = audio[:sr * 3]  # 3 seconds
    print(f"Audio length: {len(audio)} samples ({len(audio)/sr:.2f}s)")

    # Load weights
    ckpt = torch.load(os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar"), map_location='cpu')
    # Handle different checkpoint formats
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # STFT parameters
    n_fft = 512
    hop_len = 256
    win_len = 512

    # Create window (sqrt-Hann)
    window = torch.sqrt(torch.hann_window(win_len))

    # ============================================================
    # Python Offline Model
    # ============================================================
    print("\n=== Python Offline Model ===")
    offline_model = GTCRN()
    offline_model.load_state_dict(state_dict)
    offline_model.eval()

    # STFT - use return_complex=False to get (B, F, T, 2) format
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, N)
    spec = torch.stft(audio_tensor, n_fft, hop_len, win_len, window, return_complex=False)
    # spec shape: (1, 257, T, 2) - this is what GTCRN expects
    print(f"Spec input shape: {spec.shape}")  # (B, F, T, 2)

    with torch.no_grad():
        offline_output = offline_model(spec)  # (1, 257, T, 2)

    print(f"Offline output shape: {offline_output.shape}")

    # ISTFT - convert (B, F, T, 2) to complex tensor for istft
    offline_spec = torch.complex(offline_output[..., 0], offline_output[..., 1])  # (1, 257, T)
    offline_audio = torch.istft(offline_spec, n_fft, hop_len, win_len, window)
    offline_audio = offline_audio.squeeze().numpy()

    print(f"Offline output length: {len(offline_audio)}")
    offline_energy = np.sum(offline_audio ** 2)
    print(f"Offline output energy: {offline_energy:.4f}")

    # ============================================================
    # Python Streaming Model
    # ============================================================
    print("\n=== Python Streaming Model ===")
    stream_model = StreamGTCRN()
    # Use convert_to_stream to properly convert weights
    convert_to_stream(stream_model, offline_model)
    stream_model.eval()

    # Initialize cache
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process frame by frame
    # spec shape is (1, 257, T, 2)
    num_frames = spec.shape[2]
    stream_outputs = []

    with torch.no_grad():
        for i in range(num_frames):
            # Extract single frame: (1, 257, 1, 2)
            frame_input = spec[:, :, i:i+1, :]

            # Process
            output, conv_cache, tra_cache, inter_cache = stream_model(
                frame_input, conv_cache, tra_cache, inter_cache
            )

            # output: (1, 257, 1, 2)
            stream_outputs.append(output)

    # Concatenate outputs: (1, 257, T, 2)
    stream_output = torch.cat(stream_outputs, dim=2)
    print(f"Stream output shape: {stream_output.shape}")

    # ISTFT - convert (B, F, T, 2) to complex tensor for istft
    stream_spec = torch.complex(stream_output[..., 0], stream_output[..., 1])  # (1, 257, T)
    stream_audio = torch.istft(stream_spec, n_fft, hop_len, win_len, window)
    stream_audio = stream_audio.squeeze().numpy()

    print(f"Streaming output length: {len(stream_audio)}")
    stream_energy = np.sum(stream_audio ** 2)
    print(f"Streaming output energy: {stream_energy:.4f}")

    # ============================================================
    # Compare
    # ============================================================
    print("\n=== Comparison ===")

    # Align lengths
    min_len = min(len(offline_audio), len(stream_audio))
    offline_aligned = offline_audio[:min_len]
    stream_aligned = stream_audio[:min_len]

    # Also try with 1-frame offset (streaming has inherent delay)
    offset = hop_len
    if min_len > offset:
        offline_offset = offline_audio[:-offset] if offset > 0 else offline_audio
        stream_offset = stream_audio[offset:] if offset > 0 else stream_audio
        offset_len = min(len(offline_offset), len(stream_offset))
        offline_offset = offline_offset[:offset_len]
        stream_offset = stream_offset[:offset_len]

    # Energy comparison
    offline_e = np.sum(offline_aligned ** 2)
    stream_e = np.sum(stream_aligned ** 2)
    print(f"Offline energy (aligned): {offline_e:.4f}")
    print(f"Streaming energy (aligned): {stream_e:.4f}")
    print(f"Energy ratio (stream/offline): {stream_e / offline_e:.4f}")
    print(f"RMS ratio (stream/offline): {np.sqrt(stream_e / offline_e):.4f}")

    # Correlation
    corr = np.corrcoef(offline_aligned, stream_aligned)[0, 1]
    print(f"\nCorrelation (no offset): {corr:.4f}")

    if min_len > offset:
        corr_offset = np.corrcoef(offline_offset, stream_offset)[0, 1]
        print(f"Correlation (1-frame offset): {corr_offset:.4f}")

        # Energy with offset
        offline_e_off = np.sum(offline_offset ** 2)
        stream_e_off = np.sum(stream_offset ** 2)
        print(f"\nWith 1-frame offset:")
        print(f"  Offline energy: {offline_e_off:.4f}")
        print(f"  Streaming energy: {stream_e_off:.4f}")
        print(f"  Energy ratio: {stream_e_off / offline_e_off:.4f}")
        print(f"  RMS ratio: {np.sqrt(stream_e_off / offline_e_off):.4f}")

    # Save outputs for inspection
    out_dir = os.path.join(gtcrn_dir, "test_wavs")
    wav.write(os.path.join(out_dir, "python_offline_out.wav"), sr, (offline_audio * 32768).astype(np.int16))
    wav.write(os.path.join(out_dir, "python_stream_out.wav"), sr, (stream_audio * 32768).astype(np.int16))
    print(f"\nSaved outputs to {out_dir}")

if __name__ == "__main__":
    main()
