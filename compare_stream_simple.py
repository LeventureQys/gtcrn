"""
Simple C vs Python streaming comparison using spectrum-level comparison.
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
    print("=== C vs Python Streaming Comparison ===\n")

    # Load C output audio
    c_wav_path = os.path.join(gtcrn_dir, 'GTCRN_C/build/Release/test_out_stream.wav')
    if not os.path.exists(c_wav_path):
        print(f"Error: C output not found: {c_wav_path}")
        return
    c_audio, c_sr = sf.read(c_wav_path)
    print(f"C audio: {len(c_audio)} samples")

    # Load model and generate Python streaming output
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

    # Process all frames and collect output frames
    output_frames = []
    with torch.no_grad():
        for i in range(n_frames):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            output_frames.append(out_frame)

    # Stack output
    output_spec = torch.cat(output_frames, dim=2)  # (1, 257, T, 2)

    # Use frame-by-frame ISTFT to match C implementation
    py_output = []
    for i in range(n_frames):
        spec_frame = output_spec[0, :, i, :]  # (257, 2)
        spec_complex = torch.complex(spec_frame[:, 0], spec_frame[:, 1])

        # ISTFT for single frame (with overlap-add handling like C)
        # Compute inverse FFT
        spec_full = torch.zeros(512, dtype=torch.complex64)
        spec_full[:257] = spec_complex
        spec_full[257:] = spec_complex[1:-1].flip(0).conj()
        frame_time = torch.fft.ifft(spec_full).real
        frame_time = frame_time * window
        py_output.append(frame_time.numpy())

    # Simple overlap-add
    total_len = (n_frames - 1) * hop_length + win_length
    py_audio = np.zeros(total_len)
    for i, frame in enumerate(py_output):
        start = i * hop_length
        py_audio[start:start + win_length] += frame

    # Trim to match input length (accounting for padding)
    py_audio = py_audio[256:256 + len(audio)]

    print(f"Python audio: {len(py_audio)} samples")

    # There's a 256-sample offset due to how C and Python handle the initial padding
    # C outputs: starts from sample 0 of input (after internal padding is consumed)
    # Python: padded audio produces output starting from the padding region
    # So we need to align them - C output lags Python by 256 samples
    c_audio_aligned = c_audio[256:]  # Skip first 256 samples of C output
    py_audio_aligned = py_audio[:-256] if len(py_audio) > 256 else py_audio  # Trim last 256 of Python

    # Align lengths
    min_len = min(len(c_audio_aligned), len(py_audio_aligned))
    c_audio_aligned = c_audio_aligned[:min_len]
    py_audio_aligned = py_audio_aligned[:min_len]

    # Compute correlation on aligned signals
    correlation = np.corrcoef(c_audio_aligned, py_audio_aligned)[0, 1]

    # Compute energy ratio
    c_energy = np.sum(c_audio_aligned**2)
    py_energy = np.sum(py_audio_aligned**2)
    energy_ratio = c_energy / py_energy if py_energy > 0 else 0

    # Compute SNR
    diff = c_audio_aligned - py_audio_aligned
    snr = 10 * np.log10(py_energy / (np.sum(diff**2) + 1e-12))

    print(f"\n=== Results (after 256-sample alignment) ===")
    print(f"Correlation: {correlation:.6f}")
    print(f"Energy ratio (C/Python): {energy_ratio:.6f}")
    print(f"SNR (C vs Python): {snr:.2f} dB")

    # Also check with offsets
    print(f"\n=== Checking with offsets ===")
    for offset in [0, 128, 256, 384, 512]:
        if offset > 0:
            c_shifted = c_audio[offset:]
            py_shifted = py_audio[:-offset]
        else:
            c_shifted = c_audio
            py_shifted = py_audio
        min_l = min(len(c_shifted), len(py_shifted))
        if min_l > 0:
            corr = np.corrcoef(c_shifted[:min_l], py_shifted[:min_l])[0, 1]
            print(f"  Offset {offset}: correlation = {corr:.6f}")

if __name__ == "__main__":
    main()
