"""
Compare Python streaming vs offline output.
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
    print("=== Python Stream vs Offline Comparison ===\n")

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

    # STFT params
    n_fft = 512
    hop_length = 256
    window = torch.sqrt(torch.hann_window(n_fft))

    # Offline processing - input is STFT output
    audio_tensor = torch.from_numpy(audio.astype(np.float32))
    spec_full = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                           win_length=n_fft, window=window, return_complex=False)
    spec_full = spec_full.unsqueeze(0)  # (1, 257, T, 2)

    with torch.no_grad():
        output_spec_offline = offline_model(spec_full)  # (1, 257, T, 2)

    # ISTFT to get audio
    output_spec_complex_offline = torch.complex(output_spec_offline[0, :, :, 0], output_spec_offline[0, :, :, 1])
    enhanced_offline = torch.istft(output_spec_complex_offline, n_fft=n_fft, hop_length=hop_length,
                                    win_length=n_fft, window=window, length=len(audio))
    enhanced_offline = enhanced_offline.numpy()

    # Stream processing - use same spec_full
    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    n_frames = spec_full.shape[2]
    print(f"Number of frames: {n_frames}")

    # Process frame by frame
    output_specs = []
    with torch.no_grad():
        for i in range(n_frames):
            frame = spec_full[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            output_specs.append(out_frame)

    # Stack outputs
    output_spec = torch.cat(output_specs, dim=2)  # (1, 257, T, 2)

    # ISTFT to get audio
    output_spec_complex = torch.complex(output_spec[0, :, :, 0], output_spec[0, :, :, 1])
    enhanced_stream = torch.istft(output_spec_complex, n_fft=n_fft, hop_length=hop_length,
                                   win_length=n_fft, window=window, length=len(audio))
    enhanced_stream = enhanced_stream.numpy()

    print(f"Offline output: {len(enhanced_offline)} samples")
    print(f"Stream output: {len(enhanced_stream)} samples")

    # Compare
    min_len = min(len(enhanced_offline), len(enhanced_stream))
    off = enhanced_offline[:min_len]
    stm = enhanced_stream[:min_len]

    correlation = np.corrcoef(off, stm)[0, 1]
    mse = np.mean((off - stm)**2)
    max_diff = np.max(np.abs(off - stm))

    print(f"\n=== Python Stream vs Offline ===")
    print(f"Correlation: {correlation:.6f}")
    print(f"MSE: {mse:.8f}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"\nFirst 10 samples (offline): {off[:10]}")
    print(f"First 10 samples (stream):  {stm[:10]}")

    # Save for reference
    sf.write(os.path.join(gtcrn_dir, 'test_wavs/output_py_offline.wav'), enhanced_offline, sr)
    sf.write(os.path.join(gtcrn_dir, 'test_wavs/output_py_stream_proper.wav'), enhanced_stream, sr)
    print(f"\nSaved outputs to test_wavs/output_py_offline.wav and output_py_stream_proper.wav")

if __name__ == "__main__":
    main()
