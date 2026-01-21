"""
Compare C streaming output with Python streaming output.
"""
import os
import sys
import numpy as np
import soundfile as sf
from scipy.io import wavfile

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    # Read C output
    c_wav_path = os.path.join(gtcrn_dir, 'GTCRN_C/build/Release/test_out_stream.wav')
    c_audio, c_sr = sf.read(c_wav_path)

    # Read Python streaming output (generate if not exists)
    py_wav_path = os.path.join(gtcrn_dir, 'test_wavs/enhanced_stream/python_stream_out.wav')

    if not os.path.exists(py_wav_path):
        # Generate Python streaming output
        import torch
        from gtcrn_stream import StreamGTCRN
        from modules.convert import convert_to_stream
        from gtcrn import GTCRN

        model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
        offline_model = GTCRN().eval()
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        offline_model.load_state_dict(checkpoint['model'])
        stream_model = StreamGTCRN().eval()
        convert_to_stream(stream_model, offline_model)

        # Load noisy audio
        noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
        audio, sr = sf.read(noisy_path)

        # STFT params
        n_fft = 512
        hop_length = 256
        win_length = 512
        window = torch.sqrt(torch.hann_window(win_length))

        # Pad audio
        audio_padded = np.concatenate([np.zeros(256), audio])
        audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))
        spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                          win_length=win_length, window=window, return_complex=False,
                          center=False)
        spec = spec.unsqueeze(0)

        # Initialize caches
        conv_cache = torch.zeros(2, 1, 16, 16, 33)
        tra_cache = torch.zeros(2, 3, 1, 1, 16)
        inter_cache = torch.zeros(2, 1, 33, 16)

        # Process all frames
        n_frames = spec.shape[2]
        output_frames = []

        with torch.no_grad():
            for i in range(n_frames):
                frame = spec[:, :, i:i+1, :]
                out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                    frame, conv_cache, tra_cache, inter_cache
                )
                output_frames.append(out_frame)

        # Stack and ISTFT
        output_spec = torch.cat(output_frames, dim=2)  # (1, 257, T, 2)
        output_spec_complex = torch.complex(output_spec[..., 0], output_spec[..., 1])
        output_spec_complex = output_spec_complex.squeeze(0)  # (257, T)

        # Use overlap-add ISTFT - length must be specified for center=False
        output_length = (output_spec_complex.shape[1] - 1) * hop_length + win_length
        output_audio = torch.istft(output_spec_complex, n_fft=n_fft, hop_length=hop_length,
                                   win_length=win_length, window=window, center=False,
                                   onesided=True, length=output_length)

        # Remove padding
        py_audio = output_audio[256:256+len(audio)].numpy()

        # Save
        os.makedirs(os.path.dirname(py_wav_path), exist_ok=True)
        sf.write(py_wav_path, py_audio, sr)
        print(f"Generated Python streaming output: {py_wav_path}")
    else:
        py_audio, py_sr = sf.read(py_wav_path)

    # Compare
    print(f"\n=== Comparison: C Streaming vs Python Streaming ===")
    print(f"C audio shape: {c_audio.shape}")
    print(f"Python audio shape: {py_audio.shape}")

    # Align lengths
    min_len = min(len(c_audio), len(py_audio))
    c_audio = c_audio[:min_len]
    py_audio = py_audio[:min_len]

    # Compute correlation
    correlation = np.corrcoef(c_audio, py_audio)[0, 1]
    print(f"Correlation: {correlation:.6f}")

    # Compute energy ratio
    c_energy = np.sum(c_audio**2)
    py_energy = np.sum(py_audio**2)
    energy_ratio = c_energy / py_energy
    print(f"Energy ratio (C/Python): {energy_ratio:.6f}")

    # Compute SNR (treating Python as reference)
    diff = c_audio - py_audio
    snr = 10 * np.log10(py_energy / np.sum(diff**2 + 1e-12))
    print(f"SNR (C vs Python): {snr:.2f} dB")

    # Check with offset
    print(f"\n=== Checking with offsets ===")
    for offset in [0, 128, 256, 384, 512]:
        if offset > 0:
            c_shifted = c_audio[offset:]
            py_shifted = py_audio[:-offset]
        else:
            c_shifted = c_audio
            py_shifted = py_audio
        min_l = min(len(c_shifted), len(py_shifted))
        corr = np.corrcoef(c_shifted[:min_l], py_shifted[:min_l])[0, 1]
        print(f"  Offset {offset}: correlation = {corr:.6f}")

if __name__ == "__main__":
    main()
