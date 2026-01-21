"""
Print all EnConv0 channels for comparison with C.
"""
import os
import sys
import numpy as np
import torch
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream
from gtcrn import GTCRN

def main():
    # Load model
    model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # Load test audio with padding
    test_wav = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(test_wav)
    audio_padded = np.concatenate([np.zeros(256), audio])

    # STFT parameters
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # STFT with center=False
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False,
                      center=False)
    spec = spec.unsqueeze(0)  # (1, freq, time, 2)

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process 5 frames to warm up
    with torch.no_grad():
        for i in range(5):
            frame = spec[:, :, i:i+1, :]
            out, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )

        # Frame 5: manual trace
        frame = spec[:, :, 5:6, :]
        x = frame  # (1, 257, 1, 2)

        # Create feature tensor (mag, real, imag)
        mag = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-12)  # (1, 257, 1)
        feat = torch.stack([mag, x[..., 0], x[..., 1]], dim=1)  # (1, 3, 257, 1)
        feat = feat.permute(0, 1, 3, 2)  # (1, 3, 1, 257)

        # ERB compression
        erb = stream_model.erb
        feat_erb = erb.bm(feat)  # (1, 3, 1, 129)

        # SFE
        sfe = stream_model.sfe
        feat_sfe = sfe(feat_erb)  # (1, 9, 1, 129)

        # Encoder conv 0
        en_out0 = stream_model.encoder.en_convs[0](feat_sfe)  # (1, 16, 1, 65)

        en_out0_np = en_out0[0, :, 0, :].numpy()  # (16, 65)

        print("=== EnConv0 output for C comparison ===")
        for ch in range(16):
            print(f"ch{ch} first 3: {en_out0_np[ch, 0]:.6f} {en_out0_np[ch, 1]:.6f} {en_out0_np[ch, 2]:.6f}  sum: {en_out0_np[ch].sum():.6f}")

if __name__ == "__main__":
    main()
