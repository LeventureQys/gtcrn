"""
Debug mask application in Python to compare with C.
Focus on Frame 6 (5 in 0-indexed Python)
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
    print("=== Debug Mask Application ===\n")

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

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio (same as C)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # STFT with center=False (same as C)
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False,
                      center=False)
    spec = spec.unsqueeze(0)  # (1, 257, T, 2)
    n_frames = spec.shape[2]

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process frames 0-5 to build state
    target_frame = 5  # This is Frame 6 in C (1-indexed)
    with torch.no_grad():
        for i in range(target_frame):
            frame = spec[:, :, i:i+1, :]
            _, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )

        # Now process frame 5 manually to trace mask
        frame = spec[:, :, target_frame:target_frame+1, :]

        print(f"=== Python Frame {target_frame} (C Frame {target_frame+1}) ===\n")

        # Input spectrum
        spec_real = frame[0, :, 0, 0].numpy()
        spec_imag = frame[0, :, 0, 1].numpy()
        print(f"spec_real first 10: {spec_real[:10]}")
        print(f"spec_imag first 10: {spec_imag[:10]}")
        print(f"C expected spec_real first 10: 0.135563 0.137168 -0.273219 0.159885 -0.130444 -0.057147 0.249628 -0.034962 -0.137430 0.020975")
        print(f"C expected spec_imag first 10: 0.000000 -0.019988 -0.173454 0.057402 0.070443 0.075607 -0.082540 0.018239 -0.015287 0.049798")

        # Forward through model to get mask
        spec_ref = frame
        spec_real_t = frame[..., 0].permute(0,2,1)
        spec_imag_t = frame[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real_t**2 + spec_imag_t**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real_t, spec_imag_t], dim=1)

        feat = stream_model.erb.bm(feat)
        feat = stream_model.sfe(feat)
        feat, en_outs, conv_cache[0], tra_cache[0] = stream_model.encoder(feat, conv_cache[0], tra_cache[0])
        feat, inter_cache[0] = stream_model.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = stream_model.dpgrnn2(feat, inter_cache[1])
        mask, conv_cache[1], tra_cache[1] = stream_model.decoder(feat, en_outs, conv_cache[1], tra_cache[1])

        # Mask shape: (1, 2, 1, 129) - before ERB expansion
        print(f"\nmask before ERB expansion shape: {mask.shape}")
        mask_real_erb = mask[0, 0, 0, :].numpy()
        mask_imag_erb = mask[0, 1, 0, :].numpy()
        print(f"mask_real_erb sum: {mask_real_erb.sum():.6f}")
        print(f"mask_imag_erb sum: {mask_imag_erb.sum():.6f}")

        # ERB expansion
        mask_expanded = stream_model.erb.bs(mask)  # (1, 2, 1, 257)
        print(f"\nmask after ERB expansion shape: {mask_expanded.shape}")
        mask_real_exp = mask_expanded[0, 0, 0, :].numpy()
        mask_imag_exp = mask_expanded[0, 1, 0, :].numpy()
        print(f"mask_real first 10: {mask_real_exp[:10]}")
        print(f"mask_imag first 10: {mask_imag_exp[:10]}")
        print(f"mask_real sum: {mask_real_exp.sum():.6f}")
        print(f"mask_imag sum: {mask_imag_exp.sum():.6f}")
        print(f"C expected mask_real sum: 8.864581")
        print(f"C expected mask_imag sum: 0.081881")

        # Now apply complex multiplication: output = mask * spec
        # Complex mult: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        a = mask_real_exp  # mask real
        b = mask_imag_exp  # mask imag
        c = spec_real  # spec real
        d = spec_imag  # spec imag

        out_real = a * c - b * d
        out_imag = a * d + b * c

        print(f"\n=== Complex multiplication output ===")
        print(f"out_real first 10: {out_real[:10]}")
        print(f"out_imag first 10: {out_imag[:10]}")
        print(f"out_real sum: {out_real.sum():.6f}")
        print(f"out_imag sum: {out_imag.sum():.6f}")

        # Now use the model's mask module
        out_frame = stream_model.mask(mask_expanded, spec_ref)
        print(f"\n=== Model mask output ===")
        print(f"out_frame shape: {out_frame.shape}")  # (1, 257, 1, 2)
        out_real_model = out_frame[0, :, 0, 0].numpy()
        out_imag_model = out_frame[0, :, 0, 1].numpy()
        print(f"out_real_model first 10: {out_real_model[:10]}")
        print(f"out_imag_model first 10: {out_imag_model[:10]}")
        print(f"out_real_model sum: {out_real_model.sum():.6f}")
        print(f"out_imag_model sum: {out_imag_model.sum():.6f}")

if __name__ == "__main__":
    main()
