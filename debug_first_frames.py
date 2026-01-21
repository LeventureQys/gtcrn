"""
Debug Frame 1 to check if discrepancy exists from the start.
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
    print("=== Debug First Few Frames ===\n")

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

    # STFT
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False)
    spec = spec.unsqueeze(0)  # (1, 257, T, 2)

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    print("Frame-by-frame comparison (Python vs C expected from debug output):\n")
    print("Note: C debug output frame N is 1-indexed, Python frame N is 0-indexed")
    print("C Frame 1 = Python Frame 1, C Frame 2 = Python Frame 2, etc.\n")

    # C debug shows for Frame 1:
    # [DeGT0-F1] BEFORE depth_conv: h1 sum=116.40, cache sum=0.00
    # This means DeGT0 input (after all encoder + DPGRNN) has h1 sum ~116.40

    with torch.no_grad():
        for frame_idx in range(1, 8):  # Frames 1-7 (matching C Frame 1-7)
            frame = spec[:, :, frame_idx:frame_idx+1, :]

            # Process through encoder
            spec_real = frame[..., 0].permute(0,2,1)
            spec_imag = frame[..., 1].permute(0,2,1)
            spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
            feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)

            feat = stream_model.erb.bm(feat)
            feat = stream_model.sfe(feat)
            feat, en_outs, conv_cache[0], tra_cache[0] = stream_model.encoder(feat, conv_cache[0], tra_cache[0])
            feat, inter_cache[0] = stream_model.dpgrnn1(feat, inter_cache[0])
            feat, inter_cache[1] = stream_model.dpgrnn2(feat, inter_cache[1])

            # Skip connection for DeGT0
            x = feat + en_outs[4]
            x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]

            # SFE and point_conv1
            gt0 = stream_model.decoder.de_convs[0]
            x1_sfe = gt0.sfe(x1)
            h1 = gt0.point_act(gt0.point_bn1(gt0.point_conv1(x1_sfe)))

            # Get cache before depth_conv
            de_conv_cache_before = conv_cache[1, :, :, 6:16, :].sum().item()

            # Process depth_conv
            h1_dc, new_cache = gt0.depth_conv(h1, conv_cache[1, :, :, 6:16, :])
            conv_cache[1, :, :, 6:16, :] = new_cache

            # Continue through rest of decoder
            h1_dc = gt0.depth_act(gt0.depth_bn(h1_dc))
            h1_out = gt0.point_bn2(gt0.point_conv2(h1_dc))
            h1_tra, tra_cache[1, 0] = gt0.tra(h1_out, tra_cache[1, 0])
            x = gt0.shuffle(h1_tra, x2)

            # Continue with DeGT1 and DeGT2 (just update caches)
            gt1 = stream_model.decoder.de_convs[1]
            x = x + en_outs[3]
            x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]
            x1_sfe = gt1.sfe(x1)
            h1_temp = gt1.point_act(gt1.point_bn1(gt1.point_conv1(x1_sfe)))
            h1_dc, conv_cache[1, :, :, 2:6, :] = gt1.depth_conv(h1_temp, conv_cache[1, :, :, 2:6, :])
            h1_dc = gt1.depth_act(gt1.depth_bn(h1_dc))
            h1_out = gt1.point_bn2(gt1.point_conv2(h1_dc))
            h1_tra, tra_cache[1, 1] = gt1.tra(h1_out, tra_cache[1, 1])
            x = gt1.shuffle(h1_tra, x2)

            gt2 = stream_model.decoder.de_convs[2]
            x = x + en_outs[2]
            x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]
            x1_sfe = gt2.sfe(x1)
            h1_temp = gt2.point_act(gt2.point_bn1(gt2.point_conv1(x1_sfe)))
            h1_dc, conv_cache[1, :, :, :2, :] = gt2.depth_conv(h1_temp, conv_cache[1, :, :, :2, :])

            print(f"Frame {frame_idx}: h1 sum = {h1.sum().item():.2f}, cache before = {de_conv_cache_before:.2f}")

    print("\nExpected from C debug output:")
    print("  Frame 1: h1 sum=116.40, cache sum=0.00")
    print("  Frame 2: h1 sum=39.95, cache sum=116.40")
    print("  Frame 3: h1 sum=26.16, cache sum=156.35")
    print("  Frame 4: h1 sum=21.83, cache sum=182.50")
    print("  Frame 5: h1 sum=24.70, cache sum=204.34")
    print("  Frame 6: h1 sum=7.74, cache sum=229.03")

if __name__ == "__main__":
    main()
