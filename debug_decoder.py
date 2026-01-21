"""
Debug decoder GTConvBlock to trace intermediate values.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN, StreamDecoder
from modules.convert import convert_to_stream

def main():
    print("=== Debug Decoder GTConvBlock 0 ===\n")

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
    n_frames = spec.shape[2]
    print(f"Number of frames: {n_frames}")

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process frames and manually trace DeGT0 on frame 5
    target_frame = 5  # Frame 5 in Python = Frame 6 in C (1-indexed)

    # First process frames 0-4 to build up state
    with torch.no_grad():
        for i in range(target_frame):
            frame = spec[:, :, i:i+1, :]
            _, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )

        # Now manually trace frame 5
        frame = spec[:, :, target_frame:target_frame+1, :]
        print(f"\n=== Frame {target_frame} (C Frame {target_frame+1}) ===")

        # Replicate the forward pass manually to trace intermediate values
        spec_ref = frame
        spec_real = frame[..., 0].permute(0,2,1)
        spec_imag = frame[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)

        feat = stream_model.erb.bm(feat)
        feat = stream_model.sfe(feat)

        feat, en_outs, conv_cache[0], tra_cache[0] = stream_model.encoder(feat, conv_cache[0], tra_cache[0])

        feat, inter_cache[0] = stream_model.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = stream_model.dpgrnn2(feat, inter_cache[1])

        print(f"\nDPGRNN2 output (input to decoder):")
        print(f"  shape: {feat.shape}")  # (1, 16, 1, 33)
        t = feat.squeeze()  # (16, 33)
        print(f"  sum: {t.sum().item():.6f}")
        print(f"  first 10 of ch0: {t[0, :10].tolist()}")

        print(f"\nen_out4 (skip connection for DeGT0):")
        t4 = en_outs[4].squeeze()  # (16, 33)
        print(f"  sum: {t4.sum().item():.6f}")
        print(f"  first 10 of ch0: {t4[0, :10].tolist()}")

        # Input to DeGT0 after skip connection
        x = feat + en_outs[4]
        print(f"\nDeGT0 input (after skip):")
        t_in = x.squeeze()
        print(f"  sum: {t_in.sum().item():.6f}")
        print(f"  first 10 of ch0: {t_in[0, :10].tolist()}")

        # Now trace inside DeGT0
        gt0 = stream_model.decoder.de_convs[0]
        de_conv_cache = conv_cache[1, :, :, 6:16, :]  # (1, 16, 10, 33)
        de_tra_cache = tra_cache[1, 0]  # (1, 1, 16)

        x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]  # (1, 8, 1, 33), (1, 8, 1, 33)
        print(f"\nAfter split:")
        print(f"  x1 sum: {x1.sum().item():.6f}")
        print(f"  x2 sum: {x2.sum().item():.6f}")

        # SFE
        x1_sfe = gt0.sfe(x1)  # (1, 24, 1, 33)
        print(f"\nAfter SFE on x1:")
        print(f"  shape: {x1_sfe.shape}")
        print(f"  sum: {x1_sfe.sum().item():.6f}")

        # Point conv 1
        h1 = gt0.point_act(gt0.point_bn1(gt0.point_conv1(x1_sfe)))  # (1, 16, 1, 33)
        print(f"\nAfter point_conv1 + bn + prelu:")
        print(f"  shape: {h1.shape}")
        print(f"  sum: {h1.sum().item():.6f}")
        print(f"  first 10 of ch0: {h1[0, 0, 0, :10].tolist()}")

        # Depth conv - this is the key step!
        print(f"\nBefore depth_conv:")
        print(f"  h1 sum: {h1.sum().item():.6f}")
        print(f"  de_conv_cache shape: {de_conv_cache.shape}")
        print(f"  de_conv_cache sum: {de_conv_cache.sum().item():.6f}")

        h1_dc, new_cache = gt0.depth_conv(h1, de_conv_cache)
        print(f"\nAfter depth_conv:")
        print(f"  shape: {h1_dc.shape}")
        print(f"  sum: {h1_dc.sum().item():.6f}")
        print(f"  first 10 of ch0: {h1_dc[0, 0, 0, :10].tolist()}")

        h1_dc = gt0.depth_act(gt0.depth_bn(h1_dc))
        print(f"\nAfter depth_bn + prelu:")
        print(f"  sum: {h1_dc.sum().item():.6f}")

        # Point conv 2
        h1_out = gt0.point_bn2(gt0.point_conv2(h1_dc))
        print(f"\nAfter point_conv2 + bn:")
        print(f"  sum: {h1_out.sum().item():.6f}")

        # TRA
        h1_tra, _ = gt0.tra(h1_out, de_tra_cache)
        print(f"\nAfter TRA:")
        print(f"  sum: {h1_tra.sum().item():.6f}")

        # Shuffle
        x_out = gt0.shuffle(h1_tra, x2)
        print(f"\nAfter shuffle (DeGT0 output):")
        print(f"  shape: {x_out.shape}")
        print(f"  sum: {x_out.sum().item():.6f}")


if __name__ == "__main__":
    main()
