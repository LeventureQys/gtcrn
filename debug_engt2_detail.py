"""
Debug EnGT2 intermediate values in detail to compare with C implementation.

This traces all intermediate outputs within the first GTConvBlock (EnGT2):
1. Input split (x1, x2)
2. SFE output (x1_sfe)
3. After PointConv1 + BN + PReLU (h1)
4. After DepthConv + BN + PReLU (h1)
5. After PointConv2 + BN (h1)
6. After TRA (h1)
7. After shuffle (output)
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
from gtcrn_stream import StreamGTCRN, StreamGTConvBlock
from modules.convert import convert_to_stream

def abs_sum(x):
    """Return sum of absolute values"""
    return float(x.abs().sum())

def main():
    print("=== Debug EnGT2 Detail ===\n")

    # Load model
    model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])

    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # Load audio
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)
    print(f"Audio: {len(audio)} samples, {sr} Hz\n")

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio with 256 zeros (matches Python streaming)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # Compute STFT
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False,
                      center=False)
    spec = spec.unsqueeze(0)  # (1, 257, T, 2)

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process frames 0-5 to get to frame 5 (which matches C frame_count=6)
    target_frame = 5  # Python frame index (C frame_count = 6)

    with torch.no_grad():
        for i in range(target_frame):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )

    # Now process frame 5 with detailed tracing
    print(f"=== Tracing Python Frame {target_frame} (C frame_count={target_frame+1}) ===\n")

    frame = spec[:, :, target_frame:target_frame+1, :]

    # Manually trace through the model
    with torch.no_grad():
        # 1. Features extraction (mag, real, imag)
        mag = torch.sqrt(frame[..., 0]**2 + frame[..., 1]**2 + 1e-12)
        rea = frame[..., 0]
        imag = frame[..., 1]
        x = torch.stack([mag, rea, imag], dim=1)  # (1, 3, 257, 1)
        x = x.transpose(2, 3)  # (1, 3, 1, 257)
        print(f"Features (3, 1, 257) abs_sum: {abs_sum(x):.6f}")

        # 2. ERB compression
        erb_x = stream_model.erb.bm(x)
        print(f"After ERB (3, 1, 129) abs_sum: {abs_sum(erb_x):.6f}")

        # 3. SFE
        sfe_x = stream_model.sfe(erb_x)
        print(f"After SFE (9, 1, 129) abs_sum: {abs_sum(sfe_x):.6f}")

        # 4. EnConv0 (no cache - ConvBlock without streaming)
        x = stream_model.encoder.en_convs[0](sfe_x)
        print(f"After EnConv0 (16, 1, 65) abs_sum: {abs_sum(x):.6f}")

        # 5. EnConv1 (no cache - stride conv)
        x = stream_model.encoder.en_convs[1](x)
        print(f"After EnConv1 (16, 1, 33) abs_sum: {abs_sum(x):.6f}")

        # Save as input to EnGT2
        en_gt2_input = x.clone()

        # 6. EnGT2 - trace in detail
        print("\n--- EnGT2 Detail ---")
        en_gt2 = stream_model.encoder.en_convs[2]

        # Get caches
        enc_conv_cache = conv_cache[0]  # (1, 16, 16, 33)
        engt2_conv_cache = enc_conv_cache[:, :, :2, :]  # (1, 16, 2, 33)
        engt2_tra_cache = tra_cache[0, 0:1]  # (1, 1, 16)

        # Split channels
        x1 = en_gt2_input[:, :8]  # (1, 8, 1, 33)
        x2 = en_gt2_input[:, 8:]  # (1, 8, 1, 33)
        print(f"  x1 (first 8 ch) abs_sum: {abs_sum(x1):.6f}")
        print(f"  x2 (last 8 ch) abs_sum: {abs_sum(x2):.6f}")

        # SFE on x1
        x1_sfe = en_gt2.sfe(x1)
        print(f"  After SFE (24, 1, 33) abs_sum: {abs_sum(x1_sfe):.6f}")

        # Print x1_sfe values
        print(f"    x1_sfe first 10 vals: {x1_sfe.flatten()[:10].tolist()}")

        # PointConv1 + BN + PReLU
        h1 = en_gt2.point_conv1(x1_sfe)
        print(f"  After PointConv1 (16, 1, 33) abs_sum: {abs_sum(h1):.6f}")

        h1_bn1 = en_gt2.point_bn1(h1)
        print(f"  After PointBN1 abs_sum: {abs_sum(h1_bn1):.6f}")

        h1_act = en_gt2.point_act(h1_bn1)
        print(f"  After PReLU1 abs_sum: {abs_sum(h1_act):.6f}")

        # Print h1_act values for comparison
        print(f"    h1 after PReLU1 first 10 vals: {h1_act.flatten()[:10].tolist()}")

        # DepthConv (with cache)
        h1_dc, new_conv_cache = en_gt2.depth_conv(h1_act, engt2_conv_cache)
        print(f"  After DepthConv (16, 1, 33) abs_sum: {abs_sum(h1_dc):.6f}")

        # Print cache and h1_dc values
        print(f"    conv_cache shape: {engt2_conv_cache.shape}")
        print(f"    new_conv_cache first 10 vals: {new_conv_cache.flatten()[:10].tolist()}")
        print(f"    h1_dc first 10 vals: {h1_dc.flatten()[:10].tolist()}")

        h1_dc_bn = en_gt2.depth_bn(h1_dc)
        print(f"  After DepthBN abs_sum: {abs_sum(h1_dc_bn):.6f}")

        h1_dc_act = en_gt2.depth_act(h1_dc_bn)
        print(f"  After PReLU2 abs_sum: {abs_sum(h1_dc_act):.6f}")

        # PointConv2 + BN
        h1_pc2 = en_gt2.point_conv2(h1_dc_act)
        print(f"  After PointConv2 (8, 1, 33) abs_sum: {abs_sum(h1_pc2):.6f}")

        h1_pc2_bn = en_gt2.point_bn2(h1_pc2)
        print(f"  After PointBN2 abs_sum: {abs_sum(h1_pc2_bn):.6f}")

        # TRA
        h1_tra, new_tra_cache = en_gt2.tra(h1_pc2_bn, engt2_tra_cache)
        print(f"  After TRA (8, 1, 33) abs_sum: {abs_sum(h1_tra):.6f}")

        # Print TRA values
        print(f"    TRA output first 10 vals: {h1_tra.flatten()[:10].tolist()}")

        # Shuffle
        output = en_gt2.shuffle(h1_tra, x2)
        print(f"  After Shuffle (16, 1, 33) abs_sum: {abs_sum(output):.6f}")

        # Print channel sums
        print("\n  Per-channel sums:")
        for c in range(16):
            ch_sum = output[0, c].sum().item()
            print(f"    ch{c}: {ch_sum:.6f}")

        print(f"\n  Final EnGT2 output abs_sum: {abs_sum(output):.6f}")

        # Compare with expected from previous run
        print("\n=== Comparison ===")
        print("  Python EnGT2 abs_sum: 114.750290 (expected)")
        print("  C EnGT2 abs_sum: 106.983348 (from debug)")
        print(f"  This run EnGT2 abs_sum: {abs_sum(output):.6f}")

if __name__ == "__main__":
    main()
