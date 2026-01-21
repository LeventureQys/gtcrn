"""
Debug cache sum at each frame in Python.
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
    print("=== Debug Cache Sum Per Frame ===\n")

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

    # We need to manually trace through the decoder to get h1
    print("\nManual trace of DeGT0 at each frame:")
    print("(h1 = input to depth_conv, after point_conv1+bn+prelu)")

    with torch.no_grad():
        for i in range(10):
            frame = spec[:, :, i:i+1, :]

            # Manual forward pass
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

            # DeGT0 input
            x = feat + en_outs[4]
            x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]

            # SFE and PointConv1
            gt0 = stream_model.decoder.de_convs[0]
            x1_sfe = gt0.sfe(x1)
            h1 = gt0.point_act(gt0.point_bn1(gt0.point_conv1(x1_sfe)))

            # Get cache before
            de_gt0_cache_before = conv_cache[1, :, :, 6:16, :].clone()

            # Depth conv with cache
            h1_dc, new_dc_cache = gt0.depth_conv(h1, conv_cache[1, :, :, 6:16, :])
            conv_cache[1, :, :, 6:16, :] = new_dc_cache

            print(f"  Frame {i+1}: h1 sum={h1.sum().item():.2f}, cache before={de_gt0_cache_before.sum().item():.2f}, cache after={new_dc_cache.sum().item():.2f}")

            # Complete the rest of decoder (we don't care about output, just cache updates)
            h1_dc = gt0.depth_act(gt0.depth_bn(h1_dc))
            h1_out = gt0.point_bn2(gt0.point_conv2(h1_dc))
            h1_out, tra_cache[1, 0] = gt0.tra(h1_out, tra_cache[1, 0])
            x = gt0.shuffle(h1_out, x2)

            # Continue with rest of decoder (DeGT1, DeGT2) to update their caches
            gt1 = stream_model.decoder.de_convs[1]
            x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]
            x1_sfe = gt1.sfe(x1)
            h1 = gt1.point_act(gt1.point_bn1(gt1.point_conv1(x1_sfe)))
            h1_dc, conv_cache[1, :, :, 2:6, :] = gt1.depth_conv(h1, conv_cache[1, :, :, 2:6, :])
            h1_dc = gt1.depth_act(gt1.depth_bn(h1_dc))
            h1_out = gt1.point_bn2(gt1.point_conv2(h1_dc))
            h1_out, tra_cache[1, 1] = gt1.tra(h1_out, tra_cache[1, 1])
            x = gt1.shuffle(h1_out, x2) + en_outs[2]

            gt2 = stream_model.decoder.de_convs[2]
            x1, x2 = x[:,:x.shape[1]//2], x[:, x.shape[1]//2:]
            x1_sfe = gt2.sfe(x1)
            h1 = gt2.point_act(gt2.point_bn1(gt2.point_conv1(x1_sfe)))
            h1_dc, conv_cache[1, :, :, :2, :] = gt2.depth_conv(h1, conv_cache[1, :, :, :2, :])

if __name__ == "__main__":
    main()
