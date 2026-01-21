"""
Debug C streaming by comparing intermediate outputs with Python streaming.
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

    # Process 5 frames to warm up, then capture frame 5
    print("Processing frames 0-5...")

    # Hook to capture intermediate outputs
    intermediate_outputs = {}

    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                intermediate_outputs[name] = output[0].detach().clone()
            else:
                intermediate_outputs[name] = output.detach().clone()
        return hook

    # Register hooks on key layers
    # Note: StreamGTCRN has encoder, dpgrnns, decoder

    with torch.no_grad():
        for i in range(6):
            frame = spec[:, :, i:i+1, :]

            if i == 5:
                # Manually trace through the model for frame 5
                # Input preparation
                x = frame  # (1, 257, 1, 2)

                # Create feature tensor (mag, real, imag)
                mag = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-12)  # (1, 257, 1)
                feat = torch.stack([mag, x[..., 0], x[..., 1]], dim=1)  # (1, 3, 257, 1)
                feat = feat.permute(0, 1, 3, 2)  # (1, 3, 1, 257)

                print(f"\nFrame 5 Feature tensor:")
                print(f"  Shape: {feat.shape}")
                print(f"  Sum (mag): {feat[0, 0].sum().item():.6f}")
                print(f"  Sum (real): {feat[0, 1].sum().item():.6f}")
                print(f"  Sum (imag): {feat[0, 2].sum().item():.6f}")

                # ERB compression
                erb = stream_model.erb
                feat_erb = erb.bm(feat)  # (1, 3, 1, 129)
                print(f"\nAfter ERB compression:")
                print(f"  Shape: {feat_erb.shape}")
                print(f"  Sum: {feat_erb.sum().item():.6f}")

                # SFE
                sfe = stream_model.sfe
                feat_sfe = sfe(feat_erb)  # (1, 9, 1, 129)
                print(f"\nAfter SFE:")
                print(f"  Shape: {feat_sfe.shape}")
                print(f"  Sum: {feat_sfe.sum().item():.6f}")

                # Encoder conv 0
                en_out0 = stream_model.encoder.en_convs[0](feat_sfe)  # (1, 16, 1, 65)
                print(f"\nAfter EnConv0:")
                print(f"  Shape: {en_out0.shape}")
                print(f"  Sum: {en_out0.sum().item():.6f}")

                # Encoder conv 1
                # Trace step-by-step
                en_conv1_block = stream_model.encoder.en_convs[1]
                # Conv only
                x = en_conv1_block.conv(en_out0)
                print(f"\nEnConv1 after conv:")
                print(f"  Sum: {x.sum().item():.6f}")
                # BN
                x = en_conv1_block.bn(x)
                print(f"EnConv1 after BN:")
                print(f"  Sum: {x.sum().item():.6f}")
                # Activation (PReLU or Tanh)
                if hasattr(en_conv1_block, 'act'):
                    x = en_conv1_block.act(x)
                print(f"EnConv1 after activation:")
                print(f"  Sum: {x.sum().item():.6f}")

                en_out1 = x  # Use the traced output
                print(f"\nAfter EnConv1:")
                print(f"  Shape: {en_out1.shape}")
                print(f"  Sum: {en_out1.sum().item():.6f}")

            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )

            if i == 5:
                out_real = out_frame[0, :, 0, 0].numpy()
                out_imag = out_frame[0, :, 0, 1].numpy()
                out_mag = np.sqrt(out_real**2 + out_imag**2)
                print(f"\nFrame 5 final output:")
                print(f"  Real sum: {np.sum(out_real):.6f}")
                print(f"  Imag sum: {np.sum(out_imag):.6f}")
                print(f"  Mag sum: {np.sum(out_mag):.6f}")

if __name__ == "__main__":
    main()
