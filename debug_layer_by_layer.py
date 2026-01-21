# -*- coding: utf-8 -*-
"""
Debug: Layer by layer comparison between Python and C streaming
Export Python intermediate outputs for each layer
"""
import numpy as np
import torch
import soundfile as sf
import os
from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream

# Create output directory
os.makedirs('test_wavs/debug_layers', exist_ok=True)

# Load model
device = torch.device("cpu")
model = GTCRN().to(device).eval()
model.load_state_dict(torch.load("checkpoints/model_trained_on_dns3.tar", map_location=device)['model'])

# Convert to streaming model
stream_model = StreamGTCRN().to(device).eval()
convert_to_stream(stream_model, model)

# Load test audio
audio_data, sr = sf.read('test_wavs/noisy_16k/00001_1_fan_noise_level1_snr-5dB_noisy.wav', dtype='float32')
assert sr == 16000

# Create STFT for first frame
window = torch.hann_window(512).pow(0.5)
x = torch.from_numpy(audio_data[:512])  # First frame
spec = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]  # (1, 257, T, 2)

print("=" * 60)
print("Layer-by-layer Python streaming analysis")
print("=" * 60)
print(f"Spec shape: {spec.shape}")

# Get first frame
frame0 = spec[:, :, 0:1]  # (1, 257, 1, 2)
print(f"Frame 0 shape: {frame0.shape}")

# Initialize caches
conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
inter_cache = torch.zeros(2, 1, 33, 16).to(device)

# Process step by step with intermediate outputs
with torch.no_grad():
    # Step 1: Extract features
    spec_real = frame0[..., 0].permute(0, 2, 1)  # (1, 1, 257)
    spec_imag = frame0[..., 1].permute(0, 2, 1)  # (1, 1, 257)
    spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
    feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (1, 3, 1, 257)

    print(f"\n[Layer 0] Input features (3, 257):")
    print(f"  feat sum: {feat.sum():.6f}")
    feat_np = feat[0, :, 0, :].numpy()
    np.save('test_wavs/debug_layers/py_feat.npy', feat_np)
    print(f"  Saved: py_feat.npy, shape={feat_np.shape}")

    # Step 2: ERB compression
    feat_erb = stream_model.erb.bm(feat)  # (1, 3, 1, 129)
    print(f"\n[Layer 1] After ERB compression (3, 129):")
    print(f"  feat_erb sum: {feat_erb.sum():.6f}")
    feat_erb_np = feat_erb[0, :, 0, :].numpy()
    np.save('test_wavs/debug_layers/py_erb.npy', feat_erb_np)
    print(f"  Saved: py_erb.npy, shape={feat_erb_np.shape}")

    # Step 3: SFE
    feat_sfe = stream_model.sfe(feat_erb)  # (1, 9, 1, 129)
    print(f"\n[Layer 2] After SFE (9, 129):")
    print(f"  feat_sfe sum: {feat_sfe.sum():.6f}")
    feat_sfe_np = feat_sfe[0, :, 0, :].numpy()
    np.save('test_wavs/debug_layers/py_sfe.npy', feat_sfe_np)
    print(f"  Saved: py_sfe.npy, shape={feat_sfe_np.shape}")

    # Step 4: Encoder ConvBlock 0
    en_conv0 = stream_model.encoder.en_convs[0]
    x_conv0 = en_conv0.conv(feat_sfe)  # Conv2d
    x_bn0 = en_conv0.bn(x_conv0)
    x_act0 = en_conv0.act(x_bn0)  # PReLU
    print(f"\n[Layer 3] After EnConv0 (16, 65):")
    print(f"  after conv: {x_conv0.sum():.6f}")
    print(f"  after bn: {x_bn0.sum():.6f}")
    print(f"  after prelu: {x_act0.sum():.6f}")
    en0_np = x_act0[0, :, 0, :].numpy()
    np.save('test_wavs/debug_layers/py_enconv0.npy', en0_np)
    print(f"  Saved: py_enconv0.npy, shape={en0_np.shape}")

    # Step 5: Encoder ConvBlock 1
    en_conv1 = stream_model.encoder.en_convs[1]
    x_conv1 = en_conv1.conv(x_act0)  # Conv2d with groups=2
    x_bn1 = en_conv1.bn(x_conv1)
    x_act1 = en_conv1.act(x_bn1)
    print(f"\n[Layer 4] After EnConv1 (16, 33):")
    print(f"  after conv: {x_conv1.sum():.6f}")
    print(f"  after bn: {x_bn1.sum():.6f}")
    print(f"  after prelu: {x_act1.sum():.6f}")
    en1_np = x_act1[0, :, 0, :].numpy()
    np.save('test_wavs/debug_layers/py_enconv1.npy', en1_np)
    print(f"  Saved: py_enconv1.npy, shape={en1_np.shape}")

    # Step 6: Full forward pass to get final output
    y0, new_conv_cache, new_tra_cache, new_inter_cache = stream_model(
        frame0, conv_cache.clone(), tra_cache.clone(), inter_cache.clone()
    )
    print(f"\n[Final] Output frame (257, 1, 2):")
    y0_np = y0[0, :, 0, :].numpy()
    print(f"  y0 sum: {y0.sum():.6f}")
    print(f"  y0 real range: [{y0_np[:, 0].min():.6f}, {y0_np[:, 0].max():.6f}]")
    print(f"  y0 imag range: [{y0_np[:, 1].min():.6f}, {y0_np[:, 1].max():.6f}]")
    np.save('test_wavs/debug_layers/py_output.npy', y0_np)
    print(f"  Saved: py_output.npy, shape={y0_np.shape}")

    # Save input spec for C comparison
    spec_input = frame0[0, :, 0, :].numpy()  # (257, 2)
    np.save('test_wavs/debug_layers/py_spec_input.npy', spec_input)
    print(f"\n  Saved input spec: py_spec_input.npy, shape={spec_input.shape}")

print("\n" + "=" * 60)
print("All intermediate outputs saved to test_wavs/debug_layers/")
print("Use these to compare with C implementation layer by layer")
print("=" * 60)
