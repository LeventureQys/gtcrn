# -*- coding: utf-8 -*-
"""
Debug: Test first frame only
"""
import numpy as np
import torch
import struct
import soundfile as sf
from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN, ERB, SFE
from modules.convert import convert_to_stream

# Load model
device = torch.device("cpu")
model = GTCRN().to(device).eval()
model.load_state_dict(torch.load("checkpoints/model_trained_on_dns3.tar", map_location=device)['model'])

# Convert to streaming model
stream_model = StreamGTCRN().to(device).eval()
convert_to_stream(stream_model, model)

# Load test audio
audio_data, sr = sf.read('test_wavs/noisy_16k/00001_1_fan_noise_level1_snr-5dB_noisy.wav', dtype='float32')

# Create STFT for first frame
window = torch.hann_window(512).pow(0.5)
x = torch.from_numpy(audio_data[:512])  # Just first frame + overlap
spec = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]  # (1, 257, T, 2)

print("=" * 50)
print("First Frame Analysis")
print("=" * 50)
print(f"Spec shape: {spec.shape}")

# Get first frame
frame0 = spec[:, :, 0:1]  # (1, 257, 1, 2)
print(f"Frame 0 shape: {frame0.shape}")

# Save frame0 for comparison
frame0_real = frame0[0, :, 0, 0].numpy()
frame0_imag = frame0[0, :, 0, 1].numpy()
print(f"Frame 0 real: min={frame0_real.min():.6f}, max={frame0_real.max():.6f}")
print(f"Frame 0 imag: min={frame0_imag.min():.6f}, max={frame0_imag.max():.6f}")

# Initialize caches (all zeros)
conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
inter_cache = torch.zeros(2, 1, 33, 16).to(device)

# Process first frame with Python
with torch.no_grad():
    y0, _, _, _ = stream_model(frame0, conv_cache.clone(), tra_cache.clone(), inter_cache.clone())

y0_real = y0[0, :, 0, 0].numpy()
y0_imag = y0[0, :, 0, 1].numpy()
y0_mag = np.sqrt(y0_real**2 + y0_imag**2)

print(f"\nPython output frame 0:")
print(f"  Real: min={y0_real.min():.6f}, max={y0_real.max():.6f}")
print(f"  Imag: min={y0_imag.min():.6f}, max={y0_imag.max():.6f}")
print(f"  Mag: min={y0_mag.min():.6f}, max={y0_mag.max():.6f}, mean={y0_mag.mean():.6f}")

# Save Python intermediate values for comparison
# Process step by step to see intermediate values
with torch.no_grad():
    spec_ref = frame0  # (B,F,T,2)
    spec_real = frame0[..., 0].permute(0,2,1)  # (1, 1, 257)
    spec_imag = frame0[..., 1].permute(0,2,1)
    spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
    feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (1, 3, 1, 257)

    print(f"\nIntermediate values (Python):")
    print(f"  feat (3, 1, 257): sum={feat.sum():.6f}")

    # ERB compression
    feat_erb = stream_model.erb.bm(feat)  # (1, 3, 1, 129)
    print(f"  feat_erb (3, 1, 129): sum={feat_erb.sum():.6f}")

    # SFE
    feat_sfe = stream_model.sfe(feat_erb)  # (1, 9, 1, 129)
    print(f"  feat_sfe (9, 1, 129): sum={feat_sfe.sum():.6f}")

# Save Python output to binary for C comparison
np.array(y0_real, dtype=np.float32).tofile('test_wavs/output_c/py_frame0_real.bin')
np.array(y0_imag, dtype=np.float32).tofile('test_wavs/output_c/py_frame0_imag.bin')
print("\nSaved py_frame0_real.bin and py_frame0_imag.bin")

# Also save input frame
np.array(frame0_real, dtype=np.float32).tofile('test_wavs/output_c/input_frame0_real.bin')
np.array(frame0_imag, dtype=np.float32).tofile('test_wavs/output_c/input_frame0_imag.bin')
print("Saved input_frame0_real.bin and input_frame0_imag.bin")
