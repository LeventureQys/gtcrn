# -*- coding: utf-8 -*-
"""
Compare first frame outputs to find where divergence starts
"""
import numpy as np
import torch
import soundfile as sf
from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
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
x = torch.from_numpy(audio_data[:512])
spec = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]

# Get first frame
frame0 = spec[:, :, 0:1]

# Initialize caches
conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
inter_cache = torch.zeros(2, 1, 33, 16).to(device)

# Process with streaming model
with torch.no_grad():
    y_stream, _, _, _ = stream_model(frame0, conv_cache, tra_cache, inter_cache)

# Process with complete model (single frame)
with torch.no_grad():
    y_complete = model(frame0)

print("=" * 60)
print("First Frame Comparison")
print("=" * 60)
print(f"\nStreaming model output:")
print(f"  Shape: {y_stream.shape}")
print(f"  Sum: {y_stream.sum():.6f}")
print(f"  Real range: [{y_stream[0,:,0,0].min():.6f}, {y_stream[0,:,0,0].max():.6f}]")
print(f"  Imag range: [{y_stream[0,:,0,1].min():.6f}, {y_stream[0,:,0,1].max():.6f}]")

print(f"\nComplete model output:")
print(f"  Shape: {y_complete.shape}")
print(f"  Sum: {y_complete.sum():.6f}")
print(f"  Real range: [{y_complete[0,:,0,0].min():.6f}, {y_complete[0,:,0,0].max():.6f}]")
print(f"  Imag range: [{y_complete[0,:,0,1].min():.6f}, {y_complete[0,:,0,1].max():.6f}]")

# Compare
diff = (y_stream - y_complete).abs()
print(f"\nDifference:")
print(f"  Max error: {diff.max():.6e}")
print(f"  Mean error: {diff.mean():.6e}")

if diff.max() < 1e-5:
    print("\n✅ Streaming and complete models match for first frame!")
else:
    print(f"\n❌ Models differ! Max error: {diff.max():.6e}")
