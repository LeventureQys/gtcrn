# -*- coding: utf-8 -*-
"""
Debug: Compare C stream output frame by frame with Python
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
assert sr == 16000

# Create STFT
window = torch.hann_window(512).pow(0.5)
x = torch.from_numpy(audio_data)
spec = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]  # (1, F, T, 2)

print("=" * 50)
print("Frame-by-frame streaming analysis")
print("=" * 50)
print(f"Spec shape: {spec.shape}")
print(f"Total frames: {spec.shape[2]}")

# Initialize caches
conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
inter_cache = torch.zeros(2, 1, 33, 16).to(device)

# Process first few frames and check outputs
print("\nProcessing first 10 frames:")
ys = []
with torch.no_grad():
    for i in range(min(10, spec.shape[2])):
        xi = spec[:, :, i:i+1]
        yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache)
        ys.append(yi)

        # Check output statistics
        yi_real = yi[0, :, 0, 0].numpy()
        yi_imag = yi[0, :, 0, 1].numpy()
        yi_mag = np.sqrt(yi_real**2 + yi_imag**2)

        print(f"  Frame {i:3d}: output mag min={yi_mag.min():.4f}, max={yi_mag.max():.4f}, mean={yi_mag.mean():.4f}")

# Check cache states
print("\nCache statistics after 10 frames:")
print(f"  conv_cache: min={conv_cache.min():.4f}, max={conv_cache.max():.4f}, mean={conv_cache.mean():.4f}")
print(f"  tra_cache: min={tra_cache.min():.4f}, max={tra_cache.max():.4f}, mean={tra_cache.mean():.4f}")
print(f"  inter_cache: min={inter_cache.min():.4f}, max={inter_cache.max():.4f}, mean={inter_cache.mean():.4f}")

# Save cache values for comparison with C
np.save('test_wavs/output_c/py_conv_cache_10.npy', conv_cache.numpy())
np.save('test_wavs/output_c/py_tra_cache_10.npy', tra_cache.numpy())
np.save('test_wavs/output_c/py_inter_cache_10.npy', inter_cache.numpy())
print("\nSaved cache states to test_wavs/output_c/py_*_cache_10.npy")

# Process remaining frames
print(f"\nProcessing remaining {spec.shape[2] - 10} frames...")
with torch.no_grad():
    for i in range(10, spec.shape[2]):
        xi = spec[:, :, i:i+1]
        yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache)
        ys.append(yi)

y_stream = torch.cat(ys, dim=2)
y_stream_complex = torch.complex(y_stream[0, :, :, 0], y_stream[0, :, :, 1])
audio_stream = torch.istft(y_stream_complex, 512, 256, 512, window).detach().cpu().numpy()

# Load C stream output
c_stream, _ = sf.read('test_wavs/output_c/stream_fixed_test.wav')

min_len = min(len(c_stream), len(audio_stream))
diff = np.abs(c_stream[:min_len] - audio_stream[:min_len])

print(f"\nPython stream vs C stream:")
print(f"  Max error: {diff.max():.6f}")
print(f"  Mean error: {diff.mean():.6f}")
print(f"  Correlation: {np.corrcoef(c_stream[:min_len], audio_stream[:min_len])[0,1]:.6f}")

# Compare frame by frame in time domain
frame_errors = []
hop = 256
for i in range(min(len(c_stream), len(audio_stream)) // hop):
    start = i * hop
    end = start + hop
    frame_diff = np.abs(c_stream[start:end] - audio_stream[start:end]).max()
    frame_errors.append(frame_diff)

frame_errors = np.array(frame_errors)
print(f"\nPer-frame analysis:")
print(f"  Frames with max error > 0.1: {np.sum(frame_errors > 0.1)} / {len(frame_errors)}")
print(f"  Frames with max error > 0.01: {np.sum(frame_errors > 0.01)} / {len(frame_errors)}")

# Find first frame with large error
large_error_frames = np.where(frame_errors > 0.01)[0]
if len(large_error_frames) > 0:
    print(f"  First large error at frame: {large_error_frames[0]}")
    print(f"  Error value: {frame_errors[large_error_frames[0]]:.4f}")
