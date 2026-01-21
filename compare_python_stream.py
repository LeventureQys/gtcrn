# -*- coding: utf-8 -*-
"""
Compare Python streaming and complete inference outputs
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
print("Python Stream vs Complete Inference Comparison")
print("=" * 50)
print(f"Spec shape: {spec.shape}")

# Complete inference
with torch.no_grad():
    y_complete = model(spec)  # (1, F, T, 2)

# Streaming inference
conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
inter_cache = torch.zeros(2, 1, 33, 16).to(device)

ys = []
with torch.no_grad():
    for i in range(spec.shape[2]):
        xi = spec[:, :, i:i+1]
        yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache)
        ys.append(yi)
y_stream = torch.cat(ys, dim=2)

# Convert to audio
y_complete_complex = torch.complex(y_complete[0, :, :, 0], y_complete[0, :, :, 1])
y_stream_complex = torch.complex(y_stream[0, :, :, 0], y_stream[0, :, :, 1])

audio_complete = torch.istft(y_complete_complex, 512, 256, 512, window).detach().cpu().numpy()
audio_stream = torch.istft(y_stream_complex, 512, 256, 512, window).detach().cpu().numpy()

# Compare
diff = np.abs(audio_stream - audio_complete)
max_error = diff.max()
mean_error = diff.mean()

stream_energy = np.sqrt((audio_stream ** 2).mean())
complete_energy = np.sqrt((audio_complete ** 2).mean())
energy_ratio = stream_energy / complete_energy

correlation = np.corrcoef(audio_stream, audio_complete)[0, 1]

print(f"\nPython Error Analysis:")
print(f"  Max absolute error: {max_error:.8f}")
print(f"  Mean absolute error: {mean_error:.8f}")
print(f"  Energy ratio: {energy_ratio:.4f}")
print(f"  Correlation: {correlation:.6f}")

# Save for comparison
sf.write('test_wavs/output_c/py_stream_test.wav', audio_stream, 16000)
sf.write('test_wavs/output_c/py_complete_test.wav', audio_complete, 16000)
print("\nSaved: py_stream_test.wav and py_complete_test.wav")

print("\n" + "=" * 50)
if max_error < 1e-5:
    print("Python Assessment: [OK] Stream == Complete (numerical precision)")
else:
    print(f"Python Assessment: Max error = {max_error:.2e}")
print("=" * 50)
