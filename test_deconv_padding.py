# -*- coding: utf-8 -*-
"""
Test StreamConvTranspose2d padding logic
"""
import torch
import torch.nn as nn
from modules.convolution import StreamConvTranspose2d
from modules.convert import convert_to_stream

# Test parameters matching decoder conv3: (16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2)
print("=" * 60)
print("Testing StreamConvTranspose2d")
print("=" * 60)

# Create layers
deconv_orig = nn.ConvTranspose2d(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, bias=True)
deconv_stream = StreamConvTranspose2d(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, bias=True)

# Convert weights
convert_to_stream(deconv_stream, deconv_orig)

# Test input: single frame (16, 33)
torch.manual_seed(42)
x_single = torch.randn(1, 16, 1, 33)

print(f"\nInput shape: {x_single.shape}")
print(f"Kernel: (1, 5), Stride: (1, 2), Padding: (0, 2)")
print(f"Expected output freq: 33*2 = 66? Let's see...")

# Stream forward
cache = torch.zeros(1, 16, 0, 33)  # No cache for kernel_t=1
with torch.no_grad():
    y_stream, _ = deconv_stream(x_single, cache)

print(f"\nStream output shape: {y_stream.shape}")
print(f"Stream output sum: {y_stream.sum():.6f}")
print(f"Stream output range: [{y_stream.min():.6f}, {y_stream.max():.6f}]")

# Check padding calculation
print(f"\nPadding calculation:")
print(f"  F_size = 5, F_stride = 2, F_pad = 2")
print(f"  left_pad = F_stride - 1 = {2 - 1}")
print(f"  Left padding = (F_size-1)*dilation - F_pad = (5-1)*1 - 2 = {(5-1)*1 - 2}")
print(f"  Right padding = (F_size-1)*dilation - F_pad - left_pad = (5-1)*1 - 2 - 1 = {(5-1)*1 - 2 - 1}")
print(f"  Total padding = {(5-1)*1 - 2} + {(5-1)*1 - 2 - 1} = {(5-1)*1 - 2 + (5-1)*1 - 2 - 1}")

# Manually check what Python does
print(f"\nManual check of Python logic:")
inp = x_single
bs, C, T, F = inp.shape
F_stride = 2
F_size = 5
F_pad = 2
F_dilation = 1

# Upsampling
inp_up = torch.cat([inp[:,:,:,:,None], torch.zeros([bs,C,T,F,F_stride-1])], dim = -1).reshape([bs,C,T,-1])
print(f"  After upsampling: {inp_up.shape}")

left_pad = F_stride - 1
pad_left = (F_size - 1)*F_dilation - F_pad
pad_right = (F_size - 1)*F_dilation - F_pad - left_pad
inp_padded = torch.nn.functional.pad(inp_up, pad = [pad_left, pad_right, 0, 0])
print(f"  After padding [{pad_left}, {pad_right}]: {inp_padded.shape}")

# Apply Conv2d
y_manual = deconv_stream.ConvTranspose2d(inp_padded)
print(f"  After Conv2d: {y_manual.shape}")
print(f"  Manual sum: {y_manual.sum():.6f}")

print(f"\nDifference: {(y_stream - y_manual).abs().max():.6e}")
