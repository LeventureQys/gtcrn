# -*- coding: utf-8 -*-
"""
Verify Python StreamConvTranspose2d implementation details
"""
import torch
import torch.nn as nn
from modules.convolution import StreamConvTranspose2d
from modules.convert import convert_to_stream

# Create a StreamConvTranspose2d
deconv_stream = StreamConvTranspose2d(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, bias=True)

print("=" * 60)
print("StreamConvTranspose2d Implementation Analysis")
print("=" * 60)

print(f"\nInternal Conv2d (named 'ConvTranspose2d'):")
print(f"  Type: {type(deconv_stream.ConvTranspose2d)}")
print(f"  Kernel size: {deconv_stream.ConvTranspose2d.kernel_size}")
print(f"  Stride: {deconv_stream.ConvTranspose2d.stride}")
print(f"  Padding: {deconv_stream.ConvTranspose2d.padding}")
print(f"  Dilation: {deconv_stream.ConvTranspose2d.dilation}")

print(f"\nKey insight:")
print(f"  - Python uses Conv2d (not ConvTranspose2d)")
print(f"  - Stride in Conv2d: (1, 1) - no striding in Conv2d!")
print(f"  - Padding in Conv2d: (0, 0) - no padding in Conv2d!")
print(f"  - Upsampling and padding done manually in forward()")

print(f"\nFor kernel=(1,5), stride=(1,2), pad=(0,2):")
print(f"  1. Upsample: 33 -> 66 (insert zeros)")
print(f"  2. Pad: 66 -> 69 (left=2, right=1)")
print(f"  3. Conv2d with kernel=(1,5), stride=(1,1), pad=(0,0)")
print(f"  4. Output: 69 - 5 + 1 = 65")
