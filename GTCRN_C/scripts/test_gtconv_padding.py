#!/usr/bin/env python3
"""
Test GTConvBlock padding behavior for Encoder vs Decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def test_encoder_depth_conv():
    """Test Conv2d with manual causal padding."""
    print("=== Encoder GTConvBlock Depth Conv ===")

    # Encoder: Conv2d(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), groups=16)
    dilation = 2
    pad_size = (3 - 1) * dilation  # = 4

    conv = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(0, 1),
                     dilation=(dilation, 1), groups=16)

    # Input: (1, 16, 10, 33)
    x = torch.randn(1, 16, 10, 33)
    print(f"Input shape: {x.shape}")

    # Manual causal pad
    x_padded = F.pad(x, [0, 0, pad_size, 0])  # pad left=0, right=0, top=pad_size, bottom=0
    print(f"After F.pad: {x_padded.shape}")

    y = conv(x_padded)
    print(f"After Conv2d: {y.shape}")
    print()

def test_decoder_depth_conv():
    """Test ConvTranspose2d with padding parameter."""
    print("=== Decoder GTConvBlock Depth Conv ===")

    # Decoder: ConvTranspose2d(16, 16, (3,3), stride=(1,1), padding=(2*dilation,1), dilation=(dilation,1), groups=16)
    dilation = 2
    pad_size = (3 - 1) * dilation  # = 4

    deconv = nn.ConvTranspose2d(16, 16, (3, 3), stride=(1, 1),
                                 padding=(2*dilation, 1),  # (4, 1) for dilation=2
                                 dilation=(dilation, 1), groups=16)

    # Input: (1, 16, 10, 33)
    x = torch.randn(1, 16, 10, 33)
    print(f"Input shape: {x.shape}")

    # The code still does F.pad before deconv!
    x_padded = F.pad(x, [0, 0, pad_size, 0])
    print(f"After F.pad: {x_padded.shape}")

    y = deconv(x_padded)
    print(f"After ConvTranspose2d: {y.shape}")
    print()

    # What's the correct output size?
    # For ConvTranspose2d: output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    # H_out = (14 - 1) * 1 - 2 * 4 + 2 * (3 - 1) + 1 = 13 - 8 + 4 + 1 = 10
    # W_out = (33 - 1) * 1 - 2 * 1 + 1 * (3 - 1) + 1 = 32 - 2 + 2 + 1 = 33
    print(f"Expected output H = {x_padded.shape[2] - pad_size} = 10")

    # Actually let's check WITHOUT F.pad for decoder
    y_no_pad = deconv(x)
    print(f"Without F.pad, ConvTranspose2d output: {y_no_pad.shape}")

def test_real_gtcrn():
    """Test actual GTCRN GTConvBlock behavior."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from gtcrn import GTConvBlock

    print("=== Real GTConvBlock Tests ===")

    # Encoder GTConvBlock (dilation=2)
    en_block = GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False)
    x = torch.randn(1, 16, 10, 33)
    print(f"Encoder Input: {x.shape}")
    y_en = en_block(x)
    print(f"Encoder Output: {y_en.shape}")
    print()

    # Decoder GTConvBlock (dilation=2, padding=(4,1))
    de_block = GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True)
    print(f"Decoder Input: {x.shape}")
    y_de = de_block(x)
    print(f"Decoder Output: {y_de.shape}")
    print()

    print(f"pad_size for encoder/decoder: {en_block.pad_size}")

if __name__ == "__main__":
    test_encoder_depth_conv()
    test_decoder_depth_conv()
    test_real_gtcrn()
