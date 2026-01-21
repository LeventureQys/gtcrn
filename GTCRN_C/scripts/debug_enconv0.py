#!/usr/bin/env python3
"""
Debug EnConv0 specifically to understand the C vs PyTorch difference.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN

def save_bin(data, path):
    """Save tensor to binary file."""
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    data = data.flatten().astype(np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack(f'{len(data)}f', *data))

def load_bin(path, size):
    """Load binary float array."""
    with open(path, 'rb') as f:
        data = struct.unpack(f'{size}f', f.read(4 * size))
    return np.array(data, dtype=np.float32)

def main():
    print("=== Debug EnConv0 ===\n")

    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])

    test_dir = os.path.join(project_dir, "test_data")
    os.makedirs(test_dir, exist_ok=True)

    # Get EnConv0 layer
    en_conv0 = model.encoder.en_convs[0]
    print(f"EnConv0 structure:")
    print(f"  Conv: {en_conv0.conv}")
    print(f"  BN: {en_conv0.bn}")
    print(f"  Act: {en_conv0.act}")

    conv = en_conv0.conv
    print(f"\nConv details:")
    print(f"  in_channels: {conv.in_channels}")
    print(f"  out_channels: {conv.out_channels}")
    print(f"  kernel_size: {conv.kernel_size}")
    print(f"  stride: {conv.stride}")
    print(f"  padding: {conv.padding}")
    print(f"  groups: {conv.groups}")
    print(f"  weight shape: {conv.weight.shape}")
    print(f"  bias shape: {conv.bias.shape}")

    # Print weight sums for verification
    print(f"\n  Conv weight sum: {conv.weight.sum():.6f}")
    print(f"  Conv bias sum: {conv.bias.sum():.6f}")

    bn = en_conv0.bn
    print(f"\nBN details:")
    print(f"  weight sum (gamma): {bn.weight.sum():.6f}")
    print(f"  bias sum (beta): {bn.bias.sum():.6f}")
    print(f"  running_mean sum: {bn.running_mean.sum():.6f}")
    print(f"  running_var sum: {bn.running_var.sum():.6f}")

    prelu = en_conv0.act
    print(f"\nPReLU details:")
    print(f"  num_parameters: {prelu.num_parameters}")
    print(f"  weight: {prelu.weight.data}")

    # Load SFE output (input to EnConv0)
    py_feat_sfe = load_bin(os.path.join(test_dir, "py_feat_sfe.bin"), 9 * 10 * 129)
    py_feat_sfe = py_feat_sfe.reshape(1, 9, 10, 129)
    feat_sfe = torch.from_numpy(py_feat_sfe)

    print(f"\nInput (feat_sfe): shape={feat_sfe.shape}, sum={feat_sfe.sum():.6f}")

    with torch.no_grad():
        # Step-by-step through EnConv0
        # 1. Conv only
        conv_out = conv(feat_sfe)
        print(f"\n1. After Conv only: shape={conv_out.shape}, sum={conv_out.sum():.6f}")
        save_bin(conv_out.numpy(), os.path.join(test_dir, "py_enconv0_conv.bin"))

        # 2. BN only
        bn_out = bn(conv_out)
        print(f"2. After BN: shape={bn_out.shape}, sum={bn_out.sum():.6f}")
        save_bin(bn_out.numpy(), os.path.join(test_dir, "py_enconv0_bn.bin"))

        # 3. PReLU
        prelu_out = prelu(bn_out)
        print(f"3. After PReLU: shape={prelu_out.shape}, sum={prelu_out.sum():.6f}")
        save_bin(prelu_out.numpy(), os.path.join(test_dir, "py_enconv0_prelu.bin"))

        # Full forward
        full_out = en_conv0(feat_sfe)
        print(f"4. Full EnConv0: shape={full_out.shape}, sum={full_out.sum():.6f}")

    # Export Conv weights in C format
    # PyTorch Conv2d weight: (out_ch, in_ch, kH, kW) = (16, 9, 1, 5)
    print(f"\n=== Exporting Conv Weights ===")
    w = conv.weight.detach().numpy()  # (16, 9, 1, 5)
    print(f"Weight shape: {w.shape}")
    print(f"First 10 weights: {w.flatten()[:10]}")
    save_bin(w, os.path.join(test_dir, "py_enconv0_weight.bin"))
    save_bin(conv.bias.detach().numpy(), os.path.join(test_dir, "py_enconv0_bias.bin"))

    # Verify manual conv calculation
    print(f"\n=== Manual Conv2d Verification ===")
    x = feat_sfe.numpy()  # (1, 9, 10, 129)
    w = conv.weight.detach().numpy()  # (16, 9, 1, 5)
    b = conv.bias.detach().numpy()  # (16,)

    # Conv2d(9, 16, (1,5), stride=(1,2), padding=(0,2))
    # Output size: (B, 16, 10, 65) since (129 + 2*2 - 5) / 2 + 1 = 65

    out_h = 10
    out_w = 65
    out = np.zeros((1, 16, out_h, out_w), dtype=np.float32)

    for oc in range(16):
        for oh in range(out_h):
            for ow in range(out_w):
                val = b[oc]
                for ic in range(9):
                    for kh in range(1):
                        for kw in range(5):
                            ih = oh * 1 - 0 + kh * 1  # stride=1, padding=0, dilation=1
                            iw = ow * 2 - 2 + kw * 1  # stride=2, padding=2, dilation=1
                            if 0 <= ih < 10 and 0 <= iw < 129:
                                val += x[0, ic, ih, iw] * w[oc, ic, kh, kw]
                out[0, oc, oh, ow] = val

    print(f"Manual conv output sum: {out.sum():.6f}")
    print(f"PyTorch conv output sum: {conv_out.sum():.6f}")
    print(f"Difference: {abs(out.sum() - conv_out.sum().item()):.10f}")

    # Check specific values
    print(f"\nFirst 5 manual output: {out.flatten()[:5]}")
    print(f"First 5 PyTorch output: {conv_out.flatten()[:5].numpy()}")

if __name__ == "__main__":
    main()
