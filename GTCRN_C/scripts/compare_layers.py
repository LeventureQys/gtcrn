#!/usr/bin/env python3
"""
Compare C and PyTorch layer by layer using exported debug tensors.
Generates a detailed diff report.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN

def load_bin(path, size):
    """Load binary float array."""
    with open(path, 'rb') as f:
        data = struct.unpack(f'{size}f', f.read(4 * size))
    return np.array(data, dtype=np.float32)

def save_bin(data, path):
    """Save binary float array."""
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    data = data.flatten().astype(np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack(f'{len(data)}f', *data))

def test_basic_layers():
    """Test basic layer implementations."""
    print("=== Testing Basic Layers ===\n")

    test_dir = os.path.join(project_dir, "test_data")
    os.makedirs(test_dir, exist_ok=True)

    # Test 1: Conv2d with groups
    print("1. Conv2d (groups=2):")
    conv = nn.Conv2d(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, bias=True)

    np.random.seed(42)
    torch.manual_seed(42)
    x = torch.randn(1, 16, 10, 65)

    with torch.no_grad():
        y = conv(x)

    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Weight: {conv.weight.shape}")
    print(f"   Output sum: {y.sum():.6f}")

    save_bin(x, os.path.join(test_dir, "conv_group_input.bin"))
    save_bin(conv.weight, os.path.join(test_dir, "conv_group_weight.bin"))
    save_bin(conv.bias, os.path.join(test_dir, "conv_group_bias.bin"))
    save_bin(y, os.path.join(test_dir, "conv_group_output.bin"))
    print()

    # Test 2: ConvTranspose2d with groups
    print("2. ConvTranspose2d (groups=2):")
    deconv = nn.ConvTranspose2d(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, bias=True)

    with torch.no_grad():
        y_de = deconv(y)

    print(f"   Input: {y.shape} -> Output: {y_de.shape}")
    print(f"   Weight: {deconv.weight.shape}")
    print(f"   Output sum: {y_de.sum():.6f}")

    save_bin(y, os.path.join(test_dir, "deconv_group_input.bin"))
    save_bin(deconv.weight, os.path.join(test_dir, "deconv_group_weight.bin"))
    save_bin(deconv.bias, os.path.join(test_dir, "deconv_group_bias.bin"))
    save_bin(y_de, os.path.join(test_dir, "deconv_group_output.bin"))
    print()

    # Test 3: Depthwise Conv with dilation
    print("3. Depthwise Conv2d (dilation=2):")
    dwconv = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), groups=16)

    # Padded input (simulating F.pad)
    x_pad = F.pad(x[:, :, :, :33], [0, 0, 4, 0])  # pad_size = (3-1)*2 = 4
    print(f"   Input after pad: {x_pad.shape}")

    with torch.no_grad():
        y_dw = dwconv(x_pad)

    print(f"   Output: {y_dw.shape}")
    print(f"   Output sum: {y_dw.sum():.6f}")

    save_bin(x_pad, os.path.join(test_dir, "dwconv_input.bin"))
    save_bin(dwconv.weight, os.path.join(test_dir, "dwconv_weight.bin"))
    save_bin(dwconv.bias, os.path.join(test_dir, "dwconv_bias.bin"))
    save_bin(y_dw, os.path.join(test_dir, "dwconv_output.bin"))
    print()

    # Test 4: Depthwise ConvTranspose2d with dilation
    print("4. Depthwise ConvTranspose2d (dilation=2):")
    dwdeconv = nn.ConvTranspose2d(16, 16, (3, 3), stride=(1, 1), padding=(4, 1), dilation=(2, 1), groups=16)

    with torch.no_grad():
        y_dwde = dwdeconv(x_pad)

    print(f"   Input: {x_pad.shape} -> Output: {y_dwde.shape}")
    print(f"   Output sum: {y_dwde.sum():.6f}")

    save_bin(x_pad, os.path.join(test_dir, "dwdeconv_input.bin"))
    save_bin(dwdeconv.weight, os.path.join(test_dir, "dwdeconv_weight.bin"))
    save_bin(dwdeconv.bias, os.path.join(test_dir, "dwdeconv_bias.bin"))
    save_bin(y_dwde, os.path.join(test_dir, "dwdeconv_output.bin"))
    print()

def test_gtcrn_step_by_step():
    """Test GTCRN step by step."""
    print("=== Testing GTCRN Step by Step ===\n")

    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    test_dir = os.path.join(project_dir, "test_data")

    # Load same input as C code uses
    spec_real = load_bin(os.path.join(test_dir, "spec_real.bin"), 10 * 257)
    spec_imag = load_bin(os.path.join(test_dir, "spec_imag.bin"), 10 * 257)

    spec_real = spec_real.reshape(10, 257)
    spec_imag = spec_imag.reshape(10, 257)

    # Convert to PyTorch format: (B, F, T, 2)
    spec = torch.zeros(1, 257, 10, 2)
    for t in range(10):
        for f in range(257):
            spec[0, f, t, 0] = float(spec_real[t, f])
            spec[0, f, t, 1] = float(spec_imag[t, f])

    print(f"Input spec: {spec.shape}")
    print(f"  Real sum: {spec[..., 0].sum():.6f}")
    print(f"  Imag sum: {spec[..., 1].sum():.6f}")

    with torch.no_grad():
        # Step 1: Feature tensor
        spec_real_t = spec[..., 0].permute(0, 2, 1)  # (B, T, F)
        spec_imag_t = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real_t**2 + spec_imag_t**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real_t, spec_imag_t], dim=1)
        print(f"\n1. Feature tensor: {feat.shape}, sum={feat.sum():.6f}")

        # Step 2: ERB compression
        feat_erb = model.erb.bm(feat)
        print(f"2. After ERB bm: {feat_erb.shape}, sum={feat_erb.sum():.6f}")

        # Step 3: SFE
        feat_sfe = model.sfe(feat_erb)
        print(f"3. After SFE: {feat_sfe.shape}, sum={feat_sfe.sum():.6f}")

        # Save intermediate for C comparison
        save_bin(feat.numpy(), os.path.join(test_dir, "py_feat.bin"))
        save_bin(feat_erb.numpy(), os.path.join(test_dir, "py_feat_erb.bin"))
        save_bin(feat_sfe.numpy(), os.path.join(test_dir, "py_feat_sfe.bin"))

        # Step 4: EnConv0
        en0 = model.encoder.en_convs[0](feat_sfe)
        print(f"4. After EnConv0: {en0.shape}, sum={en0.sum():.6f}")
        save_bin(en0.numpy(), os.path.join(test_dir, "py_en_conv0.bin"))

        # Step 5: EnConv1
        en1 = model.encoder.en_convs[1](en0)
        print(f"5. After EnConv1: {en1.shape}, sum={en1.sum():.6f}")
        save_bin(en1.numpy(), os.path.join(test_dir, "py_en_conv1.bin"))

        # Step 6-8: GTConvBlocks
        en2 = model.encoder.en_convs[2](en1)
        print(f"6. After EnGT2: {en2.shape}, sum={en2.sum():.6f}")
        save_bin(en2.numpy(), os.path.join(test_dir, "py_en_gt2.bin"))

        en3 = model.encoder.en_convs[3](en2)
        print(f"7. After EnGT3: {en3.shape}, sum={en3.sum():.6f}")
        save_bin(en3.numpy(), os.path.join(test_dir, "py_en_gt3.bin"))

        en4 = model.encoder.en_convs[4](en3)
        print(f"8. After EnGT4: {en4.shape}, sum={en4.sum():.6f}")
        save_bin(en4.numpy(), os.path.join(test_dir, "py_en_gt4.bin"))

        # Step 9-10: DPGRNN
        dp1 = model.dpgrnn1(en4)
        print(f"9. After DPGRNN1: {dp1.shape}, sum={dp1.sum():.6f}")
        save_bin(dp1.numpy(), os.path.join(test_dir, "py_dpgrnn1.bin"))

        dp2 = model.dpgrnn2(dp1)
        print(f"10. After DPGRNN2: {dp2.shape}, sum={dp2.sum():.6f}")
        save_bin(dp2.numpy(), os.path.join(test_dir, "py_dpgrnn2.bin"))

        # Full model
        en_outs = [en0, en1, en2, en3, en4]
        de_out = model.decoder(dp2, en_outs)
        print(f"11. After Decoder: {de_out.shape}, sum={de_out.sum():.6f}")

        mask = model.erb.bs(de_out)
        print(f"12. Mask: {mask.shape}, sum={mask.sum():.6f}")

        output = model(spec)
        print(f"\n13. Full output: {output.shape}, sum={output.sum():.6f}")
        save_bin(output.numpy(), os.path.join(test_dir, "py_full_output.bin"))

if __name__ == "__main__":
    test_basic_layers()
    test_gtcrn_step_by_step()
