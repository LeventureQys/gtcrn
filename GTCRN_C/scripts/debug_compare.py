#!/usr/bin/env python3
"""
Compare C and PyTorch outputs layer by layer.
Requires debug tensors from both implementations.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn.functional as F

# Add gtcrn directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN

def read_debug_tensor(filepath):
    """Read tensor from debug binary file."""
    with open(filepath, 'rb') as f:
        ndim = struct.unpack('I', f.read(4))[0]
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('I', f.read(4))[0])
        size = 1
        for s in shape:
            size *= s
        data = struct.unpack(f'{size}f', f.read(4 * size))
        return np.array(data, dtype=np.float32).reshape(shape)

def compare_tensors(name, a, b, rtol=1e-4, atol=1e-5):
    """Compare two tensors and print statistics."""
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        return False

    diff = np.abs(a - b)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Relative error
    rel_diff = diff / (np.abs(b) + 1e-8)
    max_rel = np.max(rel_diff)

    passed = max_diff < atol or max_rel < rtol
    status = "PASS" if passed else "FAIL"

    print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, max_rel={max_rel:.4f} [{status}]")

    if not passed:
        # Find where the difference is largest
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    Largest diff at {idx}: PyTorch={b[idx]:.6f}, C={a[idx]:.6f}")

    return passed

def test_sfe():
    """Test SFE layer."""
    print("\n=== Testing SFE ===")

    from gtcrn import SFE
    sfe = SFE(kernel_size=3, stride=1)

    # Simple test
    x = torch.arange(60, dtype=torch.float32).reshape(1, 3, 4, 5)
    y = sfe(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    # Verify first channel group
    # For channel 0: left neighbor, center, right neighbor
    expected = torch.zeros(1, 9, 4, 5)
    for c in range(3):
        for t in range(4):
            for f in range(5):
                f_left = f - 1 if f > 0 else -1
                f_right = f + 1 if f < 4 else -1

                v_left = x[0, c, t, f_left].item() if f_left >= 0 else 0
                v_center = x[0, c, t, f].item()
                v_right = x[0, c, t, f_right].item() if f_right >= 0 else 0

                expected[0, c*3+0, t, f] = v_left
                expected[0, c*3+1, t, f] = v_center
                expected[0, c*3+2, t, f] = v_right

    diff = (y - expected).abs().max()
    print(f"SFE verification diff: {diff:.8f}")
    if diff > 1e-6:
        print("SFE implementation may differ!")

def test_conv2d_layer():
    """Test Conv2d implementation."""
    print("\n=== Testing Conv2d Layer ===")

    import torch.nn as nn

    # EnConv0: Conv2d(9, 16, (1,5), stride=(1,2), padding=(0,2))
    conv = nn.Conv2d(9, 16, (1, 5), stride=(1, 2), padding=(0, 2))

    np.random.seed(42)
    x = torch.from_numpy(np.random.randn(1, 9, 63, 129).astype(np.float32))

    with torch.no_grad():
        y = conv(x)

    print(f"Input: {x.shape} -> Output: {y.shape}")
    print(f"Output sample: {y.flatten()[:5].tolist()}")

def test_erb():
    """Test ERB layer."""
    print("\n=== Testing ERB ===")

    from gtcrn import ERB
    erb = ERB(65, 64)

    np.random.seed(42)
    x = torch.from_numpy(np.random.randn(1, 3, 10, 257).astype(np.float32))

    with torch.no_grad():
        y_bm = erb.bm(x)
        y_bs = erb.bs(y_bm)

    print(f"Input: {x.shape}")
    print(f"After bm: {y_bm.shape}")
    print(f"After bs: {y_bs.shape}")

    # Check low frequency bins are preserved
    low_diff = (x[..., :65] - y_bm[..., :65]).abs().max()
    print(f"Low freq diff: {low_diff:.8f}")

def test_gtconvblock():
    """Test GTConvBlock with explicit weights."""
    print("\n=== Testing GTConvBlock ===")

    from gtcrn import GTConvBlock

    # Encoder GTConvBlock (dilation=1)
    en_block = GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False)

    np.random.seed(42)
    x = torch.from_numpy(np.random.randn(1, 16, 10, 33).astype(np.float32))

    with torch.no_grad():
        y = en_block(x)

    print(f"Encoder GTConvBlock: {x.shape} -> {y.shape}")
    print(f"Output sample: {y.flatten()[:5].tolist()}")

    # Decoder GTConvBlock (dilation=1)
    de_block = GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2,1), dilation=(1,1), use_deconv=True)

    with torch.no_grad():
        y_de = de_block(x)

    print(f"Decoder GTConvBlock: {x.shape} -> {y_de.shape}")
    print(f"Output sample: {y_de.flatten()[:5].tolist()}")

def test_dpgrnn():
    """Test DPGRNN implementation."""
    print("\n=== Testing DPGRNN ===")

    from gtcrn import DPGRNN

    dpgrnn = DPGRNN(16, 33, 16).eval()

    np.random.seed(42)
    x = torch.from_numpy(np.random.randn(1, 16, 10, 33).astype(np.float32))

    with torch.no_grad():
        y = dpgrnn(x)

    print(f"DPGRNN: {x.shape} -> {y.shape}")
    print(f"Output sample: {y.flatten()[:5].tolist()}")

def debug_full_model():
    """Debug full model layer by layer."""
    print("\n=== Full Model Debug ===")

    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")

    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # Small test input
    np.random.seed(42)
    spec = torch.from_numpy(np.random.randn(1, 257, 10, 2).astype(np.float32)) * 0.1

    with torch.no_grad():
        # Step by step
        spec_ref = spec
        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)

        print(f"1. Feature: {feat.shape}, range=[{feat.min():.4f}, {feat.max():.4f}]")

        feat_erb = model.erb.bm(feat)
        print(f"2. After ERB bm: {feat_erb.shape}")

        feat_sfe = model.sfe(feat_erb)
        print(f"3. After SFE: {feat_sfe.shape}")

        # Encoder
        en_out0 = model.encoder.en_convs[0](feat_sfe)
        print(f"4. After EnConv0: {en_out0.shape}")

        en_out1 = model.encoder.en_convs[1](en_out0)
        print(f"5. After EnConv1: {en_out1.shape}")

        en_out2 = model.encoder.en_convs[2](en_out1)
        print(f"6. After EnGT2: {en_out2.shape}")

        en_out3 = model.encoder.en_convs[3](en_out2)
        print(f"7. After EnGT3: {en_out3.shape}")

        en_out4 = model.encoder.en_convs[4](en_out3)
        print(f"8. After EnGT4: {en_out4.shape}")

        # DPGRNN
        dp1_out = model.dpgrnn1(en_out4)
        print(f"9. After DPGRNN1: {dp1_out.shape}")

        dp2_out = model.dpgrnn2(dp1_out)
        print(f"10. After DPGRNN2: {dp2_out.shape}")

        # Decoder with skip connections
        en_outs = [en_out0, en_out1, en_out2, en_out3, en_out4]

        de_in = dp2_out + en_out4
        de_out0 = model.decoder.de_convs[0](de_in)
        print(f"11. After DeGT0: {de_out0.shape}")

        de_in1 = de_out0 + en_out3
        de_out1 = model.decoder.de_convs[1](de_in1)
        print(f"12. After DeGT1: {de_out1.shape}")

        de_in2 = de_out1 + en_out2
        de_out2 = model.decoder.de_convs[2](de_in2)
        print(f"13. After DeGT2: {de_out2.shape}")

        de_in3 = de_out2 + en_out1
        de_out3 = model.decoder.de_convs[3](de_in3)
        print(f"14. After DeConv3: {de_out3.shape}")

        de_in4 = de_out3 + en_out0
        de_out4 = model.decoder.de_convs[4](de_in4)
        print(f"15. After DeConv4: {de_out4.shape}")

        # ERB expansion
        mask = model.erb.bs(de_out4)
        print(f"16. Mask: {mask.shape}, range=[{mask.min():.4f}, {mask.max():.4f}]")

        # Apply mask
        spec_enh = model.mask(mask, spec_ref.permute(0,3,2,1))
        spec_enh = spec_enh.permute(0,3,2,1)
        print(f"17. Enhanced: {spec_enh.shape}")

if __name__ == "__main__":
    test_sfe()
    test_erb()
    test_conv2d_layer()
    test_gtconvblock()
    test_dpgrnn()
    debug_full_model()
