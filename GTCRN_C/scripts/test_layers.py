#!/usr/bin/env python3
"""
Test individual layer implementations in C against PyTorch.
Generates simple test cases with known inputs.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn as nn

# Add gtcrn directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN

def test_conv2d():
    """Test Conv2d layer."""
    print("=== Testing Conv2d ===")

    # EnConv0: Conv2d(9, 16, (1,5), stride=(1,2), padding=(0,2))
    # Input: (1, 9, T, 129), Output: (1, 16, T, 65)
    T = 4
    conv = nn.Conv2d(9, 16, (1, 5), stride=(1, 2), padding=(0, 2), bias=True)

    # Use small random input
    np.random.seed(42)
    x = torch.from_numpy(np.random.randn(1, 9, T, 129).astype(np.float32))

    with torch.no_grad():
        y = conv(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output first 5 values: {y.flatten()[:5].tolist()}")

    # Export weights and input for C testing
    output_dir = os.path.join(project_dir, "test_data")
    os.makedirs(output_dir, exist_ok=True)

    # Save input
    with open(os.path.join(output_dir, "conv2d_input.bin"), 'wb') as f:
        data = x.numpy().astype(np.float32).flatten()
        f.write(struct.pack(f'{len(data)}f', *data))

    # Save weights
    with open(os.path.join(output_dir, "conv2d_weight.bin"), 'wb') as f:
        data = conv.weight.numpy().astype(np.float32).flatten()
        f.write(struct.pack(f'{len(data)}f', *data))

    with open(os.path.join(output_dir, "conv2d_bias.bin"), 'wb') as f:
        data = conv.bias.numpy().astype(np.float32).flatten()
        f.write(struct.pack(f'{len(data)}f', *data))

    # Save expected output
    with open(os.path.join(output_dir, "conv2d_output.bin"), 'wb') as f:
        data = y.numpy().astype(np.float32).flatten()
        f.write(struct.pack(f'{len(data)}f', *data))

    print(f"Test data saved to: {output_dir}")

def test_gtcrn_simple():
    """Test GTCRN with simple input."""
    print("\n=== Testing GTCRN Simple ===")

    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")

    # Load model
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # Simple test input: (1, 257, 4, 2) - 4 frames
    np.random.seed(42)
    spec = torch.from_numpy(np.random.randn(1, 257, 4, 2).astype(np.float32)) * 0.1

    # Step-by-step forward
    with torch.no_grad():
        spec_ref = spec  # (B,F,T,2)

        spec_real = spec[..., 0].permute(0,2,1)  # (B, T, F)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        print(f"feat shape: {feat.shape}")
        print(f"feat first 10 values: {feat.flatten()[:10].tolist()}")

        # ERB compression
        feat_erb = model.erb.bm(feat)  # (B,3,T,129)
        print(f"feat_erb shape: {feat_erb.shape}")
        print(f"feat_erb first 10 values: {feat_erb.flatten()[:10].tolist()}")

        # SFE
        feat_sfe = model.sfe(feat_erb)  # (B,9,T,129)
        print(f"feat_sfe shape: {feat_sfe.shape}")
        print(f"feat_sfe first 10 values: {feat_sfe.flatten()[:10].tolist()}")

        # EnConv0
        en_conv0_out = model.encoder.en_convs[0](feat_sfe)
        print(f"en_conv0_out shape: {en_conv0_out.shape}")
        print(f"en_conv0_out first 10 values: {en_conv0_out.flatten()[:10].tolist()}")

def test_sfe():
    """Test SFE (unfold) operation."""
    print("\n=== Testing SFE ===")

    from gtcrn import SFE

    sfe = SFE(kernel_size=3, stride=1)

    # Test input: (1, 3, 4, 5)
    x = torch.arange(60).float().reshape(1, 3, 4, 5)
    print(f"Input:\n{x}")

    y = sfe(x)
    print(f"Output shape: {y.shape}")
    print(f"Output:\n{y}")

    # Export for C testing
    output_dir = os.path.join(project_dir, "test_data")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "sfe_input.bin"), 'wb') as f:
        data = x.numpy().astype(np.float32).flatten()
        f.write(struct.pack(f'{len(data)}f', *data))

    with open(os.path.join(output_dir, "sfe_output.bin"), 'wb') as f:
        data = y.numpy().astype(np.float32).flatten()
        f.write(struct.pack(f'{len(data)}f', *data))

def main():
    test_sfe()
    test_conv2d()
    test_gtcrn_simple()

if __name__ == "__main__":
    main()
