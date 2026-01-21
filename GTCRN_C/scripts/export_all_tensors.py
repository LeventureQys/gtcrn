#!/usr/bin/env python3
"""
Export all intermediate tensors from GTCRN for C comparison.
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
    print("=== Exporting GTCRN Intermediate Tensors ===\n")

    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    test_dir = os.path.join(project_dir, "test_data")
    os.makedirs(test_dir, exist_ok=True)

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

    with torch.no_grad():
        # Step 1: Feature tensor
        spec_real_t = spec[..., 0].permute(0, 2, 1)  # (B, T, F)
        spec_imag_t = spec[..., 1].permute(0, 2, 1)
        spec_mag = torch.sqrt(spec_real_t**2 + spec_imag_t**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real_t, spec_imag_t], dim=1)
        print(f"1. Feature: {feat.shape}, sum={feat.sum():.6f}")
        save_bin(feat, os.path.join(test_dir, "py_feat.bin"))

        # Step 2: ERB
        feat_erb = model.erb.bm(feat)
        print(f"2. ERB: {feat_erb.shape}, sum={feat_erb.sum():.6f}")
        save_bin(feat_erb, os.path.join(test_dir, "py_feat_erb.bin"))

        # Step 3: SFE
        feat_sfe = model.sfe(feat_erb)
        print(f"3. SFE: {feat_sfe.shape}, sum={feat_sfe.sum():.6f}")
        save_bin(feat_sfe, os.path.join(test_dir, "py_feat_sfe.bin"))

        # Step 4: EnConv0
        en0 = model.encoder.en_convs[0](feat_sfe)
        print(f"4. EnConv0: {en0.shape}, sum={en0.sum():.6f}")
        save_bin(en0, os.path.join(test_dir, "py_en_conv0.bin"))

        # Step 5: EnConv1
        en1 = model.encoder.en_convs[1](en0)
        print(f"5. EnConv1: {en1.shape}, sum={en1.sum():.6f}")
        save_bin(en1, os.path.join(test_dir, "py_en_conv1.bin"))

        # Step 6-8: GTConvBlocks
        en2 = model.encoder.en_convs[2](en1)
        print(f"6. EnGT2: {en2.shape}, sum={en2.sum():.6f}")
        save_bin(en2, os.path.join(test_dir, "py_en_gt2.bin"))

        en3 = model.encoder.en_convs[3](en2)
        print(f"7. EnGT3: {en3.shape}, sum={en3.sum():.6f}")
        save_bin(en3, os.path.join(test_dir, "py_en_gt3.bin"))

        en4 = model.encoder.en_convs[4](en3)
        print(f"8. EnGT4: {en4.shape}, sum={en4.sum():.6f}")
        save_bin(en4, os.path.join(test_dir, "py_en_gt4.bin"))

        # Store encoder outputs
        en_outs = [en0, en1, en2, en3, en4]

        # Step 9-10: DPGRNN
        dp1 = model.dpgrnn1(en4)
        print(f"9. DPGRNN1: {dp1.shape}, sum={dp1.sum():.6f}")
        save_bin(dp1, os.path.join(test_dir, "py_dpgrnn1.bin"))

        dp2 = model.dpgrnn2(dp1)
        print(f"10. DPGRNN2: {dp2.shape}, sum={dp2.sum():.6f}")
        save_bin(dp2, os.path.join(test_dir, "py_dpgrnn2.bin"))

        # Step 11-15: Decoder with skip connections
        de_in = dp2 + en4
        de0 = model.decoder.de_convs[0](de_in)
        print(f"11. DeGT0: {de0.shape}, sum={de0.sum():.6f}")
        save_bin(de0, os.path.join(test_dir, "py_de_gt0.bin"))

        de_in1 = de0 + en3
        de1 = model.decoder.de_convs[1](de_in1)
        print(f"12. DeGT1: {de1.shape}, sum={de1.sum():.6f}")
        save_bin(de1, os.path.join(test_dir, "py_de_gt1.bin"))

        de_in2 = de1 + en2
        de2 = model.decoder.de_convs[2](de_in2)
        print(f"13. DeGT2: {de2.shape}, sum={de2.sum():.6f}")
        save_bin(de2, os.path.join(test_dir, "py_de_gt2.bin"))

        de_in3 = de2 + en1
        de3 = model.decoder.de_convs[3](de_in3)
        print(f"14. DeConv3: {de3.shape}, sum={de3.sum():.6f}")
        save_bin(de3, os.path.join(test_dir, "py_de_conv3.bin"))

        de_in4 = de3 + en0
        de4 = model.decoder.de_convs[4](de_in4)
        print(f"15. DeConv4: {de4.shape}, sum={de4.sum():.6f}")
        save_bin(de4, os.path.join(test_dir, "py_de_conv4.bin"))

        # Step 16: ERB expansion
        mask = model.erb.bs(de4)
        print(f"16. Mask: {mask.shape}, sum={mask.sum():.6f}")
        save_bin(mask, os.path.join(test_dir, "py_mask.bin"))

        # Full output
        output = model(spec)
        print(f"\n17. Full output: {output.shape}, sum={output.sum():.6f}")
        save_bin(output, os.path.join(test_dir, "py_full_output.bin"))

    print(f"\nAll tensors saved to: {test_dir}")

if __name__ == "__main__":
    main()
