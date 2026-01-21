#!/usr/bin/env python3
"""
Debug GTConvBlock step by step.
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
    if hasattr(data, 'detach'):
        data = data.detach().cpu().numpy()
    data = data.flatten().astype(np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack(f'{len(data)}f', *data))

def load_bin(path, size):
    with open(path, 'rb') as f:
        data = struct.unpack(f'{size}f', f.read(4 * size))
    return np.array(data, dtype=np.float32)

def main():
    print("=== Debug GTConvBlock 2 (Encoder) ===\n")

    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])

    test_dir = os.path.join(project_dir, "test_data")

    # Get GTConvBlock 2 (first GTConvBlock)
    gt2 = model.encoder.en_convs[2]
    print(f"GTConvBlock 2:")
    print(f"  pad_size: {gt2.pad_size}")
    print(f"  use_deconv: {gt2.use_deconv}")
    print(f"  SFE: {gt2.sfe}")
    print(f"  point_conv1: {gt2.point_conv1}")
    print(f"  point_conv2: {gt2.point_conv2}")
    print(f"  depth_conv: {gt2.depth_conv}")

    # Load EnConv1 output as input to GTConvBlock 2
    en1_out = load_bin(os.path.join(test_dir, "py_en_conv1.bin"), 16 * 10 * 33)
    en1_out = torch.from_numpy(en1_out.reshape(1, 16, 10, 33))

    print(f"\nInput (en_conv1 output): shape={en1_out.shape}, sum={en1_out.sum():.6f}")

    with torch.no_grad():
        # Step by step through GTConvBlock
        x = en1_out

        # 1. Split channels
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        print(f"\n1. Split channels:")
        print(f"   x1: shape={x1.shape}, sum={x1.sum():.6f}")
        print(f"   x2: shape={x2.shape}, sum={x2.sum():.6f}")
        save_bin(x1, os.path.join(test_dir, "py_gt2_x1.bin"))
        save_bin(x2, os.path.join(test_dir, "py_gt2_x2.bin"))

        # 2. SFE on x1
        x1_sfe = gt2.sfe(x1)
        print(f"\n2. SFE on x1:")
        print(f"   x1_sfe: shape={x1_sfe.shape}, sum={x1_sfe.sum():.6f}")
        save_bin(x1_sfe, os.path.join(test_dir, "py_gt2_x1_sfe.bin"))

        # 3. Point conv 1
        h1 = gt2.point_conv1(x1_sfe)
        print(f"\n3. Point conv 1:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_pc1.bin"))

        # 4. BN1
        h1 = gt2.point_bn1(h1)
        print(f"\n4. Point BN1:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_bn1.bin"))

        # 5. PReLU 1
        h1 = gt2.point_act(h1)
        print(f"\n5. Point PReLU 1:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        print(f"   PReLU weight: {gt2.point_act.weight.data}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_prelu1.bin"))

        # 6. Pad
        h1_padded = F.pad(h1, [0, 0, gt2.pad_size, 0])
        print(f"\n6. Pad (pad_size={gt2.pad_size}):")
        print(f"   h1_padded: shape={h1_padded.shape}, sum={h1_padded.sum():.6f}")
        save_bin(h1_padded, os.path.join(test_dir, "py_gt2_padded.bin"))

        # 7. Depth conv
        h1 = gt2.depth_conv(h1_padded)
        print(f"\n7. Depth conv:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        print(f"   depth_conv: {gt2.depth_conv}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_dc.bin"))

        # 8. Depth BN
        h1 = gt2.depth_bn(h1)
        print(f"\n8. Depth BN:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_bn2.bin"))

        # 9. Depth PReLU
        h1 = gt2.depth_act(h1)
        print(f"\n9. Depth PReLU:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        print(f"   PReLU weight: {gt2.depth_act.weight.data}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_prelu2.bin"))

        # 10. Point conv 2
        h1 = gt2.point_conv2(h1)
        print(f"\n10. Point conv 2:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_pc2.bin"))

        # 11. Point BN2
        h1 = gt2.point_bn2(h1)
        print(f"\n11. Point BN2:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_bn3.bin"))

        # 12. TRA
        h1 = gt2.tra(h1)
        print(f"\n12. TRA:")
        print(f"   h1: shape={h1.shape}, sum={h1.sum():.6f}")
        save_bin(h1, os.path.join(test_dir, "py_gt2_tra.bin"))

        # 13. Channel shuffle
        output = gt2.shuffle(h1, x2)
        print(f"\n13. Channel shuffle:")
        print(f"   output: shape={output.shape}, sum={output.sum():.6f}")
        save_bin(output, os.path.join(test_dir, "py_gt2_output.bin"))

        # Full block output
        full_out = gt2(en1_out)
        print(f"\n14. Full GTConvBlock output: shape={full_out.shape}, sum={full_out.sum():.6f}")

if __name__ == "__main__":
    main()
