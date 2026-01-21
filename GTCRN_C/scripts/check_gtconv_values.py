#!/usr/bin/env python3
"""
Debug C GTConvBlock by comparing each step.
"""

import os
import sys
import struct
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
test_dir = os.path.join(project_dir, "test_data")

def load_bin(path, size):
    with open(path, 'rb') as f:
        data = struct.unpack(f'{size}f', f.read(4 * size))
    return np.array(data, dtype=np.float32)

def compare(name, py_path, size, c_sum=None):
    try:
        py_data = load_bin(os.path.join(test_dir, py_path), size)
        py_sum = py_data.sum()
        print(f"{name}: Py sum={py_sum:.6f}", end="")
        if c_sum is not None:
            diff = abs(c_sum - py_sum)
            status = "PASS" if diff < 0.01 else "FAIL"
            print(f", C sum={c_sum:.6f}, diff={diff:.6f} [{status}]")
        else:
            print()
        return py_data
    except:
        print(f"{name}: File not found")
        return None

print("=== GTConvBlock 2 Step-by-Step Comparison ===\n")

# PyTorch reference values from debug_gtconv.py output
py_values = {
    "x1": 166.480637,
    "x2": 438.723145,
    "x1_sfe": 489.689758,
    "pc1": -769.807617,
    "bn1": 579.725769,
    "prelu1": 1695.994629,
    "padded": 1695.994751,
    "dc": 1298.125122,
    "bn2": 1193.047852,
    "prelu2": 2434.243164,
    "pc2": -560.052490,
    "bn3": 1993.616577,
    "tra": 815.181396,
    "output": 1253.904663
}

compare("x1", "py_gt2_x1.bin", 8 * 10 * 33)
compare("x2", "py_gt2_x2.bin", 8 * 10 * 33)
compare("x1_sfe", "py_gt2_x1_sfe.bin", 24 * 10 * 33)
compare("pc1", "py_gt2_pc1.bin", 16 * 10 * 33)
compare("bn1", "py_gt2_bn1.bin", 16 * 10 * 33)
compare("prelu1", "py_gt2_prelu1.bin", 16 * 10 * 33)
compare("padded", "py_gt2_padded.bin", 16 * 12 * 33)
compare("dc", "py_gt2_dc.bin", 16 * 10 * 33)
compare("bn2", "py_gt2_bn2.bin", 16 * 10 * 33)
compare("prelu2", "py_gt2_prelu2.bin", 16 * 10 * 33)
compare("pc2", "py_gt2_pc2.bin", 8 * 10 * 33)
compare("bn3", "py_gt2_bn3.bin", 8 * 10 * 33)
compare("tra", "py_gt2_tra.bin", 8 * 10 * 33)
compare("output", "py_gt2_output.bin", 16 * 10 * 33)
