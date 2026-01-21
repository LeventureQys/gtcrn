#!/usr/bin/env python3
"""
Debug C streaming ISTFT overlap-add.
"""

import os
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

def main():
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    py_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_stream_nocenter.wav")

    c_stream, _ = sf.read(c_stream_path)
    py_stream, _ = sf.read(py_stream_path)

    hop = 256

    # Look at frames after the warmup period
    print("=== Samples around frame boundaries (after warmup) ===\n")

    # Frame 100 (after warmup)
    for frame in [100, 101, 102]:
        start = frame * hop
        end = start + hop

        print(f"Frame {frame} (samples {start}-{end}):")
        # Show first 10 and last 10 samples
        c_seg = c_stream[start:end]
        py_seg_shifted = py_stream[start-hop:end-hop] if start >= hop else py_stream[:end-hop]

        print(f"  C first 5:  {c_seg[:5]}")
        print(f"  Py first 5: {py_seg_shifted[:5] if len(py_seg_shifted) >= 5 else 'N/A'}")
        print(f"  C last 5:   {c_seg[-5:]}")
        print(f"  Py last 5:  {py_seg_shifted[-5:] if len(py_seg_shifted) >= 5 else 'N/A'}")

        if len(py_seg_shifted) == len(c_seg):
            ratio = np.mean(np.abs(c_seg)) / np.mean(np.abs(py_seg_shifted) + 1e-10)
            print(f"  C/Py amplitude ratio: {ratio:.4f}")
        print()


if __name__ == "__main__":
    main()
