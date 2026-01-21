#!/usr/bin/env python3
"""
Compare outputs with 1-frame shift.
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

    print("=== Testing if C is shifted by 1 frame ===\n")

    # Compare C frame N with Python frame N-1
    print("C frame N vs Python frame N (no shift):")
    for frame in range(5):
        start = frame * hop
        end = start + hop
        c_seg = c_stream[start:end]
        py_seg = py_stream[start:end]
        if len(c_seg) > 0 and len(py_seg) > 0:
            corr = np.corrcoef(c_seg, py_seg)[0, 1]
            print(f"  Frame {frame}: corr={corr:.4f}")

    print("\nC frame N vs Python frame N-1 (shift by 1):")
    for c_frame in range(1, 6):
        py_frame = c_frame - 1
        c_start = c_frame * hop
        c_end = c_start + hop
        py_start = py_frame * hop
        py_end = py_start + hop
        c_seg = c_stream[c_start:c_end]
        py_seg = py_stream[py_start:py_end]
        if len(c_seg) > 0 and len(py_seg) > 0:
            corr = np.corrcoef(c_seg, py_seg)[0, 1]
            print(f"  C frame {c_frame} vs Py frame {py_frame}: corr={corr:.4f}")

    # Also try correlation of entire signals with offset
    print("\n=== Correlation with different offsets ===")
    for offset in range(0, 10):
        offset_samples = offset * hop
        min_len = min(len(c_stream) - offset_samples, len(py_stream))
        if min_len > 0:
            corr = np.corrcoef(c_stream[offset_samples:offset_samples+min_len], py_stream[:min_len])[0, 1]
            print(f"  Offset {offset} frames ({offset_samples} samples): corr={corr:.4f}")


if __name__ == "__main__":
    main()
