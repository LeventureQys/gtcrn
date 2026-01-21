#!/usr/bin/env python3
"""
Compare C and Python streaming outputs in detail.
"""

import os
import sys
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

def main():
    print("=== Streaming Output Comparison ===\n")

    # Load outputs
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")
    py_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_stream.wav")

    c_output, sr_c = sf.read(c_stream_path)
    py_output, sr_py = sf.read(py_stream_path)

    print(f"C output: {len(c_output)} samples, sr={sr_c}")
    print(f"Python output: {len(py_output)} samples, sr={sr_py}")

    min_len = min(len(c_output), len(py_output))

    # Overall statistics
    c_rms = np.sqrt(np.mean(c_output**2))
    py_rms = np.sqrt(np.mean(py_output**2))
    print(f"\nC RMS: {c_rms:.6f}")
    print(f"Python RMS: {py_rms:.6f}")
    print(f"Energy ratio (C/Py): {c_rms/py_rms:.4f}")

    correlation = np.corrcoef(c_output[:min_len], py_output[:min_len])[0, 1]
    print(f"\nOverall correlation: {correlation:.6f}")

    # Segment analysis
    print("\n=== Segment Analysis ===")
    hop = 256  # samples per frame
    for frame_start in [0, 10, 50, 100, 200]:
        start = frame_start * hop
        end = start + 10 * hop  # 10 frames
        if end > min_len:
            continue

        c_seg = c_output[start:end]
        py_seg = py_output[start:end]

        corr = np.corrcoef(c_seg, py_seg)[0, 1]
        c_rms_seg = np.sqrt(np.mean(c_seg**2))
        py_rms_seg = np.sqrt(np.mean(py_seg**2))

        print(f"  Frames {frame_start}-{frame_start+10}: corr={corr:.4f}, "
              f"C_RMS={c_rms_seg:.6f}, Py_RMS={py_rms_seg:.6f}, ratio={c_rms_seg/py_rms_seg:.4f}")

    # Find where they diverge most
    print("\n=== Sample-wise Difference ===")
    diff = np.abs(c_output[:min_len] - py_output[:min_len])
    max_diff_idx = np.argmax(diff)
    print(f"Max difference at sample {max_diff_idx}: {diff[max_diff_idx]:.6f}")
    print(f"  C value: {c_output[max_diff_idx]:.6f}")
    print(f"  Py value: {py_output[max_diff_idx]:.6f}")

    # Print first 20 samples
    print("\n=== First 20 samples ===")
    print("Sample | C output    | Py output   | Diff")
    print("-" * 50)
    for i in range(min(20, min_len)):
        print(f"{i:6d} | {c_output[i]:11.6f} | {py_output[i]:11.6f} | {c_output[i]-py_output[i]:11.6f}")

    # Samples around frame boundaries (256 samples each)
    print("\n=== First 5 frame outputs (sample 0 of each frame) ===")
    print("Frame | C output    | Py output   | Diff")
    print("-" * 50)
    for frame in range(5):
        idx = frame * hop
        if idx < min_len:
            print(f"{frame:5d} | {c_output[idx]:11.6f} | {py_output[idx]:11.6f} | {c_output[idx]-py_output[idx]:11.6f}")


if __name__ == "__main__":
    main()
