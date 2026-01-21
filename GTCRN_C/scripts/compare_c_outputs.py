#!/usr/bin/env python3
"""
Compare C complete vs C streaming outputs.
"""

import os
import sys
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

def main():
    print("=== C Complete vs C Streaming Comparison ===\n")

    # Load outputs
    c_complete_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_complete.wav")
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")

    c_complete, sr_c = sf.read(c_complete_path)
    c_stream, sr_s = sf.read(c_stream_path)

    print(f"C complete: {len(c_complete)} samples, sr={sr_c}")
    print(f"C stream: {len(c_stream)} samples, sr={sr_s}")

    min_len = min(len(c_complete), len(c_stream))

    # Overall statistics
    c_complete_rms = np.sqrt(np.mean(c_complete**2))
    c_stream_rms = np.sqrt(np.mean(c_stream**2))
    print(f"\nC complete RMS: {c_complete_rms:.6f}")
    print(f"C stream RMS: {c_stream_rms:.6f}")
    print(f"Energy ratio (stream/complete): {c_stream_rms/c_complete_rms:.4f}")

    correlation = np.corrcoef(c_complete[:min_len], c_stream[:min_len])[0, 1]
    print(f"\nOverall correlation: {correlation:.6f}")

    # Sample-wise difference
    diff = np.abs(c_complete[:min_len] - c_stream[:min_len])
    max_diff_idx = np.argmax(diff)
    print(f"\nMax difference at sample {max_diff_idx}: {diff[max_diff_idx]:.6f}")

    # First 20 samples
    print("\n=== First 20 samples ===")
    print("Sample | Complete    | Stream      | Diff")
    print("-" * 55)
    for i in range(min(20, min_len)):
        print(f"{i:6d} | {c_complete[i]:11.6f} | {c_stream[i]:11.6f} | {c_complete[i]-c_stream[i]:11.6f}")

    # Also load and compare Python streaming output
    py_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_debug/enhanced_py_stream.wav")
    if os.path.exists(py_stream_path):
        py_stream, sr_py = sf.read(py_stream_path)
        print(f"\n\n=== Python Streaming ===")
        print(f"Python stream: {len(py_stream)} samples, sr={sr_py}")
        print(f"Python stream RMS: {np.sqrt(np.mean(py_stream**2)):.6f}")

        min_len2 = min(len(c_complete), len(py_stream))
        corr_py_complete = np.corrcoef(c_complete[:min_len2], py_stream[:min_len2])[0, 1]
        print(f"Correlation (Python stream vs C complete): {corr_py_complete:.6f}")

        min_len3 = min(len(c_stream), len(py_stream))
        corr_c_py = np.corrcoef(c_stream[:min_len3], py_stream[:min_len3])[0, 1]
        print(f"Correlation (C stream vs Python stream): {corr_c_py:.6f}")


if __name__ == "__main__":
    main()
