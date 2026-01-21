#!/usr/bin/env python3
"""
Verify C streaming with 1-frame offset produces correct output.
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

    c_stream, sr = sf.read(c_stream_path)
    py_stream, _ = sf.read(py_stream_path)

    hop = 256

    # Align by removing first frame from C (shift C left by 256 samples)
    c_aligned = c_stream[hop:]  # Remove first 256 samples
    py_aligned = py_stream[:-hop] if len(py_stream) > hop else py_stream  # Remove last 256 samples

    min_len = min(len(c_aligned), len(py_aligned))
    c_aligned = c_aligned[:min_len]
    py_aligned = py_aligned[:min_len]

    # Overall correlation
    correlation = np.corrcoef(c_aligned, py_aligned)[0, 1]
    print(f"Overall correlation (after alignment): {correlation:.6f}")

    # RMS
    c_rms = np.sqrt(np.mean(c_aligned**2))
    py_rms = np.sqrt(np.mean(py_aligned**2))
    print(f"C RMS: {c_rms:.6f}, Python RMS: {py_rms:.6f}")
    print(f"Energy ratio: {c_rms/py_rms:.4f}")

    # Segment correlations
    print("\n=== Segment Analysis (after alignment) ===")
    for start_frame in [0, 10, 50, 100, 500, 1000]:
        start = start_frame * hop
        end = start + 10 * hop
        if end <= min_len:
            c_seg = c_aligned[start:end]
            py_seg = py_aligned[start:end]
            corr = np.corrcoef(c_seg, py_seg)[0, 1]
            print(f"  Frames {start_frame}-{start_frame+10}: corr={corr:.4f}")

    # Save aligned C output for comparison
    output_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream_aligned.wav")
    sf.write(output_path, c_aligned, sr)
    print(f"\nSaved aligned C output to: {output_path}")


if __name__ == "__main__":
    main()
