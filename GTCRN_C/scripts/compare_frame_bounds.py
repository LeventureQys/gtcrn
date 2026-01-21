#!/usr/bin/env python3
"""
Compare outputs around frame boundaries.
"""

import os
import numpy as np
import soundfile as sf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)

def main():
    c_complete_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_complete.wav")
    c_stream_path = os.path.join(gtcrn_dir, "test_wavs/output_16k/enhanced_c_stream.wav")

    c_complete, _ = sf.read(c_complete_path)
    c_stream, _ = sf.read(c_stream_path)

    hop = 256
    for frame in range(10):
        start = frame * hop
        end = start + hop

        complete_segment = c_complete[start:end]
        stream_segment = c_stream[start:end]

        complete_sum = np.sum(np.abs(complete_segment))
        stream_sum = np.sum(np.abs(stream_segment))

        print(f"Frame {frame} (samples {start}-{end}): "
              f"complete sum={complete_sum:.4f}, stream sum={stream_sum:.4f}")


if __name__ == "__main__":
    main()
