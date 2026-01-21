"""
Check if there's a time offset between C and Python outputs.
"""
import os
import numpy as np
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))

# Load outputs
c_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_c_stream.wav')
py_output_path = os.path.join(gtcrn_dir, 'test_wavs/output_py_stream_proper.wav')

c_out, sr = sf.read(c_output_path)
py_out, _ = sf.read(py_output_path)

# Check cross-correlation to find optimal offset
def cross_correlate(a, b, max_offset=5000):
    """Find the offset that maximizes correlation."""
    best_corr = -1
    best_offset = 0

    for offset in range(-max_offset, max_offset + 1, 100):
        if offset > 0:
            a_slice = a[offset:]
            b_slice = b[:len(a_slice)]
        elif offset < 0:
            a_slice = a[:offset]
            b_slice = b[-offset:len(a_slice)-offset]
        else:
            min_len = min(len(a), len(b))
            a_slice = a[:min_len]
            b_slice = b[:min_len]

        min_len = min(len(a_slice), len(b_slice))
        if min_len > 1000:
            corr = np.corrcoef(a_slice[:min_len], b_slice[:min_len])[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_offset = offset

    # Fine-tune around best offset
    for offset in range(best_offset - 100, best_offset + 100):
        if offset > 0:
            a_slice = a[offset:]
            b_slice = b[:len(a_slice)]
        elif offset < 0:
            a_slice = a[:offset]
            b_slice = b[-offset:len(a_slice)-offset]
        else:
            min_len = min(len(a), len(b))
            a_slice = a[:min_len]
            b_slice = b[:min_len]

        min_len = min(len(a_slice), len(b_slice))
        if min_len > 1000:
            corr = np.corrcoef(a_slice[:min_len], b_slice[:min_len])[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_offset = offset

    return best_offset, best_corr

print("Searching for optimal offset...")
offset, corr = cross_correlate(c_out, py_out)
print(f"Best offset: {offset} samples ({offset/sr*1000:.1f} ms)")
print(f"Correlation at best offset: {corr:.6f}")

# Try frame-aligned offsets
print("\nChecking frame-aligned offsets (multiples of 256 samples):")
for frame_off in range(-5, 6):
    offset = frame_off * 256
    if offset >= 0:
        c_slice = c_out[offset:]
        p_slice = py_out[:len(c_slice)]
    else:
        c_slice = c_out[:-offset]
        p_slice = py_out[-offset:]

    min_len = min(len(c_slice), len(p_slice))
    if min_len > 1000:
        corr = np.corrcoef(c_slice[:min_len], p_slice[:min_len])[0, 1]
        print(f"  Offset {frame_off:+d} frames ({offset:+5d} samples): corr = {corr:.6f}")

# Check where C output first becomes non-zero
first_nonzero = np.where(np.abs(c_out) > 1e-6)[0]
if len(first_nonzero) > 0:
    print(f"\nC output first non-zero at sample {first_nonzero[0]}")
else:
    print("\nC output is all zeros!")

first_nonzero_py = np.where(np.abs(py_out) > 1e-6)[0]
if len(first_nonzero_py) > 0:
    print(f"Python output first non-zero at sample {first_nonzero_py[0]}")
