"""
Compare streaming inference results between C and Python implementations.
"""
import os
import sys
import time
import numpy as np
import torch
import soundfile as sf

# Add parent directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)

from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream

def run_python_streaming(input_path, output_path, model_path):
    """Run Python streaming inference."""
    print(f"[Python Streaming] Processing: {input_path}")

    # Load audio
    audio_data, sr = sf.read(input_path, dtype='float32')
    assert sr == 16000, f"Sample rate must be 16000, got {sr}"

    device = torch.device("cpu")

    # Load offline model
    model = GTCRN().to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])

    # Convert to streaming model
    stream_model = StreamGTCRN().to(device).eval()
    convert_to_stream(stream_model, model)

    # STFT parameters
    window = torch.hann_window(512).pow(0.5)

    # Convert to tensor and do STFT
    x = torch.from_numpy(audio_data)
    x = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]  # (1, F, T, 2)

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33).to(device)
    tra_cache = torch.zeros(2, 3, 1, 1, 16).to(device)
    inter_cache = torch.zeros(2, 1, 33, 16).to(device)

    # Streaming inference
    ys = []
    times = []
    n_frames = x.shape[2]

    print(f"  Processing {n_frames} frames...")
    for i in range(n_frames):
        xi = x[:, :, i:i+1]
        tic = time.perf_counter()
        with torch.no_grad():
            yi, conv_cache, tra_cache, inter_cache = stream_model(xi, conv_cache, tra_cache, inter_cache)
        toc = time.perf_counter()
        times.append((toc - tic) * 1000)
        ys.append(yi)

    ys = torch.cat(ys, dim=2)

    # ISTFT
    ys_complex = torch.complex(ys[0, :, :, 0], ys[0, :, :, 1])
    y = torch.istft(ys_complex, 512, 256, 512, window).detach().cpu().numpy()

    # Save output
    sf.write(output_path, y.squeeze(), 16000)

    print(f"  Mean inference time: {sum(times)/len(times):.2f}ms per frame")
    print(f"  Output saved to: {output_path}")

    return y.squeeze()

def run_python_offline(input_path, output_path, model_path):
    """Run Python offline inference for reference."""
    print(f"[Python Offline] Processing: {input_path}")

    # Load audio
    audio_data, sr = sf.read(input_path, dtype='float32')
    assert sr == 16000, f"Sample rate must be 16000, got {sr}"

    device = torch.device("cpu")

    # Load model
    model = GTCRN().to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])

    # STFT parameters
    window = torch.hann_window(512).pow(0.5)

    # Convert to tensor and do STFT
    x = torch.from_numpy(audio_data)
    x = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]  # (1, F, T, 2)

    # Offline inference
    with torch.no_grad():
        y = model(x)  # (1, F, T, 2)

    # ISTFT
    y_complex = torch.complex(y[0, :, :, 0], y[0, :, :, 1])
    y = torch.istft(y_complex, 512, 256, 512, window).detach().cpu().numpy()

    # Save output
    sf.write(output_path, y.squeeze(), 16000)
    print(f"  Output saved to: {output_path}")

    return y.squeeze()

def compare_outputs(py_output, c_output_path, py_offline_output=None):
    """Compare Python and C outputs."""
    print("\n" + "="*60)
    print("Comparison Results")
    print("="*60)

    # Load C output
    if os.path.exists(c_output_path):
        c_output, sr = sf.read(c_output_path, dtype='float32')
        print(f"C output loaded: {len(c_output)} samples")
    else:
        print(f"C output not found: {c_output_path}")
        return

    print(f"Python streaming output: {len(py_output)} samples")

    # Align lengths
    min_len = min(len(py_output), len(c_output))
    py_output = py_output[:min_len]
    c_output = c_output[:min_len]

    # Compute metrics
    diff = py_output - c_output
    max_abs_diff = np.abs(diff).max()
    mean_abs_diff = np.abs(diff).mean()
    rmse = np.sqrt(np.mean(diff**2))

    # SNR-like metric
    signal_power = np.mean(py_output**2)
    noise_power = np.mean(diff**2)
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')

    # Correlation
    corr = np.corrcoef(py_output, c_output)[0, 1]

    print(f"\n[Python Streaming vs C Streaming]")
    print(f"  Max absolute difference: {max_abs_diff:.6f}")
    print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  SNR (dB): {snr:.2f}")
    print(f"  Correlation: {corr:.6f}")

    # Compare with offline if available
    if py_offline_output is not None:
        py_offline_output = py_offline_output[:min_len]
        diff_stream_offline = py_output - py_offline_output
        max_stream_offline = np.abs(diff_stream_offline).max()
        print(f"\n[Python Streaming vs Python Offline]")
        print(f"  Max absolute difference: {max_stream_offline:.6f}")

        diff_c_offline = c_output - py_offline_output
        max_c_offline = np.abs(diff_c_offline).max()
        print(f"\n[C Streaming vs Python Offline]")
        print(f"  Max absolute difference: {max_c_offline:.6f}")

    # Verdict
    print("\n" + "-"*60)
    if max_abs_diff < 0.01:
        print("[PASS] C and Python outputs are very close!")
    elif max_abs_diff < 0.2:
        print("[GOOD] C and Python outputs are reasonably close (float precision diff)")
    else:
        print("[FAIL] C and Python outputs are significantly different")
    print("-"*60)

if __name__ == "__main__":
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
    model_path = os.path.join(base_dir, "checkpoints/model_trained_on_dns3.tar")

    # Output paths
    py_stream_output = os.path.join(base_dir, "test_wavs/output_c/py_streaming.wav")
    py_offline_output_path = os.path.join(base_dir, "test_wavs/output_c/py_offline.wav")
    c_output = os.path.join(base_dir, "test_wavs/output_c/c_streaming.wav")

    # Create output directory
    os.makedirs(os.path.dirname(py_stream_output), exist_ok=True)

    # Check if model exists
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = os.path.join(base_dir, "onnx_models/model_trained_on_dns3.tar")

    print(f"Model path: {model_path}")
    print(f"Input file: {input_file}")
    print()

    # Run Python offline (reference)
    py_offline = run_python_offline(input_file, py_offline_output_path, model_path)

    print()

    # Run Python streaming
    py_stream = run_python_streaming(input_file, py_stream_output, model_path)

    print()

    # Compare
    compare_outputs(py_stream, c_output, py_offline)
