#!/usr/bin/env python3
"""
Compare C and PyTorch GTCRN outputs.
This script:
1. Runs PyTorch inference
2. Runs C inference
3. Compares the outputs
"""

import os
import sys
import numpy as np
import soundfile as sf
import subprocess

# Add parent directory to path for gtcrn import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from gtcrn import GTCRN

def run_pytorch_inference(input_wav, checkpoint_path, output_wav):
    """Run PyTorch inference and save output."""
    print("\n=== PyTorch Inference ===")

    # Load model
    model = GTCRN().eval()
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt)

    # Load audio
    audio, sr = sf.read(input_wav)
    print(f"Input: {len(audio)} samples, {sr} Hz")

    # STFT
    window = torch.hann_window(512).pow(0.5)
    x = torch.from_numpy(audio).float().unsqueeze(0)
    spec = torch.stft(x, 512, 256, 512, window, return_complex=False)
    print(f"STFT shape: {spec.shape}")  # (1, 257, T, 2)

    # Inference
    with torch.no_grad():
        spec_enh = model(spec)

    # ISTFT
    spec_enh_complex = torch.complex(spec_enh[..., 0], spec_enh[..., 1])
    y = torch.istft(spec_enh_complex, 512, 256, 512, window, length=len(audio))

    # Save
    output = y.squeeze().numpy()
    sf.write(output_wav, output, sr)
    print(f"Output saved to {output_wav}")

    return output

def run_c_inference(input_wav, weights_path, output_wav, demo_exe):
    """Run C inference."""
    print("\n=== C Inference ===")

    if not os.path.exists(demo_exe):
        print(f"Error: Demo executable not found: {demo_exe}")
        return None

    cmd = [demo_exe, weights_path, input_wav, output_wav]
    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error (code {result.returncode}):")
        print(result.stderr)
        return None

    if os.path.exists(output_wav):
        audio, sr = sf.read(output_wav)
        print(f"C output: {len(audio)} samples")
        return audio
    return None

def compare_outputs(pytorch_out, c_out, noisy_in):
    """Compare PyTorch and C outputs."""
    print("\n=== Comparison ===")

    min_len = min(len(pytorch_out), len(c_out), len(noisy_in))
    pytorch_out = pytorch_out[:min_len]
    c_out = c_out[:min_len]
    noisy_in = noisy_in[:min_len]

    # Basic stats
    print(f"\nNoisy input:  min={noisy_in.min():.4f}, max={noisy_in.max():.4f}, std={noisy_in.std():.4f}")
    print(f"PyTorch out:  min={pytorch_out.min():.4f}, max={pytorch_out.max():.4f}, std={pytorch_out.std():.4f}")
    print(f"C output:     min={c_out.min():.4f}, max={c_out.max():.4f}, std={c_out.std():.4f}")

    # Compare C vs Noisy (check if C is just passing through)
    c_vs_noisy = np.abs(c_out - noisy_in)
    print(f"\nC vs Noisy:   max_diff={c_vs_noisy.max():.6f}, mean_diff={c_vs_noisy.mean():.6f}")
    if c_vs_noisy.max() < 1e-5:
        print("  WARNING: C output equals noisy input (model not processing!)")

    # Compare C vs PyTorch
    c_vs_pytorch = np.abs(c_out - pytorch_out)
    mse = np.mean((c_out - pytorch_out) ** 2)
    print(f"C vs PyTorch: max_diff={c_vs_pytorch.max():.6f}, mean_diff={c_vs_pytorch.mean():.6f}, MSE={mse:.6f}")

    if c_vs_pytorch.max() < 0.01:
        print("  PASS: C and PyTorch outputs are nearly identical")
    elif c_vs_pytorch.max() < 0.1:
        print("  WARN: Small differences between C and PyTorch")
    else:
        print("  FAIL: Significant differences between C and PyTorch")

    # Compare PyTorch vs Noisy (verify enhancement is happening)
    pytorch_vs_noisy = np.abs(pytorch_out - noisy_in)
    print(f"PyTorch vs Noisy: max_diff={pytorch_vs_noisy.max():.6f}")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(script_dir, '..', '..')
    c_project_dir = os.path.join(script_dir, '..')

    input_wav = os.path.join(project_dir, 'test_wavs', 'noisy_16k', '00027_1_keyboard_loud_snr+5dB_noisy.wav')
    checkpoint = os.path.join(project_dir, 'checkpoints', 'model_trained_on_dns3.tar')

    pytorch_output = os.path.join(project_dir, 'test_wavs', 'output_c', 'pytorch_output.wav')
    c_output = os.path.join(project_dir, 'test_wavs', 'output_c', 'c_output.wav')

    weights_path = os.path.join(c_project_dir, 'weights', 'gtcrn_weights.bin')
    demo_exe = os.path.join(c_project_dir, 'build', 'Release', 'gtcrn_demo.exe')

    # Check files exist
    if not os.path.exists(input_wav):
        print(f"Error: Input wav not found: {input_wav}")
        return
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found: {checkpoint}")
        return

    # Load noisy input
    noisy_in, sr = sf.read(input_wav)

    # Run PyTorch
    pytorch_out = run_pytorch_inference(input_wav, checkpoint, pytorch_output)

    # Run C
    if os.path.exists(demo_exe) and os.path.exists(weights_path):
        c_out = run_c_inference(input_wav, weights_path, c_output, demo_exe)
        if c_out is not None:
            compare_outputs(pytorch_out, c_out, noisy_in)
    else:
        print(f"\nC demo not found. Skipping C comparison.")
        print(f"  Expected: {demo_exe}")
        print(f"  Weights:  {weights_path}")

if __name__ == "__main__":
    main()
