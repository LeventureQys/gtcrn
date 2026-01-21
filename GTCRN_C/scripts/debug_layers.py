#!/usr/bin/env python3
"""
GTCRN Layer-by-layer Debug Script

This script exports intermediate tensors from PyTorch model
to help debug the C implementation.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

# Add gtcrn directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN

def export_tensor_binary(tensor, filepath):
    """Export tensor to binary file for C comparison."""
    data = tensor.detach().cpu().numpy().astype(np.float32)
    with open(filepath, 'wb') as f:
        f.write(struct.pack('I', len(data.shape)))
        for dim in data.shape:
            f.write(struct.pack('I', dim))
        f.write(struct.pack(f'{data.size}f', *data.flatten()))
    return data

def main():
    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    input_wav = os.path.join(gtcrn_dir, "test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav")
    output_dir = os.path.join(project_dir, "debug_tensors")

    os.makedirs(output_dir, exist_ok=True)

    print("=== GTCRN Layer Debug ===\n")

    # Load model
    print(f"Loading model from {model_path}")
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # Load and prepare input
    print(f"Loading input: {input_wav}")
    audio_in, sr = sf.read(input_wav, dtype='float32')

    # Use only first 1 second for debugging
    audio_in = audio_in[:16000]

    window = torch.hann_window(512).pow(0.5)
    x = torch.from_numpy(audio_in)
    spec = torch.stft(x, 512, 256, 512, window, return_complex=False)[None]  # (1, F, T, 2)

    print(f"Input spec shape: {spec.shape}")

    # Save input spectrogram
    export_tensor_binary(spec, os.path.join(output_dir, "input_spec.bin"))

    # Step-by-step forward pass with intermediate exports
    spec_ref = spec  # (B,F,T,2)

    # Extract real/imag
    spec_real = spec[..., 0].permute(0,2,1)  # (B, T, F)
    spec_imag = spec[..., 1].permute(0,2,1)
    spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
    feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

    print(f"\nStep 1: Feature tensor: {feat.shape}")
    export_tensor_binary(feat, os.path.join(output_dir, "feat_input.bin"))

    # ERB compression
    feat_erb = model.erb.bm(feat)  # (B,3,T,129)
    print(f"Step 2: After ERB bm: {feat_erb.shape}")
    export_tensor_binary(feat_erb, os.path.join(output_dir, "feat_erb.bin"))

    # SFE
    feat_sfe = model.sfe(feat_erb)  # (B,9,T,129)
    print(f"Step 3: After SFE: {feat_sfe.shape}")
    export_tensor_binary(feat_sfe, os.path.join(output_dir, "feat_sfe.bin"))

    # Encoder
    en_conv0 = model.encoder.en_convs[0]
    en_conv1 = model.encoder.en_convs[1]

    # EnConv0: (9, T, 129) -> (16, T, 65)
    x = en_conv0(feat_sfe)
    print(f"Step 4: After EnConv0: {x.shape}")
    export_tensor_binary(x, os.path.join(output_dir, "en_conv0_out.bin"))

    # EnConv1: (16, T, 65) -> (16, T, 33)
    x = en_conv1(x)
    print(f"Step 5: After EnConv1: {x.shape}")
    export_tensor_binary(x, os.path.join(output_dir, "en_conv1_out.bin"))

    # EnGTConv2-4
    en_outs = [None, x.clone()]  # en_out0=None for simplicity, en_out1=current x

    x = model.encoder.en_convs[2](x)
    print(f"Step 6: After EnGTConv2: {x.shape}")
    export_tensor_binary(x, os.path.join(output_dir, "en_gt2_out.bin"))
    en_outs.append(x.clone())

    x = model.encoder.en_convs[3](x)
    print(f"Step 7: After EnGTConv3: {x.shape}")
    export_tensor_binary(x, os.path.join(output_dir, "en_gt3_out.bin"))
    en_outs.append(x.clone())

    x = model.encoder.en_convs[4](x)
    print(f"Step 8: After EnGTConv4: {x.shape}")
    export_tensor_binary(x, os.path.join(output_dir, "en_gt4_out.bin"))
    en_outs.append(x.clone())

    # DPGRNN 1 & 2
    x = model.dpgrnn1(x)
    print(f"Step 9: After DPGRNN1: {x.shape}")
    export_tensor_binary(x, os.path.join(output_dir, "dpgrnn1_out.bin"))

    x = model.dpgrnn2(x)
    print(f"Step 10: After DPGRNN2: {x.shape}")
    export_tensor_binary(x, os.path.join(output_dir, "dpgrnn2_out.bin"))

    # Decoder (with full processing)
    feat, en_outs = model.encoder(feat_sfe)
    feat = model.dpgrnn1(feat)
    feat = model.dpgrnn2(feat)
    m_feat = model.decoder(feat, en_outs)
    print(f"Step 11: After Decoder: {m_feat.shape}")
    export_tensor_binary(m_feat, os.path.join(output_dir, "decoder_out.bin"))

    # ERB expansion
    m = model.erb.bs(m_feat)
    print(f"Step 12: After ERB bs: {m.shape}")
    export_tensor_binary(m, os.path.join(output_dir, "mask.bin"))

    # Apply mask
    spec_enh = model.mask(m, spec_ref.permute(0,3,2,1))
    spec_enh = spec_enh.permute(0,3,2,1)
    print(f"Step 13: Enhanced spec: {spec_enh.shape}")
    export_tensor_binary(spec_enh, os.path.join(output_dir, "enhanced_spec.bin"))

    print(f"\nDebug tensors saved to: {output_dir}")

    # Print some statistics
    print("\n=== Tensor Statistics ===")
    print(f"Input spec: min={spec.min():.4f}, max={spec.max():.4f}, mean={spec.mean():.4f}")
    print(f"Mask: min={m.min():.4f}, max={m.max():.4f}, mean={m.mean():.4f}")
    print(f"Enhanced: min={spec_enh.min():.4f}, max={spec_enh.max():.4f}, mean={spec_enh.mean():.4f}")

    # Full inference for comparison
    with torch.no_grad():
        full_output = model(spec)

    print(f"\nFull inference output: {full_output.shape}")

    # Verify step-by-step matches full
    diff = (spec_enh - full_output).abs().max()
    print(f"Step-by-step vs full inference diff: {diff:.8f}")

if __name__ == "__main__":
    main()
