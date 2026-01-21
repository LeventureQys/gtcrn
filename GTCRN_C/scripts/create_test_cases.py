#!/usr/bin/env python3
"""
Create test case for individual C layers.
Generates input, weights, and expected output for each layer.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(project_dir)
sys.path.insert(0, gtcrn_dir)

def save_tensor(tensor, filepath):
    """Save tensor to binary file."""
    data = tensor.detach().cpu().numpy().astype(np.float32).flatten()
    with open(filepath, 'wb') as f:
        f.write(struct.pack(f'{len(data)}f', *data))
    return data

def test_conv_transpose2d():
    """Test ConvTranspose2d with dilation."""
    print("=== Testing ConvTranspose2d ===")
    output_dir = os.path.join(project_dir, "test_data")
    os.makedirs(output_dir, exist_ok=True)

    # Decoder depth conv: ConvTranspose2d(16, 16, (3,3), stride=(1,1),
    #                                      padding=(2*dilation, 1), dilation=(dilation,1), groups=16)
    dilation = 2

    deconv = nn.ConvTranspose2d(16, 16, (3, 3), stride=(1, 1),
                                 padding=(2*dilation, 1),
                                 dilation=(dilation, 1), groups=16)

    # Input: (1, 16, 14, 33) - after F.pad with pad_size=4
    torch.manual_seed(42)
    x = torch.randn(1, 16, 14, 33)

    with torch.no_grad():
        y = deconv(x)

    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Weight shape: {deconv.weight.shape}")

    # Save
    save_tensor(x, os.path.join(output_dir, "deconv_input.bin"))
    save_tensor(deconv.weight, os.path.join(output_dir, "deconv_weight.bin"))
    save_tensor(deconv.bias, os.path.join(output_dir, "deconv_bias.bin"))
    save_tensor(y, os.path.join(output_dir, "deconv_output.bin"))

    print(f"Output first 10: {y.flatten()[:10].tolist()}")
    print()

def test_layernorm():
    """Test LayerNorm."""
    print("=== Testing LayerNorm ===")
    output_dir = os.path.join(project_dir, "test_data")

    # DPGRNN LayerNorm: nn.LayerNorm((33, 16), eps=1e-8)
    ln = nn.LayerNorm((33, 16), eps=1e-8)

    torch.manual_seed(42)
    x = torch.randn(10, 33, 16)  # (batch*time, freq, channels)

    with torch.no_grad():
        y = ln(x)

    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"gamma shape: {ln.weight.shape}")
    print(f"beta shape: {ln.bias.shape}")

    save_tensor(x, os.path.join(output_dir, "ln_input.bin"))
    save_tensor(ln.weight, os.path.join(output_dir, "ln_gamma.bin"))
    save_tensor(ln.bias, os.path.join(output_dir, "ln_beta.bin"))
    save_tensor(y, os.path.join(output_dir, "ln_output.bin"))

    print(f"Output first 10: {y.flatten()[:10].tolist()}")
    print()

def test_bidirectional_gru():
    """Test bidirectional GRU."""
    print("=== Testing Bidirectional GRU ===")
    output_dir = os.path.join(project_dir, "test_data")

    # GRNN: bidirectional GRU(input=8, hidden=4)
    gru = nn.GRU(8, 4, num_layers=1, batch_first=True, bidirectional=True)

    torch.manual_seed(42)
    x = torch.randn(1, 33, 8)  # (batch, seq_len, input)

    with torch.no_grad():
        y, h = gru(x)

    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")  # (1, 33, 8) - 4*2 for bidirectional
    print(f"Hidden: {h.shape}")

    # Save weights - note PyTorch's weight layout
    print(f"weight_ih_l0 shape: {gru.weight_ih_l0.shape}")  # (3*hidden, input)
    print(f"weight_hh_l0 shape: {gru.weight_hh_l0.shape}")  # (3*hidden, hidden)

    save_tensor(x, os.path.join(output_dir, "bigru_input.bin"))
    save_tensor(gru.weight_ih_l0, os.path.join(output_dir, "bigru_ih_fwd.bin"))
    save_tensor(gru.weight_hh_l0, os.path.join(output_dir, "bigru_hh_fwd.bin"))
    save_tensor(gru.bias_ih_l0, os.path.join(output_dir, "bigru_bih_fwd.bin"))
    save_tensor(gru.bias_hh_l0, os.path.join(output_dir, "bigru_bhh_fwd.bin"))
    save_tensor(gru.weight_ih_l0_reverse, os.path.join(output_dir, "bigru_ih_rev.bin"))
    save_tensor(gru.weight_hh_l0_reverse, os.path.join(output_dir, "bigru_hh_rev.bin"))
    save_tensor(gru.bias_ih_l0_reverse, os.path.join(output_dir, "bigru_bih_rev.bin"))
    save_tensor(gru.bias_hh_l0_reverse, os.path.join(output_dir, "bigru_bhh_rev.bin"))
    save_tensor(y, os.path.join(output_dir, "bigru_output.bin"))

    print(f"Output first 10: {y.flatten()[:10].tolist()}")
    print()

def test_simple_encoder():
    """Test simple encoder path to find where divergence starts."""
    print("=== Testing Simple Encoder Path ===")

    from gtcrn import GTCRN, SFE, ERB

    model_path = os.path.join(gtcrn_dir, "checkpoints/model_trained_on_dns3.tar")
    model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    output_dir = os.path.join(project_dir, "test_data")
    os.makedirs(output_dir, exist_ok=True)

    # Small test: 10 frames
    torch.manual_seed(42)
    spec = torch.randn(1, 257, 10, 2) * 0.1

    with torch.no_grad():
        # Step 1: Feature extraction
        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)

        print(f"1. Feature: {feat.shape}, sum={feat.sum():.6f}")
        save_tensor(spec_real, os.path.join(output_dir, "spec_real.bin"))
        save_tensor(spec_imag, os.path.join(output_dir, "spec_imag.bin"))
        save_tensor(feat, os.path.join(output_dir, "feat.bin"))

        # Step 2: ERB
        feat_erb = model.erb.bm(feat)
        print(f"2. ERB: {feat_erb.shape}, sum={feat_erb.sum():.6f}")
        save_tensor(feat_erb, os.path.join(output_dir, "feat_erb.bin"))

        # Step 3: SFE
        feat_sfe = model.sfe(feat_erb)
        print(f"3. SFE: {feat_sfe.shape}, sum={feat_sfe.sum():.6f}")
        save_tensor(feat_sfe, os.path.join(output_dir, "feat_sfe.bin"))

        # Step 4: EnConv0
        en0 = model.encoder.en_convs[0](feat_sfe)
        print(f"4. EnConv0: {en0.shape}, sum={en0.sum():.6f}")
        save_tensor(en0, os.path.join(output_dir, "en_conv0.bin"))

        # Step 5: EnConv1
        en1 = model.encoder.en_convs[1](en0)
        print(f"5. EnConv1: {en1.shape}, sum={en1.sum():.6f}")
        save_tensor(en1, os.path.join(output_dir, "en_conv1.bin"))

        # Step 6: EnGT2
        en2 = model.encoder.en_convs[2](en1)
        print(f"6. EnGT2: {en2.shape}, sum={en2.sum():.6f}")
        save_tensor(en2, os.path.join(output_dir, "en_gt2.bin"))

        # Full output
        output = model(spec)
        print(f"\nFull output: {output.shape}, sum={output.sum():.6f}")
        save_tensor(output, os.path.join(output_dir, "full_output.bin"))

    print()

def main():
    test_conv_transpose2d()
    test_layernorm()
    test_bidirectional_gru()
    test_simple_encoder()

if __name__ == "__main__":
    main()
