#!/usr/bin/env python3
"""
Analyze GTCRN model weight dimensions for C struct generation.
"""

import os
import sys

# Add gtcrn directory for imports
gtcrn_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, gtcrn_dir)

import torch
from gtcrn import GTCRN


def analyze_model():
    """Analyze model weight dimensions."""
    # Load model
    model = GTCRN().eval()
    checkpoint_path = os.path.join(gtcrn_dir, "checkpoints", "model_trained_on_dns3.tar")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    print("=" * 60)
    print("GTCRN Model Weight Analysis")
    print("=" * 60)

    total_params = 0

    # Iterate through all named parameters
    for name, param in model.named_parameters():
        size = param.numel()
        total_params += size
        print(f"{name}: shape={list(param.shape)}, size={size}")

    print()
    print("=" * 60)
    print("State Dict (includes running_mean/var for BatchNorm)")
    print("=" * 60)

    total_state = 0
    for name, tensor in model.state_dict().items():
        size = tensor.numel()
        total_state += size
        print(f"{name}: shape={list(tensor.shape)}, size={size}")

    print()
    print("=" * 60)
    print(f"Total trainable parameters: {total_params}")
    print(f"Total state dict entries: {total_state}")
    print(f"Expected binary file size: {total_state * 4 + 8} bytes (with 8-byte header)")
    print("=" * 60)

    # Detailed analysis of key modules
    print("\n=== Detailed Analysis ===\n")

    # ERB
    print("ERB module:")
    print(f"  erb_fc.weight: {list(model.erb.erb_fc.weight.shape)}")
    print(f"  ierb_fc.weight: {list(model.erb.ierb_fc.weight.shape)}")

    # DPGRNN
    print("\nDPGRNN1 Intra RNN:")
    print(f"  intra_rnn.rnn1 (bidirectional):")
    print(f"    weight_ih_l0: {list(model.dpgrnn1.intra_rnn.rnn1.weight_ih_l0.shape)}")
    print(f"    weight_hh_l0: {list(model.dpgrnn1.intra_rnn.rnn1.weight_hh_l0.shape)}")
    print(f"    weight_ih_l0_reverse: {list(model.dpgrnn1.intra_rnn.rnn1.weight_ih_l0_reverse.shape)}")
    print(f"    weight_hh_l0_reverse: {list(model.dpgrnn1.intra_rnn.rnn1.weight_hh_l0_reverse.shape)}")

    print("\nDPGRNN1 Inter RNN:")
    print(f"  inter_rnn.rnn1 (unidirectional):")
    print(f"    weight_ih_l0: {list(model.dpgrnn1.inter_rnn.rnn1.weight_ih_l0.shape)}")
    print(f"    weight_hh_l0: {list(model.dpgrnn1.inter_rnn.rnn1.weight_hh_l0.shape)}")

    print("\nDPGRNN FC layers:")
    print(f"  intra_fc.weight: {list(model.dpgrnn1.intra_fc.weight.shape)}")
    print(f"  inter_fc.weight: {list(model.dpgrnn1.inter_fc.weight.shape)}")

    print("\nDPGRNN LayerNorm:")
    print(f"  intra_ln.weight: {list(model.dpgrnn1.intra_ln.weight.shape)}")
    print(f"  inter_ln.weight: {list(model.dpgrnn1.inter_ln.weight.shape)}")

    # TRA
    print("\nTRA in GTConvBlock:")
    tra = model.encoder.en_convs[2].tra
    print(f"  att_gru: input_size={tra.att_gru.input_size}, hidden_size={tra.att_gru.hidden_size}")
    print(f"    weight_ih: {list(tra.att_gru.weight_ih_l0.shape)}")
    print(f"    weight_hh: {list(tra.att_gru.weight_hh_l0.shape)}")
    print(f"  att_fc.weight: {list(tra.att_fc.weight.shape)}")


if __name__ == "__main__":
    analyze_model()
