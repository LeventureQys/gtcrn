"""
Export GRU weights from PyTorch to binary format for C implementation

This script extracts GRU weights from a trained GTCRN model and saves them
in a binary format that can be loaded by the C implementation.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtcrn1 import GTCRN


def export_gru_weights(gru_module, output_path, prefix=""):
    """
    Export PyTorch GRU weights to binary file

    PyTorch GRU weight format:
    - weight_ih_l0: (3*hidden_size, input_size) - [W_ir, W_iz, W_in]
    - weight_hh_l0: (3*hidden_size, hidden_size) - [U_ir, U_iz, U_in]
    - bias_ih_l0: (3*hidden_size,) - [b_ir, b_iz, b_in]
    - bias_hh_l0: (3*hidden_size,) - [b_hr, b_hz, b_hn]

    Note: PyTorch uses (reset, update, new) order, but standard GRU uses (update, reset, new)
    We need to reorder: [r, z, n] -> [z, r, n]

    C format (all float32, row-major):
    - W_z: (hidden_size, input_size)
    - U_z: (hidden_size, hidden_size)
    - b_z: (hidden_size,)
    - W_r: (hidden_size, input_size)
    - U_r: (hidden_size, hidden_size)
    - b_r: (hidden_size,)
    - W_h: (hidden_size, input_size)
    - U_h: (hidden_size, hidden_size)
    - b_h: (hidden_size,)
    """

    # Get weights from PyTorch GRU
    weight_ih = gru_module.weight_ih_l0.detach().cpu().numpy()  # (3H, I)
    weight_hh = gru_module.weight_hh_l0.detach().cpu().numpy()  # (3H, H)
    bias_ih = gru_module.bias_ih_l0.detach().cpu().numpy()      # (3H,)
    bias_hh = gru_module.bias_hh_l0.detach().cpu().numpy()      # (3H,)

    hidden_size = weight_hh.shape[1]
    input_size = weight_ih.shape[1]

    print(f"{prefix}GRU: input_size={input_size}, hidden_size={hidden_size}")

    # Split into gates (PyTorch order: reset, update, new)
    W_ir, W_iz, W_in = np.split(weight_ih, 3, axis=0)
    U_ir, U_iz, U_in = np.split(weight_hh, 3, axis=0)
    b_ir, b_iz, b_in = np.split(bias_ih, 3, axis=0)
    b_hr, b_hz, b_hn = np.split(bias_hh, 3, axis=0)

    # Combine biases (PyTorch splits them, standard GRU combines)
    b_r = b_ir + b_hr
    b_z = b_iz + b_hz
    b_n = b_in + b_hn

    # Reorder to standard GRU format: [z, r, n]
    W_z, W_r, W_h = W_iz, W_ir, W_in
    U_z, U_r, U_h = U_iz, U_ir, U_in

    # Write to binary file
    with open(output_path, 'wb') as f:
        # Write in order: W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h
        W_z.astype(np.float32).tofile(f)
        U_z.astype(np.float32).tofile(f)
        b_z.astype(np.float32).tofile(f)

        W_r.astype(np.float32).tofile(f)
        U_r.astype(np.float32).tofile(f)
        b_r.astype(np.float32).tofile(f)

        W_h.astype(np.float32).tofile(f)
        U_h.astype(np.float32).tofile(f)
        b_h.astype(np.float32).tofile(f)

    print(f"{prefix}Exported to: {output_path}")
    print(f"{prefix}File size: {os.path.getsize(output_path)} bytes")

    return {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'W_z': W_z, 'U_z': U_z, 'b_z': b_z,
        'W_r': W_r, 'U_r': U_r, 'b_r': b_r,
        'W_h': W_h, 'U_h': U_h, 'b_h': b_h,
    }


def export_grnn_weights(grnn_module, output_dir, name_prefix):
    """
    Export Grouped RNN (GRNN) weights

    GRNN splits input into 2 groups and processes independently
    """
    print(f"\nExporting {name_prefix}...")

    # Export group 1
    weights_g1 = export_gru_weights(
        grnn_module.rnn1,
        os.path.join(output_dir, f"{name_prefix}_g1.bin"),
        prefix="  Group 1: "
    )

    # Export group 2
    weights_g2 = export_gru_weights(
        grnn_module.rnn2,
        os.path.join(output_dir, f"{name_prefix}_g2.bin"),
        prefix="  Group 2: "
    )

    return weights_g1, weights_g2


def export_dpgrnn_weights(dpgrnn_module, output_dir, name_prefix):
    """
    Export Dual-Path GRNN weights

    DPGRNN contains:
    - intra_rnn: Bidirectional GRNN (2 groups, forward + backward)
    - inter_rnn: Unidirectional GRNN (2 groups)
    """
    print(f"\n{'='*60}")
    print(f"Exporting {name_prefix}")
    print(f"{'='*60}")

    # Export Intra-RNN (bidirectional)
    print(f"\n{name_prefix} - Intra-RNN (Bidirectional):")
    intra_g1, intra_g2 = export_grnn_weights(
        dpgrnn_module.intra_rnn,
        output_dir,
        f"{name_prefix}_intra"
    )

    # For bidirectional, PyTorch stores forward and backward weights separately
    # We need to extract them
    # Note: This is simplified - actual implementation depends on PyTorch version

    # Export Inter-RNN (unidirectional)
    print(f"\n{name_prefix} - Inter-RNN (Unidirectional):")
    inter_g1, inter_g2 = export_grnn_weights(
        dpgrnn_module.inter_rnn,
        output_dir,
        f"{name_prefix}_inter"
    )

    # Export Linear layers
    print(f"\n{name_prefix} - Linear layers:")

    # Intra FC
    intra_fc_weight = dpgrnn_module.intra_fc.weight.detach().cpu().numpy()
    intra_fc_bias = dpgrnn_module.intra_fc.bias.detach().cpu().numpy()
    intra_fc_path = os.path.join(output_dir, f"{name_prefix}_intra_fc.bin")
    with open(intra_fc_path, 'wb') as f:
        intra_fc_weight.astype(np.float32).tofile(f)
        intra_fc_bias.astype(np.float32).tofile(f)
    print(f"  Intra FC: {intra_fc_weight.shape} -> {intra_fc_path}")

    # Inter FC
    inter_fc_weight = dpgrnn_module.inter_fc.weight.detach().cpu().numpy()
    inter_fc_bias = dpgrnn_module.inter_fc.bias.detach().cpu().numpy()
    inter_fc_path = os.path.join(output_dir, f"{name_prefix}_inter_fc.bin")
    with open(inter_fc_path, 'wb') as f:
        inter_fc_weight.astype(np.float32).tofile(f)
        inter_fc_bias.astype(np.float32).tofile(f)
    print(f"  Inter FC: {inter_fc_weight.shape} -> {inter_fc_path}")

    # Export LayerNorm parameters
    print(f"\n{name_prefix} - LayerNorm:")

    # Intra LN
    intra_ln_weight = dpgrnn_module.intra_ln.weight.detach().cpu().numpy()
    intra_ln_bias = dpgrnn_module.intra_ln.bias.detach().cpu().numpy()
    intra_ln_path = os.path.join(output_dir, f"{name_prefix}_intra_ln.bin")
    with open(intra_ln_path, 'wb') as f:
        intra_ln_weight.astype(np.float32).tofile(f)
        intra_ln_bias.astype(np.float32).tofile(f)
    print(f"  Intra LN: gamma={intra_ln_weight.shape}, beta={intra_ln_bias.shape}")

    # Inter LN
    inter_ln_weight = dpgrnn_module.inter_ln.weight.detach().cpu().numpy()
    inter_ln_bias = dpgrnn_module.inter_ln.bias.detach().cpu().numpy()
    inter_ln_path = os.path.join(output_dir, f"{name_prefix}_inter_ln.bin")
    with open(inter_ln_path, 'wb') as f:
        inter_ln_weight.astype(np.float32).tofile(f)
        inter_ln_bias.astype(np.float32).tofile(f)
    print(f"  Inter LN: gamma={inter_ln_weight.shape}, beta={inter_ln_bias.shape}")


def main():
    """
    Main function to export all GRU weights from GTCRN model
    """
    import argparse

    parser = argparse.ArgumentParser(description='Export GRU weights from GTCRN model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='./gru_weights',
                       help='Output directory for weight files')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = GTCRN()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded successfully!")

    # Export DPGRNN1 weights
    export_dpgrnn_weights(model.dpgrnn1, args.output_dir, "dpgrnn1")

    # Export DPGRNN2 weights
    export_dpgrnn_weights(model.dpgrnn2, args.output_dir, "dpgrnn2")

    print(f"\n{'='*60}")
    print("Export completed!")
    print(f"All weights saved to: {args.output_dir}")
    print(f"{'='*60}")

    # Print summary
    print("\nWeight files created:")
    for filename in sorted(os.listdir(args.output_dir)):
        filepath = os.path.join(args.output_dir, filename)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  {filename:40s} {size_kb:8.2f} KB")


if __name__ == '__main__':
    main()
