"""
Export PyTorch GTCRN model weights to binary format for C implementation

Usage:
    python export_weights.py --model model.pth --output weights/
"""

import torch
import numpy as np
import os
import argparse
from pathlib import Path


def export_conv2d(conv, prefix, output_dir):
    """Export Conv2d layer weights"""
    weight = conv.weight.detach().cpu().numpy().astype(np.float32)
    weight.tofile(os.path.join(output_dir, f"{prefix}_weight.bin"))

    if conv.bias is not None:
        bias = conv.bias.detach().cpu().numpy().astype(np.float32)
        bias.tofile(os.path.join(output_dir, f"{prefix}_bias.bin"))

    print(f"Exported Conv2d: {prefix}, weight shape: {weight.shape}")


def export_batchnorm2d(bn, prefix, output_dir):
    """Export BatchNorm2d layer weights"""
    weight = bn.weight.detach().cpu().numpy().astype(np.float32)
    bias = bn.bias.detach().cpu().numpy().astype(np.float32)
    running_mean = bn.running_mean.detach().cpu().numpy().astype(np.float32)
    running_var = bn.running_var.detach().cpu().numpy().astype(np.float32)

    weight.tofile(os.path.join(output_dir, f"{prefix}_weight.bin"))
    bias.tofile(os.path.join(output_dir, f"{prefix}_bias.bin"))
    running_mean.tofile(os.path.join(output_dir, f"{prefix}_running_mean.bin"))
    running_var.tofile(os.path.join(output_dir, f"{prefix}_running_var.bin"))

    print(f"Exported BatchNorm2d: {prefix}, num_features: {len(weight)}")


def export_linear(linear, prefix, output_dir):
    """Export Linear layer weights"""
    weight = linear.weight.detach().cpu().numpy().astype(np.float32)
    weight.tofile(os.path.join(output_dir, f"{prefix}_weight.bin"))

    if linear.bias is not None:
        bias = linear.bias.detach().cpu().numpy().astype(np.float32)
        bias.tofile(os.path.join(output_dir, f"{prefix}_bias.bin"))

    print(f"Exported Linear: {prefix}, weight shape: {weight.shape}")


def export_prelu(prelu, prefix, output_dir):
    """Export PReLU layer weights"""
    weight = prelu.weight.detach().cpu().numpy().astype(np.float32)
    weight.tofile(os.path.join(output_dir, f"{prefix}_weight.bin"))

    print(f"Exported PReLU: {prefix}, num_parameters: {len(weight)}")


def export_layernorm(ln, prefix, output_dir):
    """Export LayerNorm layer weights"""
    if ln.weight is not None:
        weight = ln.weight.detach().cpu().numpy().astype(np.float32)
        weight.tofile(os.path.join(output_dir, f"{prefix}_weight.bin"))

    if ln.bias is not None:
        bias = ln.bias.detach().cpu().numpy().astype(np.float32)
        bias.tofile(os.path.join(output_dir, f"{prefix}_bias.bin"))

    print(f"Exported LayerNorm: {prefix}")


def export_gru(gru, prefix, output_dir):
    """
    Export GRU layer weights

    PyTorch GRU weight format:
    - weight_ih_l0: [3*hidden_size, input_size] (W_z, W_r, W_h stacked)
    - weight_hh_l0: [3*hidden_size, hidden_size] (U_z, U_r, U_h stacked)
    - bias_ih_l0: [3*hidden_size] (b_z, b_r, b_h stacked)
    - bias_hh_l0: [3*hidden_size] (usually zeros, combined with bias_ih)
    """
    hidden_size = gru.hidden_size
    input_size = gru.input_size

    # Get weights
    weight_ih = gru.weight_ih_l0.detach().cpu().numpy().astype(np.float32)
    weight_hh = gru.weight_hh_l0.detach().cpu().numpy().astype(np.float32)
    bias_ih = gru.bias_ih_l0.detach().cpu().numpy().astype(np.float32)
    bias_hh = gru.bias_hh_l0.detach().cpu().numpy().astype(np.float32)

    # Split into update, reset, and new gates
    # PyTorch order: reset, update, new (r, z, h)
    # Our order: update, reset, new (z, r, h)

    # Extract reset gate (r)
    W_r = weight_ih[0:hidden_size, :]
    U_r = weight_hh[0:hidden_size, :]
    b_r = bias_ih[0:hidden_size] + bias_hh[0:hidden_size]

    # Extract update gate (z)
    W_z = weight_ih[hidden_size:2*hidden_size, :]
    U_z = weight_hh[hidden_size:2*hidden_size, :]
    b_z = bias_ih[hidden_size:2*hidden_size] + bias_hh[hidden_size:2*hidden_size]

    # Extract new gate (h)
    W_h = weight_ih[2*hidden_size:3*hidden_size, :]
    U_h = weight_hh[2*hidden_size:3*hidden_size, :]
    b_h = bias_ih[2*hidden_size:3*hidden_size] + bias_hh[2*hidden_size:3*hidden_size]

    # Save in our format
    W_z.tofile(os.path.join(output_dir, f"{prefix}_W_z.bin"))
    U_z.tofile(os.path.join(output_dir, f"{prefix}_U_z.bin"))
    b_z.tofile(os.path.join(output_dir, f"{prefix}_b_z.bin"))

    W_r.tofile(os.path.join(output_dir, f"{prefix}_W_r.bin"))
    U_r.tofile(os.path.join(output_dir, f"{prefix}_U_r.bin"))
    b_r.tofile(os.path.join(output_dir, f"{prefix}_b_r.bin"))

    W_h.tofile(os.path.join(output_dir, f"{prefix}_W_h.bin"))
    U_h.tofile(os.path.join(output_dir, f"{prefix}_U_h.bin"))
    b_h.tofile(os.path.join(output_dir, f"{prefix}_b_h.bin"))

    print(f"Exported GRU: {prefix}, input_size: {input_size}, hidden_size: {hidden_size}")


def export_convblock(block, prefix, output_dir):
    """Export ConvBlock (Conv + BN + Activation)"""
    export_conv2d(block.conv, f"{prefix}_conv", output_dir)
    export_batchnorm2d(block.bn, f"{prefix}_bn", output_dir)

    if hasattr(block, 'act') and isinstance(block.act, torch.nn.PReLU):
        export_prelu(block.act, f"{prefix}_prelu", output_dir)


def export_gtconvblock(block, prefix, output_dir):
    """Export GTConvBlock"""
    # Point Conv1
    export_conv2d(block.point_conv1, f"{prefix}_point_conv1", output_dir)
    export_batchnorm2d(block.point_bn1, f"{prefix}_point_bn1", output_dir)
    export_prelu(block.point_act, f"{prefix}_point_prelu", output_dir)

    # Depth Conv
    export_conv2d(block.depth_conv, f"{prefix}_depth_conv", output_dir)
    export_batchnorm2d(block.depth_bn, f"{prefix}_depth_bn", output_dir)
    export_prelu(block.depth_act, f"{prefix}_depth_prelu", output_dir)

    # Point Conv2
    export_conv2d(block.point_conv2, f"{prefix}_point_conv2", output_dir)
    export_batchnorm2d(block.point_bn2, f"{prefix}_point_bn2", output_dir)

    # TRA
    if hasattr(block, 'tra'):
        export_gru(block.tra.att_gru, f"{prefix}_tra_gru", output_dir)
        export_linear(block.tra.att_fc, f"{prefix}_tra_fc", output_dir)


def export_encoder(encoder, output_dir):
    """Export Encoder"""
    encoder_dir = os.path.join(output_dir, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)

    # Conv blocks
    export_convblock(encoder.conv1, "conv1", encoder_dir)
    export_convblock(encoder.conv2, "conv2", encoder_dir)

    # GTConv blocks
    export_gtconvblock(encoder.gtconv1, "gtconv1", encoder_dir)
    export_gtconvblock(encoder.gtconv2, "gtconv2", encoder_dir)
    export_gtconvblock(encoder.gtconv3, "gtconv3", encoder_dir)

    print(f"Exported Encoder to {encoder_dir}")


def export_decoder(decoder, output_dir):
    """Export Decoder"""
    decoder_dir = os.path.join(output_dir, "decoder")
    os.makedirs(decoder_dir, exist_ok=True)

    # GTConv blocks
    export_gtconvblock(decoder.gtconv1, "gtconv1", decoder_dir)
    export_gtconvblock(decoder.gtconv2, "gtconv2", decoder_dir)
    export_gtconvblock(decoder.gtconv3, "gtconv3", decoder_dir)

    # Conv blocks
    export_convblock(decoder.conv1, "conv1", decoder_dir)
    export_convblock(decoder.conv2, "conv2", decoder_dir)

    print(f"Exported Decoder to {decoder_dir}")


def export_dpgrnn(dpgrnn, prefix, output_dir):
    """Export DPGRNN"""
    dpgrnn_dir = os.path.join(output_dir, prefix)
    os.makedirs(dpgrnn_dir, exist_ok=True)

    # Intra RNN (Bidirectional GRNN)
    # Note: For grouped GRU, need to split the weights
    # This is simplified - actual implementation depends on your GRNN structure

    # Intra GRU
    if hasattr(dpgrnn, 'intra_rnn'):
        export_gru(dpgrnn.intra_rnn, "intra_gru", dpgrnn_dir)

    # Intra Linear
    if hasattr(dpgrnn, 'intra_fc'):
        export_linear(dpgrnn.intra_fc, "intra_fc", dpgrnn_dir)

    # Intra LayerNorm
    if hasattr(dpgrnn, 'intra_ln'):
        export_layernorm(dpgrnn.intra_ln, "intra_ln", dpgrnn_dir)

    # Inter RNN (Unidirectional GRNN)
    if hasattr(dpgrnn, 'inter_rnn'):
        export_gru(dpgrnn.inter_rnn, "inter_gru", dpgrnn_dir)

    # Inter Linear
    if hasattr(dpgrnn, 'inter_fc'):
        export_linear(dpgrnn.inter_fc, "inter_fc", dpgrnn_dir)

    # Inter LayerNorm
    if hasattr(dpgrnn, 'inter_ln'):
        export_layernorm(dpgrnn.inter_ln, "inter_ln", dpgrnn_dir)

    print(f"Exported DPGRNN to {dpgrnn_dir}")


def export_gtcrn_model(model, output_dir):
    """Export complete GTCRN model"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("Exporting GTCRN Model Weights")
    print("="*70 + "\n")

    # Export Encoder
    export_encoder(model.encoder, output_dir)

    # Export DPGRNN layers
    export_dpgrnn(model.dpgrnn1, "dpgrnn1", output_dir)
    export_dpgrnn(model.dpgrnn2, "dpgrnn2", output_dir)

    # Export Decoder
    export_decoder(model.decoder, output_dir)

    print("\n" + "="*70)
    print(f"Successfully exported all weights to: {output_dir}")
    print("="*70 + "\n")

    # Create a README
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("GTCRN Model Weights\n")
        f.write("="*70 + "\n\n")
        f.write("This directory contains exported weights from PyTorch GTCRN model.\n")
        f.write("All weights are in binary format (float32, little-endian).\n\n")
        f.write("Directory structure:\n")
        f.write("  encoder/     - Encoder weights\n")
        f.write("  dpgrnn1/     - First DPGRNN layer weights\n")
        f.write("  dpgrnn2/     - Second DPGRNN layer weights\n")
        f.write("  decoder/     - Decoder weights\n\n")
        f.write("To load in C:\n")
        f.write("  GTCRN* model = gtcrn_create();\n")
        f.write(f"  load_gtcrn_weights(model, \"{output_dir}\");\n")


def main():
    parser = argparse.ArgumentParser(description='Export GTCRN weights to binary format')
    parser.add_argument('--model', type=str, required=True, help='Path to PyTorch model (.pth)')
    parser.add_argument('--output', type=str, default='weights/', help='Output directory')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model}")

    # You need to import your GTCRN model class here
    # from gtcrn1 import GTCRN
    # model = GTCRN()
    # model.load_state_dict(torch.load(args.model))
    # model.eval()

    # Export weights
    # export_gtcrn_model(model, args.output)

    print("\nNote: You need to uncomment and modify the model loading code")
    print("      to match your specific GTCRN implementation.")


if __name__ == '__main__':
    main()
