#!/usr/bin/env python3
"""
GTCRN Stream Weight Export Script

Exports PyTorch streaming model weights (after convert_to_stream) to binary format for C implementation.
This ensures C implementation uses the same weights as Python streaming model.

Usage:
    python export_stream_weights.py --model checkpoints/model_trained_on_dns3.tar --output weights/gtcrn_stream_weights.bin
"""

import os
import sys
import struct
import argparse
import numpy as np
import torch

# Add gtcrn directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
gtcrn_c_dir = os.path.dirname(script_dir)
gtcrn_dir = os.path.dirname(gtcrn_c_dir)
sys.path.insert(0, gtcrn_dir)
from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream


def export_tensor(f, tensor, name=""):
    """Export a tensor to binary file in row-major order."""
    if tensor is None:
        return 0

    data = tensor.detach().cpu().numpy().astype(np.float32)
    flat = data.flatten()
    f.write(struct.pack(f'{len(flat)}f', *flat))

    print(f"  {name}: shape={list(tensor.shape)}, size={len(flat)}")
    return len(flat)


def export_conv2d(f, conv, bn, prefix):
    """Export Conv2d + BatchNorm2d weights."""
    total = 0

    # Conv weights: (out_ch, in_ch/groups, kH, kW)
    total += export_tensor(f, conv.weight, f"{prefix}.conv.weight")
    total += export_tensor(f, conv.bias, f"{prefix}.conv.bias")

    # BatchNorm
    total += export_tensor(f, bn.weight, f"{prefix}.bn.weight")  # gamma
    total += export_tensor(f, bn.bias, f"{prefix}.bn.bias")      # beta
    total += export_tensor(f, bn.running_mean, f"{prefix}.bn.mean")
    total += export_tensor(f, bn.running_var, f"{prefix}.bn.var")

    return total


def export_stream_conv2d(f, conv, bn, prefix):
    """Export StreamConv2d + BatchNorm2d weights."""
    total = 0

    # StreamConv2d wraps Conv2d
    total += export_tensor(f, conv.Conv2d.weight, f"{prefix}.conv.weight")
    total += export_tensor(f, conv.Conv2d.bias, f"{prefix}.conv.bias")

    # BatchNorm
    total += export_tensor(f, bn.weight, f"{prefix}.bn.weight")
    total += export_tensor(f, bn.bias, f"{prefix}.bn.bias")
    total += export_tensor(f, bn.running_mean, f"{prefix}.bn.mean")
    total += export_tensor(f, bn.running_var, f"{prefix}.bn.var")

    return total


def export_stream_conv_transpose2d(f, deconv, bn, prefix):
    """Export StreamConvTranspose2d + BatchNorm2d weights.

    Note: StreamConvTranspose2d uses Conv2d internally with flipped weights.
    We export the Conv2d weights (already flipped by convert_to_stream).
    """
    total = 0

    # StreamConvTranspose2d wraps Conv2d (with flipped weights)
    total += export_tensor(f, deconv.ConvTranspose2d.weight, f"{prefix}.conv.weight")
    total += export_tensor(f, deconv.ConvTranspose2d.bias, f"{prefix}.conv.bias")

    # BatchNorm
    total += export_tensor(f, bn.weight, f"{prefix}.bn.weight")
    total += export_tensor(f, bn.bias, f"{prefix}.bn.bias")
    total += export_tensor(f, bn.running_mean, f"{prefix}.bn.mean")
    total += export_tensor(f, bn.running_var, f"{prefix}.bn.var")

    return total


def export_regular_conv_transpose2d(f, deconv, bn, prefix):
    """Export regular ConvTranspose2d + BatchNorm2d weights.

    Regular ConvTranspose2d weight layout: (in_ch, out_ch/groups, kH, kW)
    For C implementation using upsampling+Conv2d approach, we need to:
    1. Handle grouped convolution properly
    2. Transpose within each group
    3. Flip the kernel in both H and W dimensions
    """
    import torch
    total = 0

    # Get original weight: (in_ch, out_ch/groups, kH, kW)
    weight = deconv.weight
    in_ch = deconv.in_channels
    out_ch = deconv.out_channels
    groups = deconv.groups
    kH, kW = weight.shape[2], weight.shape[3]

    in_ch_per_group = in_ch // groups
    out_ch_per_group = out_ch // groups

    if groups > 1:
        # For grouped ConvTranspose2d, we need to handle weight layout carefully
        # ConvTranspose2d: weight[g*in_ch_per_group:(g+1)*in_ch_per_group, :, :, :]
        # Conv2d: weight[g*out_ch_per_group:(g+1)*out_ch_per_group, :, :, :]

        # Reshape to (groups, in_ch_per_group, out_ch_per_group, kH, kW)
        weight_reshaped = weight.view(groups, in_ch_per_group, out_ch_per_group, kH, kW)

        # Transpose within each group: swap in and out channel dimensions
        weight_transposed = weight_reshaped.permute(0, 2, 1, 3, 4)

        # Reshape back to (out_ch, in_ch_per_group, kH, kW)
        weight_converted = weight_transposed.reshape(out_ch, in_ch_per_group, kH, kW)
    else:
        # For non-grouped, simple permute works
        weight_converted = weight.permute(1, 0, 2, 3)

    # Flip the kernel in both H and W dimensions (for convolution equivalence)
    weight_flipped = torch.flip(weight_converted, dims=[-2, -1])

    total += export_tensor(f, weight_flipped, f"{prefix}.conv.weight")
    total += export_tensor(f, deconv.bias, f"{prefix}.conv.bias")

    # BatchNorm
    total += export_tensor(f, bn.weight, f"{prefix}.bn.weight")
    total += export_tensor(f, bn.bias, f"{prefix}.bn.bias")
    total += export_tensor(f, bn.running_mean, f"{prefix}.bn.mean")
    total += export_tensor(f, bn.running_var, f"{prefix}.bn.var")

    return total


def export_prelu(f, prelu, prefix):
    """Export PReLU weights."""
    return export_tensor(f, prelu.weight, f"{prefix}.prelu")


def export_gru(f, gru, prefix, bidirectional=False):
    """Export GRU weights."""
    total = 0

    # Forward direction
    total += export_tensor(f, gru.weight_ih_l0, f"{prefix}.weight_ih")
    total += export_tensor(f, gru.weight_hh_l0, f"{prefix}.weight_hh")
    total += export_tensor(f, gru.bias_ih_l0, f"{prefix}.bias_ih")
    total += export_tensor(f, gru.bias_hh_l0, f"{prefix}.bias_hh")

    # Reverse direction (if bidirectional)
    if bidirectional:
        total += export_tensor(f, gru.weight_ih_l0_reverse, f"{prefix}.weight_ih_rev")
        total += export_tensor(f, gru.weight_hh_l0_reverse, f"{prefix}.weight_hh_rev")
        total += export_tensor(f, gru.bias_ih_l0_reverse, f"{prefix}.bias_ih_rev")
        total += export_tensor(f, gru.bias_hh_l0_reverse, f"{prefix}.bias_hh_rev")

    return total


def export_linear(f, linear, prefix):
    """Export Linear layer weights."""
    total = 0
    total += export_tensor(f, linear.weight, f"{prefix}.weight")
    if linear.bias is not None:
        total += export_tensor(f, linear.bias, f"{prefix}.bias")
    return total


def export_layernorm(f, ln, prefix):
    """Export LayerNorm weights."""
    total = 0
    total += export_tensor(f, ln.weight, f"{prefix}.weight")
    total += export_tensor(f, ln.bias, f"{prefix}.bias")
    return total


def export_stream_gtconvblock(f, block, prefix, is_encoder=True):
    """Export StreamGTConvBlock weights."""
    import torch.nn as nn
    total = 0

    # Point conv 1 - check type
    if hasattr(block.point_conv1, 'Conv2d'):
        # It's a StreamConv2d
        total += export_stream_conv2d(f, block.point_conv1, block.point_bn1, f"{prefix}.pc1")
    elif hasattr(block.point_conv1, 'ConvTranspose2d'):
        # It's a StreamConvTranspose2d
        total += export_stream_conv_transpose2d(f, block.point_conv1, block.point_bn1, f"{prefix}.pc1")
    elif isinstance(block.point_conv1, nn.ConvTranspose2d):
        # It's a regular ConvTranspose2d - convert weights for C implementation
        total += export_regular_conv_transpose2d(f, block.point_conv1, block.point_bn1, f"{prefix}.pc1")
    else:
        # It's a regular Conv2d
        total += export_conv2d(f, block.point_conv1, block.point_bn1, f"{prefix}.pc1")
    total += export_prelu(f, block.point_act, f"{prefix}.prelu1")

    # Depth conv - check type
    if hasattr(block.depth_conv, 'Conv2d'):
        # It's a StreamConv2d
        total += export_stream_conv2d(f, block.depth_conv, block.depth_bn, f"{prefix}.dc")
    elif hasattr(block.depth_conv, 'ConvTranspose2d'):
        # It's a StreamConvTranspose2d
        total += export_stream_conv_transpose2d(f, block.depth_conv, block.depth_bn, f"{prefix}.dc")
    elif isinstance(block.depth_conv, nn.ConvTranspose2d):
        # It's a regular ConvTranspose2d - convert weights for C implementation
        total += export_regular_conv_transpose2d(f, block.depth_conv, block.depth_bn, f"{prefix}.dc")
    else:
        # It's a regular Conv2d
        total += export_conv2d(f, block.depth_conv, block.depth_bn, f"{prefix}.dc")
    total += export_prelu(f, block.depth_act, f"{prefix}.prelu2")

    # Point conv 2 - check type
    if hasattr(block.point_conv2, 'Conv2d'):
        # It's a StreamConv2d
        total += export_stream_conv2d(f, block.point_conv2, block.point_bn2, f"{prefix}.pc2")
    elif hasattr(block.point_conv2, 'ConvTranspose2d'):
        # It's a StreamConvTranspose2d
        total += export_stream_conv_transpose2d(f, block.point_conv2, block.point_bn2, f"{prefix}.pc2")
    elif isinstance(block.point_conv2, nn.ConvTranspose2d):
        # It's a regular ConvTranspose2d - convert weights for C implementation
        total += export_regular_conv_transpose2d(f, block.point_conv2, block.point_bn2, f"{prefix}.pc2")
    else:
        # It's a regular Conv2d
        total += export_conv2d(f, block.point_conv2, block.point_bn2, f"{prefix}.pc2")

    # TRA
    total += export_gru(f, block.tra.att_gru, f"{prefix}.tra.gru")
    total += export_linear(f, block.tra.att_fc, f"{prefix}.tra.fc")

    return total


def export_dpgrnn(f, dpgrnn, prefix):
    """Export DPGRNN weights."""
    total = 0

    # Intra RNN (bidirectional grouped)
    total += export_gru(f, dpgrnn.intra_rnn.rnn1, f"{prefix}.intra.rnn1", bidirectional=True)
    total += export_gru(f, dpgrnn.intra_rnn.rnn2, f"{prefix}.intra.rnn2", bidirectional=True)
    total += export_linear(f, dpgrnn.intra_fc, f"{prefix}.intra.fc")
    total += export_layernorm(f, dpgrnn.intra_ln, f"{prefix}.intra.ln")

    # Inter RNN (unidirectional grouped)
    total += export_gru(f, dpgrnn.inter_rnn.rnn1, f"{prefix}.inter.rnn1", bidirectional=False)
    total += export_gru(f, dpgrnn.inter_rnn.rnn2, f"{prefix}.inter.rnn2", bidirectional=False)
    total += export_linear(f, dpgrnn.inter_fc, f"{prefix}.inter.fc")
    total += export_layernorm(f, dpgrnn.inter_ln, f"{prefix}.inter.ln")

    return total


def export_stream_model(stream_model, output_path):
    """Export complete StreamGTCRN model to binary file."""
    import torch.nn as nn

    print(f"Exporting streaming model to {output_path}")
    total_params = 0

    with open(output_path, 'wb') as f:
        # Write header
        f.write(b'GTCS')  # Magic: GTCRN Stream
        f.write(struct.pack('I', 1))  # Version

        print("\n=== ERB Filterbank ===")
        total_params += export_tensor(f, stream_model.erb.erb_fc.weight, "erb_fc.weight")
        total_params += export_tensor(f, stream_model.erb.ierb_fc.weight, "ierb_fc.weight")

        print("\n=== Encoder ===")

        # ConvBlock 0
        print("ConvBlock 0:")
        conv0 = stream_model.encoder.en_convs[0]
        total_params += export_conv2d(f, conv0.conv, conv0.bn, "en_conv0")
        total_params += export_prelu(f, conv0.act, "en_prelu0")

        # ConvBlock 1
        print("ConvBlock 1:")
        conv1 = stream_model.encoder.en_convs[1]
        total_params += export_conv2d(f, conv1.conv, conv1.bn, "en_conv1")
        total_params += export_prelu(f, conv1.act, "en_prelu1")

        # GTConvBlocks 2-4
        print("GTConvBlock 2 (dilation=1):")
        total_params += export_stream_gtconvblock(f, stream_model.encoder.en_convs[2], "en_gt2", is_encoder=True)
        print("GTConvBlock 3 (dilation=2):")
        total_params += export_stream_gtconvblock(f, stream_model.encoder.en_convs[3], "en_gt3", is_encoder=True)
        print("GTConvBlock 4 (dilation=5):")
        total_params += export_stream_gtconvblock(f, stream_model.encoder.en_convs[4], "en_gt4", is_encoder=True)

        print("\n=== DPGRNN ===")
        print("DPGRNN 1:")
        total_params += export_dpgrnn(f, stream_model.dpgrnn1, "dp1")
        print("DPGRNN 2:")
        total_params += export_dpgrnn(f, stream_model.dpgrnn2, "dp2")

        print("\n=== Decoder ===")

        # GTConvBlocks 0-2
        print("GTConvBlock 0 (dilation=5):")
        total_params += export_stream_gtconvblock(f, stream_model.decoder.de_convs[0], "de_gt0", is_encoder=False)
        print("GTConvBlock 1 (dilation=2):")
        total_params += export_stream_gtconvblock(f, stream_model.decoder.de_convs[1], "de_gt1", is_encoder=False)
        print("GTConvBlock 2 (dilation=1):")
        total_params += export_stream_gtconvblock(f, stream_model.decoder.de_convs[2], "de_gt2", is_encoder=False)

        # ConvBlock 3
        print("ConvBlock 3:")
        conv3 = stream_model.decoder.de_convs[3]
        # Check if it's StreamConvTranspose2d or regular ConvTranspose2d
        if hasattr(conv3.conv, 'ConvTranspose2d'):
            total_params += export_stream_conv_transpose2d(f, conv3.conv, conv3.bn, "de_conv3")
        elif isinstance(conv3.conv, nn.ConvTranspose2d):
            # It's a regular ConvTranspose2d - convert weights for C implementation
            total_params += export_regular_conv_transpose2d(f, conv3.conv, conv3.bn, "de_conv3")
        else:
            total_params += export_conv2d(f, conv3.conv, conv3.bn, "de_conv3")
        total_params += export_prelu(f, conv3.act, "de_prelu3")

        # ConvBlock 4 (output, with Tanh)
        print("ConvBlock 4 (output):")
        conv4 = stream_model.decoder.de_convs[4]
        # Check if it's StreamConvTranspose2d or regular ConvTranspose2d
        if hasattr(conv4.conv, 'ConvTranspose2d'):
            total_params += export_stream_conv_transpose2d(f, conv4.conv, conv4.bn, "de_conv4")
        elif isinstance(conv4.conv, nn.ConvTranspose2d):
            # It's a regular ConvTranspose2d - convert weights for C implementation
            total_params += export_regular_conv_transpose2d(f, conv4.conv, conv4.bn, "de_conv4")
        else:
            total_params += export_conv2d(f, conv4.conv, conv4.bn, "de_conv4")
        # Note: Tanh activation, no learnable parameters

    print(f"\n=== Summary ===")
    print(f"Total parameters exported: {total_params}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    print(f"Expected size: {total_params * 4 + 8} bytes (params + header)")


def main():
    parser = argparse.ArgumentParser(description="Export GTCRN streaming weights to binary format")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to PyTorch model checkpoint")
    parser.add_argument("--output", type=str, default="weights/gtcrn_stream_weights.bin",
                        help="Output binary file path")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load complete model
    print(f"Loading complete model from {args.model}")
    model = GTCRN().eval()

    checkpoint = torch.load(args.model, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    # Create streaming model and convert weights
    print("Converting to streaming model...")
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, model)

    # Count parameters
    total_params = sum(p.numel() for p in stream_model.parameters())
    print(f"Streaming model has {total_params} parameters ({total_params * 4 / 1024:.1f} KB)")

    # Export
    export_stream_model(stream_model, args.output)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
