"""
Debug grouped convolution weight access pattern.
Compare C and Python weight indexing for EnConv1 grouped conv.
"""
import os
import sys
import numpy as np
import torch
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

from gtcrn_stream import StreamGTCRN
from modules.convert import convert_to_stream
from gtcrn import GTCRN

def main():
    # Load model
    model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])
    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # Get EnConv1 weights
    en_conv1 = stream_model.encoder.en_convs[1]
    weight = en_conv1.conv.weight.detach().numpy()  # (out_ch, in_ch/groups, kH, kW) = (16, 8, 1, 5)
    bias = en_conv1.conv.bias.detach().numpy()      # (16,)

    print(f"EnConv1 weight shape: {weight.shape}")
    print(f"EnConv1 bias shape: {bias.shape}")
    print(f"Groups: {en_conv1.conv.groups}")

    # Flatten weight as C would see it
    weight_flat = weight.flatten()
    print(f"\nWeight flat shape: {weight_flat.shape}")

    # C accesses: w_idx = ((oc_abs * in_ch_per_group + ic) * kernel_t + kt) * kernel_f + kf
    # For oc_abs=8, in_ch_per_group=8, kernel_t=1, kernel_f=5:
    # w_idx = (8*8 + ic) * 1 * 5 + kf = (64 + ic) * 5 + kf

    print("\n=== Weight access pattern for oc_abs=8 (first output of group 1) ===")
    oc_abs = 8
    in_ch_per_group = 8
    kernel_f = 5

    for ic in range(in_ch_per_group):
        base_idx = (oc_abs * in_ch_per_group + ic) * kernel_f
        print(f"\nic_local={ic}: C flat indices {base_idx} to {base_idx+4}")
        c_weights = weight_flat[base_idx:base_idx+5]
        print(f"  C would read: {c_weights}")

        # Python access pattern: weight[oc, ic, 0, :]
        # But wait - for grouped conv, PyTorch stores differently!
        # weight.shape = (out_ch, in_ch_per_group, kH, kW)
        # So for oc_abs=8 (in group 1), we need oc=8, and ic is local (0-7)
        py_weights = weight[oc_abs, ic, 0, :]
        print(f"  Python [oc={oc_abs}, ic={ic}, 0, :]: {py_weights}")

    # Now also check what values are at specific flat indices
    print("\n=== Flat weight values at key indices ===")
    for idx in range(320, 360):
        if idx % 5 == 0:
            print(f"\n  Index {idx}: {weight_flat[idx]:.6f}", end="")
        else:
            print(f"  {weight_flat[idx]:.6f}", end="")
    print()

    # The issue: C's formula assumes weight layout (oc, ic, kt, kf) flattened
    # But PyTorch grouped conv weight shape is (out_ch, in_ch_per_group, kH, kW)
    # When flattened, this is (out_ch * in_ch_per_group * kH * kW)
    # C index: oc_abs * (in_ch_per_group * 1 * 5) + ic * 5 + kf
    # Python: weight[oc_abs, ic, 0, kf]
    # These should match! Let's verify...

    print("\n=== Verifying flat index matches multidim access ===")
    for oc in range(16):
        for ic in range(8):
            flat_idx = oc * 8 * 1 * 5 + ic * 1 * 5
            py_val = weight[oc, ic, 0, 0]
            flat_val = weight_flat[flat_idx]
            if abs(py_val - flat_val) > 1e-6:
                print(f"MISMATCH: oc={oc}, ic={ic}: py={py_val:.6f}, flat={flat_val:.6f}")
    print("Index verification complete.")

if __name__ == "__main__":
    main()
