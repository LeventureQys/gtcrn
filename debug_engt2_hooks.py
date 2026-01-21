"""
Debug EnGT2 intermediate values using forward hooks.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf

gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, gtcrn_dir)

from gtcrn import GTCRN
from gtcrn_stream import StreamGTCRN, StreamGTConvBlock
from modules.convert import convert_to_stream

def abs_sum(x):
    """Return sum of absolute values"""
    return float(x.abs().sum())

# Global storage for intermediate values
intermediate_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            intermediate_outputs[name] = (abs_sum(output[0]), output[0].detach().clone())
        else:
            intermediate_outputs[name] = (abs_sum(output), output.detach().clone())
    return hook

def main():
    print("=== Debug EnGT2 Detail with Hooks ===\n")

    # Load model
    model_path = os.path.join(gtcrn_dir, 'checkpoints/model_trained_on_dns3.tar')
    offline_model = GTCRN().eval()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    offline_model.load_state_dict(checkpoint['model'])

    stream_model = StreamGTCRN().eval()
    convert_to_stream(stream_model, offline_model)

    # Register hooks on EnGT2 components
    en_gt2 = stream_model.encoder.en_convs[2]
    hooks = []

    # Hook SFE
    hooks.append(en_gt2.sfe.register_forward_hook(make_hook('engt2_sfe')))

    # Hook PointConv1
    hooks.append(en_gt2.point_conv1.register_forward_hook(make_hook('engt2_pc1')))
    hooks.append(en_gt2.point_bn1.register_forward_hook(make_hook('engt2_bn1')))
    hooks.append(en_gt2.point_act.register_forward_hook(make_hook('engt2_prelu1')))

    # Hook DepthConv
    hooks.append(en_gt2.depth_conv.register_forward_hook(make_hook('engt2_dc')))
    hooks.append(en_gt2.depth_bn.register_forward_hook(make_hook('engt2_depth_bn')))
    hooks.append(en_gt2.depth_act.register_forward_hook(make_hook('engt2_depth_prelu')))

    # Hook PointConv2
    hooks.append(en_gt2.point_conv2.register_forward_hook(make_hook('engt2_pc2')))
    hooks.append(en_gt2.point_bn2.register_forward_hook(make_hook('engt2_bn2')))

    # Hook TRA
    hooks.append(en_gt2.tra.register_forward_hook(make_hook('engt2_tra')))

    # Hook full EnGT2
    hooks.append(en_gt2.register_forward_hook(make_hook('engt2_output')))

    # Also hook EnConv0 and EnConv1
    hooks.append(stream_model.encoder.en_convs[0].register_forward_hook(make_hook('enconv0')))
    hooks.append(stream_model.encoder.en_convs[1].register_forward_hook(make_hook('enconv1')))

    # Load audio
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)
    print(f"Audio: {len(audio)} samples, {sr} Hz\n")

    # STFT params
    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Pad audio with 256 zeros (matches Python streaming)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    # Compute STFT
    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False,
                      center=False)
    spec = spec.unsqueeze(0)  # (1, 257, T, 2)

    # Initialize caches
    conv_cache = torch.zeros(2, 1, 16, 16, 33)
    tra_cache = torch.zeros(2, 3, 1, 1, 16)
    inter_cache = torch.zeros(2, 1, 33, 16)

    # Process frames 0-5 to get to frame 5 (which matches C frame_count=6)
    target_frame = 5  # Python frame index (C frame_count = 6)

    with torch.no_grad():
        for i in range(target_frame):
            frame = spec[:, :, i:i+1, :]
            out_frame, conv_cache, tra_cache, inter_cache = stream_model(
                frame, conv_cache, tra_cache, inter_cache
            )
            if i == target_frame - 1:
                break  # Don't clear yet

    # Clear intermediates and process target frame
    intermediate_outputs.clear()

    print(f"=== Tracing Python Frame {target_frame} (C frame_count={target_frame+1}) ===\n")

    with torch.no_grad():
        frame = spec[:, :, target_frame:target_frame+1, :]
        out_frame, conv_cache, tra_cache, inter_cache = stream_model(
            frame, conv_cache, tra_cache, inter_cache
        )

    # Print intermediate values
    print("EnConv0 output abs_sum:", intermediate_outputs.get('enconv0', (0,))[0])
    print("EnConv1 output abs_sum:", intermediate_outputs.get('enconv1', (0,))[0])

    print("\n--- EnGT2 Detail ---")
    print("  SFE output abs_sum:", intermediate_outputs.get('engt2_sfe', (0,))[0])
    print("  PointConv1 output abs_sum:", intermediate_outputs.get('engt2_pc1', (0,))[0])
    print("  PointBN1 output abs_sum:", intermediate_outputs.get('engt2_bn1', (0,))[0])
    print("  PReLU1 output abs_sum:", intermediate_outputs.get('engt2_prelu1', (0,))[0])
    print("  DepthConv output abs_sum:", intermediate_outputs.get('engt2_dc', (0,))[0])
    print("  DepthBN output abs_sum:", intermediate_outputs.get('engt2_depth_bn', (0,))[0])
    print("  DepthPReLU output abs_sum:", intermediate_outputs.get('engt2_depth_prelu', (0,))[0])
    print("  PointConv2 output abs_sum:", intermediate_outputs.get('engt2_pc2', (0,))[0])
    print("  PointBN2 output abs_sum:", intermediate_outputs.get('engt2_bn2', (0,))[0])
    print("  TRA output abs_sum:", intermediate_outputs.get('engt2_tra', (0,))[0])
    print("  EnGT2 output abs_sum:", intermediate_outputs.get('engt2_output', (0,))[0])

    # Print first 10 values of key tensors
    for name in ['engt2_sfe', 'engt2_prelu1', 'engt2_dc', 'engt2_depth_prelu', 'engt2_bn2', 'engt2_tra']:
        if name in intermediate_outputs:
            _, tensor = intermediate_outputs[name]
            print(f"\n  {name} first 10 vals: {tensor.flatten()[:10].tolist()}")

    # Print per-channel sums for EnGT2 output
    if 'engt2_output' in intermediate_outputs:
        _, output = intermediate_outputs['engt2_output']
        print("\n  EnGT2 output per-channel sums:")
        for c in range(output.shape[1]):
            ch_sum = output[0, c].sum().item()
            print(f"    ch{c}: {ch_sum:.6f}")

    # Clean up hooks
    for h in hooks:
        h.remove()

if __name__ == "__main__":
    main()
