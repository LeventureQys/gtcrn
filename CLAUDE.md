# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GTCRN (Grouped Temporal Convolutional Recurrent Network) is an ultra-lightweight speech enhancement model from ICASSP2024. It features only **48.2K parameters** and **33.0 MMACs/s**, designed for real-time noise suppression on resource-constrained devices.

Note: The paper reports 23.7K params and 39.6 MMACs/s, but the updated values include the ERB module parameters and use concatenation instead of matrix multiplication for low-frequency ERB mapping.

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# For ONNX export (additional dependencies)
pip install onnx onnxruntime onnxsim

# Run inference (offline mode)
python infer.py

# Check model complexity (FLOPs and params)
python gtcrn.py

# Run streaming inference with ONNX export
python stream/gtcrn_stream.py
```

## Architecture

### Core Model (`gtcrn.py`)
- **GTCRN**: Main model class combining ERB filterbanks, encoder-decoder with skip connections, and dual-path GRU
- **ERB**: Equivalent Rectangular Bandwidth filterbank for frequency compression (257 bins -> 129 ERB bands)
- **SFE**: Subband Feature Extraction using unfold operation
- **TRA**: Temporal Recurrent Attention using GRU for temporal weighting
- **GTConvBlock**: Group Temporal Convolution with ShuffleNet-style channel splitting
- **DPGRNN**: Dual-path Grouped RNN (intra-frame bidirectional + inter-frame unidirectional)
- **Mask**: Complex Ratio Mask for spectrogram enhancement

### Streaming Version (`stream/`)
- **StreamGTCRN** (`stream/gtcrn_stream.py`): Streaming-compatible model with explicit cache management
- **StreamConv2d/StreamConvTranspose2d** (`stream/modules/convolution.py`): Causal streaming convolutions
- **convert_to_stream** (`stream/modules/convert.py`): Converts offline model weights to streaming format

Cache tensor shapes for streaming inference (batch_size=1):
- `conv_cache`: `(2, 1, 16, 16, 33)` - encoder/decoder convolution states
- `tra_cache`: `(2, 3, 1, 1, 16)` - encoder/decoder TRA GRU states
- `inter_cache`: `(2, 1, 33, 16)` - DPGRNN inter-frame GRU states

Note: The explicit feature shuffle in grouped RNN was replaced with implicit rearrangement via FC layers to enable streaming.

### Signal Processing Pipeline
1. Input: 16kHz audio -> STFT (512-point FFT, 256 hop, sqrt-Hann window)
2. Features: magnitude, real, imaginary components stacked
3. ERB compression -> SFE -> Encoder -> DPGRNN x2 -> Decoder -> ERB expansion
4. Output: Complex ratio mask applied to input spectrogram -> ISTFT

## Pre-trained Models

Located in `checkpoints/`:
- `model_trained_on_dns3.tar`: Trained on DNS3 dataset
- `model_trained_on_vctk.tar`: Trained on VCTK-DEMAND dataset

## Loss Function (`loss.py`)

HybridLoss combines:
- Compressed MSE on real/imaginary components (power-law compression with 0.7 exponent)
- Compressed magnitude loss (0.3 exponent)
- SI-SNR in time domain

## ONNX Export

The streaming model can be exported to ONNX for CPU deployment. Cache tensors must be passed as inputs/outputs for frame-by-frame processing. See `stream/gtcrn_stream.py` for export code.

Achieves RTF of ~0.07 on Intel i5-12400 CPU @ 2.50 GHz.

## Related Projects

- [SEtrain](https://github.com/Xiaobin-Rong/SEtrain): Training code template for DNN-based speech enhancement
- [TRT-SE](https://github.com/Xiaobin-Rong/TRT-SE): ONNX/TensorRT deployment examples for speech enhancement
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): GTCRN integration for cross-platform deployment
