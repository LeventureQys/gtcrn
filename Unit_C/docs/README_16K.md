# GTCRN 16kHz Real-Time Audio Denoising

This directory contains the 16kHz version of the GTCRN real-time audio denoising implementation in C.

## Overview

The 16kHz version is optimized for lower sample rate audio processing, with the following key differences from the 48kHz version:

### Parameter Changes (48kHz → 16kHz)

| Parameter | 48kHz | 16kHz | Ratio |
|-----------|-------|-------|-------|
| Sample Rate | 48000 Hz | 16000 Hz | ÷3 |
| FFT Size | 1536 | 512 | ÷3 |
| Hop Length | 768 | 256 | ÷3 |
| Frequency Bins | 769 | 257 | ÷3 |
| Chunk Size | 768 samples | 256 samples | ÷3 |
| Frame Duration | ~32ms | ~32ms | Same |
| Latency | ~32ms | ~32ms | Same |

## Files

### Core 16kHz Implementation
- `stft_16k.h` / `stft_16k.c` - STFT/iSTFT for 16kHz audio
- `gtcrn_streaming_16k.h` / `gtcrn_streaming_16k.c` - Streaming processor for 16kHz
- `gtcrn_streaming_optimized_16k.c` - Optimized streaming implementation for 16kHz
- `example_realtime_denoise_16k.c` - Complete example program for 16kHz

### Shared Files (Used by both 48kHz and 16kHz)
- `gtcrn_model.h` / `gtcrn_model.c` - GTCRN model architecture
- `gtcrn_modules.h` / `gtcrn_modules.c` - ERB, SFE, TRA modules
- `gtcrn_streaming_impl.c` - Streaming implementation helpers
- `stream_conv.h` / `stream_conv.c` - Streaming convolution
- `GRU.h` / `GRU.c` - GRU implementation
- `conv2d.h` / `conv2d.c` - 2D convolution
- `batchnorm2d.h` / `batchnorm2d.c` - Batch normalization
- `nn_layers.h` / `nn_layers.c` - Neural network layers
- `layernorm.h` / `layernorm.c` - Layer normalization
- `weight_loader.h` / `weight_loader.c` - Weight loading utilities

### Build Scripts
- `build_16k.bat` - Windows build script
- `build_16k.sh` - Linux/Mac build script

## Building

### Windows
```batch
build_16k.bat
```

### Linux/Mac
```bash
chmod +x build_16k.sh
./build_16k.sh
```

### Manual Compilation
```bash
gcc -o denoise_16k example_realtime_denoise_16k.c \
    gtcrn_streaming_optimized_16k.c \
    gtcrn_streaming_16k.c \
    gtcrn_streaming_impl.c \
    gtcrn_model.c \
    gtcrn_modules.c \
    stream_conv.c \
    stft_16k.c \
    weight_loader.c \
    GRU.c \
    conv2d.c \
    batchnorm2d.c \
    nn_layers.c \
    layernorm.c \
    -lm -O2
```

## Usage

### Basic Usage
```bash
# Windows
denoise_16k.exe input_16k.wav output_16k.wav weights/

# Linux/Mac
./denoise_16k input_16k.wav output_16k.wav weights/
```

### Input Requirements
- **Sample Rate**: 16000 Hz (16kHz)
- **Format**: WAV file, 16-bit PCM
- **Channels**: Mono (single channel)

### Example
```bash
# Process a 16kHz noisy audio file
./denoise_16k noisy_speech_16k.wav clean_speech_16k.wav weights/

# The program will output:
# - Processing progress
# - Real-time factor (RTF)
# - Latency information
# - Enhanced audio file
```

## Performance

### Expected Performance
- **Real-Time Factor**: < 1.0 (faster than real-time on modern CPUs)
- **Latency**: ~32ms (algorithmic latency from STFT window)
- **Memory Usage**: Lower than 48kHz version due to smaller FFT size

### Optimization Features
- Frame-by-frame processing with state caching
- Optimized STFT/iSTFT implementation
- Efficient memory management
- Persistent state buffers to avoid allocations

## Architecture

### Processing Pipeline
```
Input Audio (16kHz)
    ↓
STFT (n_fft=512, hop=256)
    ↓
Frequency Domain (257 bins)
    ↓
ERB Compression (257 → 385 bins)
    ↓
SFE (Subband Feature Extraction)
    ↓
Encoder (5 layers)
    ↓
DPGRNN (2 layers)
    ↓
Decoder (5 layers)
    ↓
ERB Decompression (385 → 257 bins)
    ↓
Complex Mask Application
    ↓
iSTFT (n_fft=512, hop=256)
    ↓
Output Audio (16kHz)
```

### State Caching
The streaming implementation maintains state caches for:
- **Encoder**: 5 convolution/TRA caches
- **DPGRNN**: 2 inter-RNN state caches
- **Decoder**: 5 convolution/TRA caches
- **Skip Connections**: 5 persistent buffers
- **STFT**: Input buffer and overlap-add buffer

## Differences from 48kHz Version

### 1. STFT Parameters
- Smaller FFT size (512 vs 1536)
- Smaller hop length (256 vs 768)
- Fewer frequency bins (257 vs 769)

### 2. Processing Chunks
- Smaller chunk size (256 vs 768 samples)
- Same frame duration (~16ms per chunk)

### 3. Memory Footprint
- Reduced memory usage due to smaller FFT
- Fewer frequency bins to process

### 4. Computational Cost
- Lower computational cost per frame
- Faster processing due to smaller FFT

## Weights

The model weights are shared between 48kHz and 16kHz versions. To export weights from PyTorch:

```python
# See export_weights.py in the parent directory
python export_weights.py --model_path checkpoint.pth --output_dir weights/
```

## Troubleshooting

### Build Errors
- Ensure all source files are in the same directory
- Check that gcc is installed and in PATH
- Verify math library (-lm) is available

### Runtime Errors
- **"Audio too short for STFT"**: Input file must be longer than FFT size (512 samples = 32ms)
- **"Failed to load weights"**: Check weights directory path and files
- **"Sample rate mismatch"**: Input must be 16kHz (use ffmpeg to resample if needed)

### Resampling Audio to 16kHz
```bash
# Using ffmpeg
ffmpeg -i input_48k.wav -ar 16000 input_16k.wav
```

## Performance Tips

1. **Use -O2 or -O3 optimization** when compiling
2. **Process in batches** for better throughput
3. **Adjust chunk size** based on latency requirements
4. **Use optimized BLAS** libraries if available

## Comparison with 48kHz Version

| Aspect | 48kHz | 16kHz |
|--------|-------|-------|
| Audio Quality | Higher | Lower |
| Processing Speed | Slower | Faster |
| Memory Usage | Higher | Lower |
| Latency | ~32ms | ~32ms |
| Use Case | High-quality audio | Voice/telephony |

## License

See the main project LICENSE file.

## References

- GTCRN Paper: "GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources"
- Original PyTorch Implementation: [Link to repository]

## Contact

For questions or issues, please open an issue on the project repository.
