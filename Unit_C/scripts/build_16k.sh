#!/bin/bash
# Build script for GTCRN 16kHz real-time denoising example
# Linux/Mac shell script

echo "========================================"
echo "Building GTCRN 16kHz Real-Time Denoiser"
echo "========================================"
echo ""

# Compiler settings
CC=gcc
CFLAGS="-O2 -Wall"
LIBS="-lm"

# Source files
SOURCES="example_realtime_denoise_16k.c \
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
    layernorm.c"

# Output executable
OUTPUT="denoise_16k"

echo "Compiling..."
$CC $CFLAGS -o $OUTPUT $SOURCES $LIBS

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Build successful!"
    echo "========================================"
    echo ""
    echo "Executable: $OUTPUT"
    echo ""
    echo "Usage:"
    echo "  ./$OUTPUT input_16k.wav output_16k.wav weights/"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Build failed!"
    echo "========================================"
    echo ""
    exit 1
fi
