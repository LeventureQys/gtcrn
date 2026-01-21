#!/bin/bash
# GTCRN C Library Build Script for Unix/Linux/macOS

echo "=== GTCRN C Library Build ==="
echo

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "[1/2] Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release
if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build
echo
echo "[2/2] Building..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo
echo "=== Build Complete ==="
echo
echo "Executable: build/gtcrn_demo"
echo "Library: build/libgtcrn.a"
echo
echo "Usage:"
echo "  ./gtcrn_demo weights/gtcrn_weights.bin input.wav output.wav"

cd ..
