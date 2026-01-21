@echo off
REM Build script for GTCRN 16kHz real-time denoising example
REM Windows batch file

echo ========================================
echo Building GTCRN 16kHz Real-Time Denoiser
echo ========================================
echo.

REM Compiler settings
set CC=gcc
set CFLAGS=-O2 -Wall
set LIBS=-lm

REM Source files
set SOURCES=example_realtime_denoise_16k.c ^
    gtcrn_streaming_optimized_16k.c ^
    gtcrn_streaming_16k.c ^
    gtcrn_streaming_impl.c ^
    gtcrn_model.c ^
    gtcrn_modules.c ^
    stream_conv.c ^
    stft_16k.c ^
    weight_loader.c ^
    GRU.c ^
    conv2d.c ^
    batchnorm2d.c ^
    nn_layers.c ^
    layernorm.c

REM Output executable
set OUTPUT=denoise_16k.exe

echo Compiling...
%CC% %CFLAGS% -o %OUTPUT% %SOURCES% %LIBS%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Build successful!
    echo ========================================
    echo.
    echo Executable: %OUTPUT%
    echo.
    echo Usage:
    echo   %OUTPUT% input_16k.wav output_16k.wav weights/
    echo.
) else (
    echo.
    echo ========================================
    echo Build failed!
    echo ========================================
    echo.
)

pause
