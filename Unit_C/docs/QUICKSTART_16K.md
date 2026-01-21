# Quick Start Guide - GTCRN 16kHz Version

## 🚀 Quick Start (3 Steps)

### 1. Build
```bash
# Windows
build_16k.bat

# Linux/Mac
chmod +x build_16k.sh
./build_16k.sh
```

### 2. Prepare Audio (if needed)
```bash
# Convert 48kHz to 16kHz using ffmpeg
ffmpeg -i input_48k.wav -ar 16000 input_16k.wav
```

### 3. Run
```bash
# Windows
denoise_16k.exe input_16k.wav output_16k.wav weights/

# Linux/Mac
./denoise_16k input_16k.wav output_16k.wav weights/
```

## 📊 Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 16000 Hz | Input audio must be 16kHz |
| FFT Size | 512 | STFT window size |
| Hop Length | 256 | STFT hop size |
| Chunk Size | 256 samples | Processing chunk (16ms) |
| Latency | ~32ms | Total processing latency |
| Freq Bins | 257 | Number of frequency bins |

## 📁 Files Overview

### Core Files (16kHz specific)
```
stft_16k.h/c                        - STFT for 16kHz
gtcrn_streaming_16k.h/c             - Streaming processor
gtcrn_streaming_optimized_16k.c     - Optimized version
example_realtime_denoise_16k.c      - Example program
```

### Shared Files (used by both 48kHz and 16kHz)
```
gtcrn_model.h/c                     - Model architecture
gtcrn_modules.h/c                   - ERB, SFE, TRA
gtcrn_streaming_impl.c              - Streaming helpers
GRU.h/c, conv2d.h/c, etc.          - Neural network layers
```

## 🔄 48kHz vs 16kHz Comparison

| Aspect | 48kHz | 16kHz |
|--------|-------|-------|
| FFT Size | 1536 | 512 |
| Hop Length | 768 | 256 |
| Freq Bins | 769 | 257 |
| Speed | 1x | ~3x faster |
| Memory | 1x | ~0.33x (67% less) |
| Latency | ~32ms | ~32ms (same) |
| Quality | Higher | Lower (but sufficient for voice) |

## ⚡ Performance Tips

1. **Compile with optimization**: Use `-O2` or `-O3`
2. **Use correct sample rate**: Input must be exactly 16kHz
3. **Batch processing**: Process multiple files for better throughput
4. **Check RTF**: Real-time factor should be < 1.0

## 🐛 Common Issues

### Issue: "Audio too short for STFT"
**Solution**: Input must be > 512 samples (32ms)

### Issue: "Sample rate mismatch"
**Solution**: Resample to 16kHz using ffmpeg:
```bash
ffmpeg -i input.wav -ar 16000 output_16k.wav
```

### Issue: Build fails
**Solution**: Ensure all source files are present:
```bash
ls -1 *.c *.h | grep -E "(16k|gtcrn|GRU|conv|batch|layer|weight|stream)"
```

### Issue: Poor quality output
**Solution**: Export and load proper weights from PyTorch model

## 📈 Expected Output

```
=================================================================
GTCRN Real-Time Audio Denoising - 16kHz Version
=================================================================

Step 1: Loading audio...
Reading WAV: input_16k.wav
  Sample rate: 16000 Hz
  Samples: 160000
  Duration: 10.00 seconds

Step 2: Creating GTCRN model...

Step 3: Loading model weights...

Step 4: Creating streaming processor for 16kHz...
GTCRN Streaming 16kHz created:
  Sample rate: 16000 Hz
  Chunk size: 256 samples
  FFT size: 512
  Hop length: 256
  Latency: ~32.0 ms

Step 5: Processing audio...
Processing 625 chunks...
  Progress: 100.0% (625/625 chunks)

Processing complete!
  Audio duration: 10.00 seconds
  Processing time: 1.23 seconds
  Real-time factor: 0.123 (8.1x faster than real-time)
  Frames processed: 625
  Average latency: 1.97 ms
  Total latency: 33.97 ms

Step 6: Saving enhanced audio...
Wrote WAV: output_16k.wav
  Samples: 160000
  Duration: 10.00 seconds

=================================================================
Done!
=================================================================
```

## 🎯 Use Cases

### ✅ Good for:
- Voice communication (VoIP, telephony)
- Speech recognition preprocessing
- Podcast/voice recording cleanup
- Low-power embedded devices
- Real-time mobile applications

### ❌ Not ideal for:
- Music processing (use 48kHz)
- High-fidelity audio (use 48kHz)
- Frequencies above 8kHz (Nyquist limit)

## 🔧 Advanced Usage

### Custom Chunk Size
Modify in `example_realtime_denoise_16k.c`:
```c
int chunk_size = 256;  // Default: 16ms
// Try: 128 (8ms), 512 (32ms), etc.
```

### Real-time Microphone Input
See `gtcrn_streaming_16k.h` for API:
```c
GTCRNStreaming_16k* stream = gtcrn_streaming_16k_create(model, 16000, 256);
gtcrn_streaming_16k_process_chunk_optimized(stream, input, output);
```

### Batch Processing
```bash
for file in *.wav; do
    ./denoise_16k "$file" "clean_$file" weights/
done
```

## 📚 Documentation

- **Full Documentation**: `README_16K.md`
- **Conversion Details**: `CONVERSION_SUMMARY_16K.md`
- **Original 48kHz**: `example_realtime_denoise.c`

## 🆘 Getting Help

1. Check `README_16K.md` for detailed documentation
2. Review `CONVERSION_SUMMARY_16K.md` for technical details
3. Compare with 48kHz version for reference
4. Check build output for specific errors

## ✅ Verification Checklist

- [ ] All source files present
- [ ] Builds without errors
- [ ] Runs without crashes
- [ ] Input is 16kHz WAV file
- [ ] Output file created successfully
- [ ] Real-time factor < 1.0
- [ ] Latency ~32ms
- [ ] Audio quality acceptable

## 🎉 Success Indicators

- ✅ Build completes without errors
- ✅ RTF < 1.0 (faster than real-time)
- ✅ Latency ~32-35ms
- ✅ Output file plays correctly
- ✅ Noise reduced in output

---

**Need more help?** See `README_16K.md` for comprehensive documentation.
