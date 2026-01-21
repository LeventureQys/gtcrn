# GTCRN C Implementation

GTCRN (Grouped Temporal Convolutional Recurrent Network) 语音增强模型的纯 C 实现，支持实时流式降噪处理。

## 项目概述

本项目是 GTCRN 模型从 PyTorch 到纯 C 语言的完整移植，适用于嵌入式设备和资源受限的平台。

### 特性

- **超轻量级**: 仅 48.2K 参数
- **低计算量**: 33.0 MMACs/s
- **实时流式处理**: 支持逐帧处理，延迟约 32ms @ 48kHz
- **跨平台**: 支持 Windows/Linux/macOS
- **无外部依赖**: 仅依赖标准 C 库和 math.h

### 性能指标

| 指标 | 值 |
|------|------|
| 延迟 | ~32ms (@ 48kHz) |
| RTF | ~0.05 (20倍快于实时) |
| 内存占用 | ~8MB |
| 采样率 | 16kHz / 48kHz |

## 目录结构

```
Unit_C/
├── 核心模型
│   ├── gtcrn_model.h/c          # GTCRN 完整模型定义
│   ├── gtcrn_modules.h/c        # ERB、SFE、TRA 模块
│   ├── gtcrn_streaming.h/c      # 流式处理 API
│   └── gtcrn_streaming_optimized.c  # 优化的流式实现
│
├── 神经网络层
│   ├── conv2d.h/c               # Conv2d / ConvTranspose2d
│   ├── GRU.h/c                  # GRU / Grouped GRU
│   ├── batchnorm2d.h/c          # BatchNorm2d
│   ├── layernorm.h/c            # LayerNorm
│   └── nn_layers.h/c            # Linear、PReLU、Tanh 等
│
├── 信号处理
│   ├── stft.h/c                 # STFT / iSTFT (48kHz)
│   ├── stft_16k.h/c             # STFT / iSTFT (16kHz)
│   └── stream_conv.h/c          # 流式卷积
│
├── 工具
│   ├── weight_loader.h/c        # 权重加载器
│   ├── export_weights.py        # PyTorch 权重导出脚本
│   └── export_gru_weights.py    # GRU 权重导出脚本
│
├── 示例程序
│   ├── example_realtime_denoise.c      # 实时降噪示例 (48kHz)
│   └── example_realtime_denoise_16k.c  # 实时降噪示例 (16kHz)
│
└── 构建文件
    ├── CMakeLists.txt           # CMake 构建配置
    ├── Makefile                 # GNU Make 构建配置
    └── build_16k.bat/.sh        # 16kHz 版本构建脚本
```

## 快速开始

### 使用 CMake 构建 (推荐)

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake ..

# 构建
cmake --build .

# 运行测试
ctest --output-on-failure
```

### 使用 Make 构建

```bash
# 构建所有目标
make

# 构建并运行测试
make test

# 清理构建文件
make clean
```

### Windows 构建

```batch
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
```

## 使用方法

### 高级 API (推荐)

```c
#include "gtcrn_streaming.h"

int main() {
    // 1. 创建模型
    GTCRN* model = gtcrn_create();

    // 2. 加载权重
    gtcrn_load_weights(model, "weights/");

    // 3. 创建流式处理器
    GTCRNStreaming* stream = gtcrn_streaming_create(model, 48000, 768);

    // 4. 处理音频
    float input[768], output[768];
    while (has_more_audio()) {
        read_audio_chunk(input, 768);
        gtcrn_streaming_process_chunk(stream, input, output);
        write_audio_chunk(output, 768);
    }

    // 5. 清理
    gtcrn_streaming_free(stream);
    gtcrn_free(model);

    return 0;
}
```

### 底层 API

```c
#include "gtcrn_model.h"
#include "stft.h"

// 直接使用模型进行频谱处理
float spec_input[769 * 1 * 2];   // (F, T, 2) 复数频谱
float spec_output[769 * 1 * 2];  // 增强后的频谱

// 分配缓存
float* conv_cache = calloc(conv_cache_size, sizeof(float));
float* tra_cache = calloc(tra_cache_size, sizeof(float));
float* inter_cache = calloc(inter_cache_size, sizeof(float));

// 逐帧处理
gtcrn_forward_stream(spec_input, spec_output,
                     conv_cache, tra_cache, inter_cache,
                     1, 769, model);
```

## 导出权重

从训练好的 PyTorch 模型导出权重：

```bash
python export_weights.py --model ../gtcrn/checkpoints/model.pth --output weights/
```

## 编译选项

### 优化选项

```cmake
# 高性能编译
cmake -DCMAKE_BUILD_TYPE=Release ..

# 调试编译
cmake -DCMAKE_BUILD_TYPE=Debug ..

# 启用 SIMD 优化 (如果支持)
cmake -DENABLE_SIMD=ON ..
```

### 目标平台

```cmake
# 16kHz 版本
cmake -DSAMPLE_RATE=16000 ..

# 48kHz 版本 (默认)
cmake -DSAMPLE_RATE=48000 ..
```

## API 参考

### 核心结构体

| 结构体 | 描述 |
|--------|------|
| `GTCRN` | GTCRN 模型 |
| `GTCRNStreaming` | 流式处理器 |
| `GRUWeights` | GRU 权重 |
| `Conv2dParams` | 卷积参数 |

### 核心函数

| 函数 | 描述 |
|------|------|
| `gtcrn_create()` | 创建 GTCRN 模型 |
| `gtcrn_streaming_create()` | 创建流式处理器 |
| `gtcrn_streaming_process_chunk()` | 处理音频块 |
| `gtcrn_streaming_reset()` | 重置处理器状态 |

## 网络架构

```
Input (48kHz audio)
    │
    ▼
STFT (1536-point FFT, 768 hop)
    │
    ▼
ERB Compression (769 → 385 bins)
    │
    ▼
Encoder (5 layers)
├── Conv2d + BN + PReLU (×2)
└── GTConvBlock (×3, dilation: 1, 2, 5)
    │
    ▼
DPGRNN (×2)
├── Intra-RNN (Bidirectional GRNN)
└── Inter-RNN (Unidirectional GRNN)
    │
    ▼
Decoder (5 layers)
├── GTConvBlock (×3, dilation: 5, 2, 1)
└── ConvTranspose2d + BN + PReLU/Tanh (×2)
    │
    ▼
ERB Expansion (385 → 769 bins)
    │
    ▼
Complex Mask Application
    │
    ▼
iSTFT
    │
    ▼
Output (Enhanced audio)
```

## 许可证

本项目是 GTCRN 论文的 C 语言实现，仅供学习和研究使用。

## 参考文献

- [GTCRN Paper](https://arxiv.org/abs/2202.08537) - GTCRN: A Speech Enhancement Model Requiring Ultralow Computational Resources (ICASSP 2024)
- [Original PyTorch Implementation](https://github.com/Xiaobin-Rong/gtcrn)
