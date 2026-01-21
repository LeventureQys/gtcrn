# GTCRN C 语言实现

GTCRN（Grouped Temporal Convolutional Recurrent Network）的纯 C 语言实现，用于实时语音增强。

## 简介

本项目是 GTCRN 模型的 C 语言移植版本，从 PyTorch 模型转换而来。GTCRN 是一个超轻量级语音降噪模型，仅有 48.2K 参数，适用于资源受限的嵌入式设备。

## 特性

- 纯 C99 实现，仅依赖标准 C 库和数学库
- 跨平台支持：Windows、Linux、macOS 及嵌入式系统
- 代码量约 2500 行
- 支持两种推理模式：
  - 离线模式：一次性处理整个音频文件
  - 流式模式：逐帧处理，适用于实时应用

## 目录结构

```
GTCRN_C/
├── include/           # 头文件
│   ├── gtcrn_types.h  # 通用类型和常量定义
│   ├── gtcrn_math.h   # 数学工具函数
│   ├── gtcrn_layers.h # 神经网络层实现
│   ├── gtcrn_fft.h    # FFT/STFT 实现
│   └── gtcrn_model.h  # 主模型 API
├── src/               # 源文件
│   ├── gtcrn_math.c
│   ├── gtcrn_layers.c
│   ├── gtcrn_fft.c
│   ├── gtcrn_model.c
│   ├── gtcrn_forward.c  # 离线推理
│   └── gtcrn_stream.c   # 流式推理
├── demo/              # 示例程序
│   ├── main.c         # 离线模式示例
│   ├── main_stream.c  # 流式模式示例
│   ├── wav_io.h
│   └── wav_io.c
├── scripts/           # 工具脚本
│   └── export_weights.py  # 权重导出脚本
├── weights/           # 模型权重（需生成）
└── CMakeLists.txt
```

## 编译

### 依赖

- CMake 3.12 或更高版本
- C 编译器（GCC、Clang、MSVC）

### 编译步骤

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

Windows (MSVC) 环境：

```cmd
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Release
```

### 编译选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `GTCRN_USE_DOUBLE` | OFF | 使用双精度浮点 |
| `GTCRN_BUILD_DEBUG_TOOLS` | OFF | 编译调试工具 |

## 使用方法

### 1. 导出权重

从 PyTorch 模型导出权重为二进制格式：

```bash
cd scripts
python export_weights.py --model ../../checkpoints/model_trained_on_dns3.tar --output ../weights/gtcrn_weights.bin
```

### 2. 运行示例

离线模式：

```bash
./gtcrn_demo weights/gtcrn_weights.bin input.wav output.wav
```

流式模式：

```bash
./gtcrn_stream_demo weights/gtcrn_weights.bin input.wav output.wav
```

### 3. API 使用

离线模式：

```c
#include "gtcrn_model.h"

gtcrn_t* model = gtcrn_create();
gtcrn_load_weights(model, "weights/gtcrn_weights.bin");

int output_len;
gtcrn_process(model, input, input_len, output, &output_len);

gtcrn_destroy(model);
```

流式模式：

```c
#include "gtcrn_model.h"

#define FRAME_SIZE 256  // 16ms @ 16kHz

gtcrn_t* model = gtcrn_create();
gtcrn_load_weights(model, "weights/gtcrn_weights.bin");
gtcrn_reset_state(model);

float input_frame[FRAME_SIZE];
float output_frame[FRAME_SIZE];

while (has_more_audio()) {
    read_audio_frame(input_frame);
    gtcrn_process_frame(model, input_frame, output_frame);
    write_audio_frame(output_frame);
}

gtcrn_destroy(model);
```

## 流式模式参数

| 参数 | 值 |
|------|-----|
| 采样率 | 16 kHz |
| 帧长 | 256 样本 |
| 帧时长 | 16 ms |
| FFT 点数 | 512 |

### 状态管理

流式模式在帧之间维护内部状态：

- 卷积缓存：存储膨胀卷积所需的历史帧
- RNN 隐藏状态：TRA 和 DPGRNN 状态
- 重叠相加缓冲区：用于 ISTFT 重建

处理新音频流前需调用 `gtcrn_reset_state()` 重置状态。

### 实时性能

| 平台 | 平均帧处理时间 | RTF |
|------|----------------|-----|
| Intel i5-12400 | ~0.8 ms | ~0.05 |
| ARM Cortex-A72 | ~2.5 ms | ~0.15 |

RTF（实时因子）小于 1.0 表示可实时处理。帧处理时间需小于 16ms。

## 模型结构

GTCRN 采用 U-Net 风格的编码器-解码器结构：

- ERB 滤波器组：将 257 个频点压缩到 129 个 ERB 频带
- SFE（子带特征提取）：捕获局部频率上下文
- GTConvBlock：分组时序卷积，膨胀系数为 1、2、5
- TRA（时序循环注意力）：基于 GRU 的时序加权
- DPGRNN：双路径分组 RNN
- 复数比值掩膜：输出应用于输入频谱图

## 许可证

本实现遵循原始 GTCRN 项目的许可证。

## 参考

- [GTCRN 论文](https://arxiv.org/abs/2303.04090) - ICASSP 2024
- [原始 PyTorch 实现](https://github.com/Xiaobin-Rong/gtcrn)
