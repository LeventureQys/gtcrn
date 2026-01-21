# GTCRN 完整模型 C 实现

## 🎉 模型框架已完成！

完整的 GTCRN (Group Temporal Convolutional Recurrent Network) 语音增强模型框架。

## 📦 已创建文件

| 文件 | 说明 |
|------|------|
| [gtcrn_model.h](gtcrn_model.h) | 模型头文件 |
| [gtcrn_model.c](gtcrn_model.c) | **模型实现** |
| [test_gtcrn_model.c](test_gtcrn_model.c) | 测试程序（6个测试） |
| [Makefile_gtcrn](Makefile_gtcrn) | 编译配置 |

## 🚀 快速开始

### Windows

```batch
cd Unit_C
gcc -Wall -O2 -std=c99 -c conv2d.c
gcc -Wall -O2 -std=c99 -c batchnorm2d.c
gcc -Wall -O2 -std=c99 -c nn_layers.c
gcc -Wall -O2 -std=c99 -c layernorm.c
gcc -Wall -O2 -std=c99 -c gtcrn_model.c
gcc -Wall -O2 -std=c99 -c test_gtcrn_model.c
gcc conv2d.o batchnorm2d.o nn_layers.o layernorm.o gtcrn_model.o test_gtcrn_model.o -o test_gtcrn_model.exe -lm
test_gtcrn_model.exe
```

### Linux/Mac

```bash
cd Unit_C
make -f Makefile_gtcrn run
```

## 🏗️ 模型架构

### 完整流程

```
输入音频 (48kHz, 1秒)
    ↓
STFT (1536 FFT, 768 hop)
    ↓
复数频谱 (B, 769, 63, 2)
    ↓
┌─────────────────────────────────────┐
│  GTCRN 模型                          │
│                                     │
│  1. 预处理                           │
│     - 分离实部/虚部                  │
│     - 计算幅度                       │
│     - 堆叠: (B, 3, T, 769)          │
│                                     │
│  2. ERB 压缩                         │
│     - 769 bins -> 385 bins         │
│                                     │
│  3. SFE (Subband Feature Extract)   │
│     - Unfold: (B, 3, T, 385)       │
│     - Output: (B, 9, T, 385)       │
│                                     │
│  4. Encoder (5 层)                  │
│     - ConvBlock 1                   │
│     - ConvBlock 2                   │
│     - GTConvBlock 1 (dilation=1)    │
│     - GTConvBlock 2 (dilation=2)    │
│     - GTConvBlock 3 (dilation=5)    │
│     - Output: (B, 16, T, 97)       │
│                                     │
│  5. DPGRNN (2 层)                   │
│     - Dual-Path RNN                 │
│     - Output: (B, 16, T, 97)       │
│                                     │
│  6. Decoder (5 层，镜像 Encoder)     │
│     - 使用跳跃连接                   │
│     - Output: (B, 2, T, 385)       │
│                                     │
│  7. ERB 恢复                         │
│     - 385 bins -> 769 bins         │
│                                     │
│  8. 复数掩码                         │
│     - 应用到输入频谱                 │
│                                     │
└─────────────────────────────────────┘
    ↓
增强频谱 (B, 769, 63, 2)
    ↓
iSTFT
    ↓
增强音频 (48kHz, 1秒)
```

## 📊 模型组件

### 1. ConvBlock

```c
typedef struct {
    FusedConvBN fused_conv_bn;  // Conv + BN 融合
    PReLUParams* prelu;         // PReLU 激活
    int use_tanh;               // 最后一层使用 Tanh
} ConvBlock;
```

**组成**：
- Conv2d / ConvTranspose2d
- BatchNorm2d（融合到 Conv）
- PReLU / Tanh

### 2. GTConvBlock

```c
typedef struct {
    UnfoldParams sfe_params;    // SFE
    FusedConvBN point_conv1;    // Point Conv + BN
    FusedConvBN depth_conv;     // Depth Conv + BN
    FusedConvBN point_conv2;    // Point Conv + BN
    // TRA (需要 GRU)
} GTConvBlock;
```

**组成**：
- SFE (Subband Feature Extraction)
- Point Conv + BN + PReLU
- Depth Conv + BN + PReLU
- Point Conv + BN
- TRA (Temporal Recurrent Attention)

### 3. Encoder

```c
typedef struct {
    ConvBlock* conv1;       // 9 -> 16
    ConvBlock* conv2;       // 16 -> 16, groups=2
    GTConvBlock* gtconv1;   // dilation=1
    GTConvBlock* gtconv2;   // dilation=2
    GTConvBlock* gtconv3;   // dilation=5
} Encoder;
```

**输入**: (B, 9, T, 385)
**输出**: (B, 16, T, 97)

### 4. DPGRNN

```c
typedef struct {
    // Intra RNN
    LinearParams* intra_fc;
    LayerNormParams* intra_ln;

    // Inter RNN
    LinearParams* inter_fc;
    LayerNormParams* inter_ln;
} DPGRNN;
```

**输入**: (B, 16, T, 97)
**输出**: (B, 16, T, 97)

### 5. Decoder

```c
typedef struct {
    GTConvBlock* gtconv1;   // dilation=5, deconv
    GTConvBlock* gtconv2;   // dilation=2, deconv
    GTConvBlock* gtconv3;   // dilation=1, deconv
    ConvBlock* conv1;       // 16 -> 16, deconv
    ConvBlock* conv2;       // 16 -> 2, deconv, tanh
} Decoder;
```

**输入**: (B, 16, T, 97)
**输出**: (B, 2, T, 385)

## 💻 使用示例

### 创建模型

```c
#include "gtcrn_model.h"

// 创建模型
GTCRN* model = gtcrn_create();

// 打印模型信息
print_gtcrn_info(model);
```

### 前向传播

```c
// 输入参数
int batch = 1;
int freq_bins = 769;  // 48kHz
int time_frames = 63; // ~1秒

// 分配内存
int total_size = batch * freq_bins * time_frames * 2;
float* spec_input = (float*)malloc(total_size * sizeof(float));
float* spec_output = (float*)malloc(total_size * sizeof(float));

// 填充输入（从 STFT 获取）
// ...

// 前向传播
gtcrn_forward(
    spec_input,
    spec_output,
    batch,
    freq_bins,
    time_frames,
    model
);

// 清理
free(spec_input);
free(spec_output);
gtcrn_free(model);
```

### 完整音频处理流程

```c
// 1. 读取音频
float* audio = load_audio("noisy.wav", &sample_rate, &num_samples);

// 2. STFT
float* spec = stft(audio, num_samples, 1536, 768);

// 3. GTCRN 处理
GTCRN* model = gtcrn_create();
float* spec_enhanced = (float*)malloc(...);
gtcrn_forward(spec, spec_enhanced, batch, 769, time_frames, model);

// 4. iSTFT
float* audio_enhanced = istft(spec_enhanced, ...);

// 5. 保存音频
save_audio("enhanced.wav", audio_enhanced, sample_rate, num_samples);
```

## ✅ 已实现的基础层

| 层类型 | 文件 | 状态 |
|--------|------|------|
| Conv2d | conv2d.c | ✅ 完成 |
| ConvTranspose2d | conv2d.c | ✅ 完成 |
| BatchNorm2d | batchnorm2d.c | ✅ 完成 |
| Conv+BN 融合 | batchnorm2d.c | ✅ 完成 |
| Linear | nn_layers.c | ✅ 完成 |
| Unfold | nn_layers.c | ✅ 完成 |
| PReLU | nn_layers.c | ✅ 完成 |
| Sigmoid | nn_layers.c | ✅ 完成 |
| Tanh | conv2d.c | ✅ 完成 |
| LayerNorm | layernorm.c | ✅ 完成 |
| Parameter | layernorm.c | ✅ 完成 |

## 🔨 待完成的工作

### 1. GRU 层（最关键）

```c
// 需要实现
typedef struct {
    LinearParams* weight_ih;  // input-hidden
    LinearParams* weight_hh;  // hidden-hidden
    float* bias_ih;
    float* bias_hh;
} GRUParams;

void gru_forward(
    const float* input,   // (batch, seq_len, input_size)
    float* output,        // (batch, seq_len, hidden_size)
    float* hidden,        // (num_layers, batch, hidden_size)
    GRUParams* params
);
```

### 2. ERB 压缩/恢复

```c
// ERB (Equivalent Rectangular Bandwidth)
void erb_compress(
    const Tensor* input,   // (B, C, T, 769)
    Tensor* output,        // (B, C, T, 385)
    ERBParams* params
);

void erb_decompress(
    const Tensor* input,   // (B, C, T, 385)
    Tensor* output,        // (B, C, T, 769)
    ERBParams* params
);
```

### 3. 完整的 GTConvBlock

需要实现：
- Channel split/shuffle
- 完整的 TRA 模块（需要 GRU）
- 所有卷积层的权重加载

### 4. 模型权重加载

```c
// 从 PyTorch 模型加载权重
int load_gtcrn_weights(
    GTCRN* model,
    const char* weight_file
);
```

### 5. STFT/iSTFT

```c
// 短时傅里叶变换
void stft(
    const float* audio,
    int num_samples,
    int n_fft,
    int hop_length,
    float* spec_real,
    float* spec_imag
);

// 逆短时傅里叶变换
void istft(
    const float* spec_real,
    const float* spec_imag,
    int n_fft,
    int hop_length,
    float* audio
);
```

## 📈 性能指标

### 模型参数

| 指标 | 值 |
|------|-----|
| 总参数 | 23.67K |
| 计算量 | 33.0 MMACs |
| 模型大小 | ~95 KB |

### 运行时性能

| 指标 | 目标值 |
|------|--------|
| 实时因子 | < 0.1 (CPU) |
| 延迟 | < 50ms |
| 内存占用 | < 10MB |

## 🎯 下一步计划

### 短期（1-2周）

1. ✅ 实现 GRU 层
2. ✅ 实现 ERB 压缩/恢复
3. ✅ 完整的 GTConvBlock
4. ✅ 模型权重加载

### 中期（2-4周）

5. ✅ STFT/iSTFT 集成
6. ✅ 端到端音频处理
7. ✅ 性能优化
8. ✅ 实时音频流处理

### 长期（1-2月）

9. ✅ SIMD 优化
10. ✅ 多线程支持
11. ✅ 移动端优化
12. ✅ 硬件加速（GPU/NPU）

## 📚 测试

程序运行 **6 个测试**：

1. **Test 1**: GTCRN 模型创建
2. **Test 2**: GTCRN 前向传播
3. **Test 3**: ConvBlock 测试
4. **Test 4**: DPGRNN 测试
5. **Test 5**: 复数掩码测试
6. **Test 6**: 完整流程说明

## 🔍 代码结构

```
Unit_C/
├── 基础层
│   ├── conv2d.h/c              ← Conv2d, ConvTranspose2d
│   ├── batchnorm2d.h/c         ← BatchNorm2d, 融合优化
│   ├── nn_layers.h/c           ← Linear, Unfold, PReLU, Sigmoid
│   └── layernorm.h/c           ← LayerNorm, Parameter
│
├── 模型
│   ├── gtcrn_model.h           ← 模型定义
│   ├── gtcrn_model.c           ← 模型实现
│   └── test_gtcrn_model.c      ← 测试程序
│
└── 构建
    └── Makefile_gtcrn          ← 编译配置
```

## ⚠️ 注意事项

### 当前版本

这是 **框架版本**，包含：
- ✅ 完整的模型结构
- ✅ 所有基础层实现
- ✅ 模型管理接口
- ⚠️ 简化的前向传播（需要 GRU）

### 完整版本需要

1. **GRU 实现** - 最关键
2. **权重加载** - 从 PyTorch 模型
3. **ERB 模块** - 频率压缩/恢复
4. **STFT/iSTFT** - 音频处理

## 💡 使用建议

### 学习和测试

```bash
# 编译运行测试
make -f Makefile_gtcrn run

# 查看模型结构
# 理解各个组件
# 为完整实现做准备
```

### 开发完整版本

1. 先实现 GRU 层
2. 实现权重加载
3. 逐步完善各个模块
4. 端到端测试

## ✨ 总结

### 已完成

- ✅ **完整的模型框架**
- ✅ **所有基础层**（10个）
- ✅ **模型管理**
- ✅ **测试程序**
- ✅ **详细文档**

### 特点

- 🚀 **超轻量级** - 23.67K 参数
- ⚡ **高效率** - 33.0 MMACs
- 📦 **模块化** - 易于扩展
- 🎯 **实时处理** - 低延迟

### 下一步

实现 GRU 层，完成完整的 GTCRN 模型！

---

**创建时间**: 2025-12-18
**语言**: C99
**状态**: 框架完成 ✅
**下一步**: 实现 GRU 层
