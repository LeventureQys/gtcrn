# GTCRN 实时流式处理实现状态

## 概述

本文档记录了GTCRN模型实时流式降噪处理的实现状态和缺失组件。

## 已完成的组件 ✓

### 1. TRA (Temporal Recurrent Attention) 模块
- **状态**: ✓ 已修复并增强
- **文件**: `gtcrn_modules.c`, `gtcrn_modules.h`
- **修复内容**:
  - 修复了 `tra_free()` 函数中的错误引用 (line 522)
  - 添加了 `tra_forward_stream()` 函数，支持GRU隐藏状态缓存
  - 支持帧间状态传递，实现真正的流式处理

**流式TRA函数签名**:
```c
void tra_forward_stream(
    const Tensor* input,      // (B, C, T, F) - 通常 T=1
    Tensor* output,           // (B, C, T, F)
    float* h_cache,           // (1, B, channels*2) - GRU隐藏状态
    TRAParams* params
);
```

### 2. 基础模块
- ✓ ERB (Equivalent Rectangular Bandwidth) - 频率压缩/恢复
- ✓ SFE (Subband Feature Extraction) - 子带特征提取
- ✓ ConvBlock - 卷积块
- ✓ GTConvBlock - 分组时间卷积块
- ✓ DPGRNN - 双路径分组RNN
- ✓ Encoder/Decoder - 编码器/解码器

## 实时流式处理所需的状态缓存

根据Python实现 (`stream/gtcrn_stream.py`)，完整的流式处理需要以下状态缓存：

### 1. 卷积缓存 (conv_cache)
- **形状**: `(2, B, C, (kT-1)*dilation, F)`
  - 2: [encoder_cache, decoder_cache]
  - B: batch size (通常为1)
  - C: channels (16)
  - (kT-1)*dilation: 时间维度的缓存大小
  - F: frequency bins (97)

- **用途**: 保存膨胀卷积的历史帧
- **位置**: GTConvBlock 中的 depth_conv

### 2. TRA缓存 (tra_cache)
- **形状**: `(2, 3, 1, B, C)`
  - 2: [encoder_cache, decoder_cache]
  - 3: 3个GTConvBlock层
  - 1: GRU层数
  - B: batch size
  - C: channels (16)

- **用途**: 保存TRA模块中GRU的隐藏状态
- **状态**: ✓ 已实现 `tra_forward_stream()`

### 3. Inter-RNN缓存 (inter_cache)
- **形状**: `(2, 1, B*F, C)`
  - 2: [dpgrnn1_cache, dpgrnn2_cache]
  - 1: GRU层数
  - B*F: batch * frequency bins
  - C: channels (16)

- **用途**: 保存DPGRNN中Inter-RNN的隐藏状态
- **状态**: ⚠️ 需要在DPGRNN中添加流式支持

## 缺失的组件 ⚠️

### 1. StreamConv2d / StreamConvTranspose2d
- **状态**: ⚠️ 需要实现
- **用途**: 支持卷积缓存的流式卷积操作
- **参考**: Python实现在 `modules/convolution.py`

**需要实现的功能**:
```c
// 流式卷积前向传播
void stream_conv2d_forward(
    const Tensor* input,      // (B, C, T, F) - T通常为1
    Tensor* output,           // (B, C_out, T, F_out)
    float* conv_cache,        // (B, C, (kT-1)*dT, F) - 历史帧缓存
    const Conv2dParams* params
);
```

### 2. DPGRNN流式支持
- **状态**: ⚠️ 需要添加
- **当前**: `dpgrnn_forward()` 不支持状态缓存
- **需要**: 添加 `dpgrnn_forward_stream()` 函数

**需要实现的功能**:
```c
void dpgrnn_forward_stream(
    const Tensor* input,      // (B, C, T, F)
    Tensor* output,           // (B, C, T, F)
    float* inter_cache,       // (1, B*F, hidden_size) - Inter-RNN状态
    DPGRNN* dpgrnn
);
```

### 3. GTConvBlock流式支持
- **状态**: ⚠️ 需要增强
- **当前**: 使用普通卷积，不支持缓存
- **需要**: 集成 StreamConv2d 和 tra_forward_stream

**需要实现的功能**:
```c
void gtconvblock_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,        // 卷积缓存
    float* tra_cache,         // TRA GRU缓存
    GTConvBlock* block
);
```

### 4. 完整的流式GTCRN模型
- **状态**: ⚠️ 需要实现
- **需要**: 整合所有流式组件

**需要实现的功能**:
```c
void gtcrn_forward_stream(
    const float* spec_input,  // (B, F, T, 2) - T=1
    float* spec_output,       // (B, F, T, 2)
    float* conv_cache,        // (2, B, C, 16, F)
    float* tra_cache,         // (2, 3, 1, B, C)
    float* inter_cache,       // (2, 1, B*F, C)
    GTCRN* model
);
```

## 实现优先级

### 高优先级 (必需)
1. **StreamConv2d** - 核心流式卷积操作
2. **DPGRNN流式支持** - Inter-RNN状态管理
3. **GTConvBlock流式集成** - 整合卷积和TRA缓存

### 中优先级 (重要)
4. **完整流式GTCRN** - 顶层流式接口
5. **状态初始化函数** - 便捷的缓存创建和管理

### 低优先级 (优化)
6. **内存优化** - 减少临时缓冲区分配
7. **性能优化** - SIMD加速、多线程等

## 实时处理流程

```
输入音频帧 (16ms @ 48kHz = 768 samples)
    ↓
STFT (1536 FFT, 768 hop) → (1, 769, 1, 2)
    ↓
gtcrn_forward_stream(spec, conv_cache, tra_cache, inter_cache)
    ├─ ERB压缩: 769 → 385 bins
    ├─ SFE: 3 → 9 channels
    ├─ Encoder (使用 conv_cache, tra_cache)
    │   ├─ ConvBlock x2
    │   └─ GTConvBlock x3 (stream)
    ├─ DPGRNN x2 (使用 inter_cache)
    ├─ Decoder (使用 conv_cache, tra_cache)
    │   ├─ GTConvBlock x3 (stream)
    │   └─ ConvBlock x2
    ├─ ERB恢复: 385 → 769 bins
    └─ 复数掩码应用
    ↓
ISTFT → 增强音频帧
```

## 性能目标

- **延迟**: < 16ms (单帧处理时间)
- **RTF**: < 0.1 (实时因子)
- **内存**: < 10MB (包括所有缓存)

## 测试计划

1. **单元测试**
   - 测试 `tra_forward_stream()` 与批处理版本的一致性
   - 测试状态缓存的正确传递

2. **集成测试**
   - 逐帧处理与批处理结果对比
   - 验证流式误差 < 1e-5

3. **性能测试**
   - 测量单帧处理时间
   - 测量内存使用

## 参考文件

- Python流式实现: `stream/gtcrn_stream.py`
- Python卷积模块: `modules/convolution.py`
- C语言TRA实现: `Unit_C/gtcrn_modules.c`
- C语言模型实现: `Unit_C/gtcrn_model.c`

## 更新日志

### 2024-12-19
- ✓ 修复 TRA 模块的 `tra_free()` 错误
- ✓ 实现 `tra_forward_stream()` 支持GRU状态缓存
- ✓ 添加流式处理状态管理文档
- ⚠️ 识别缺失的StreamConv2d和DPGRNN流式支持
