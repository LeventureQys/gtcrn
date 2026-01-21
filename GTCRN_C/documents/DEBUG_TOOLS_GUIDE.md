# GTCRN C 调试工具使用指南

本文档介绍 GTCRN C 实现中的调试和验证工具的使用方法。

## 目录

1. [编译调试工具](#编译调试工具)
2. [主要演示程序](#主要演示程序)
3. [调试工具详解](#调试工具详解)
4. [Python 比较脚本](#python-比较脚本)
5. [常见调试场景](#常见调试场景)

---

## 编译调试工具

调试工具默认不编译。需要启用 `GTCRN_BUILD_DEBUG_TOOLS` 选项：

```bash
# 进入构建目录
cd GTCRN_C/build

# 配置并启用调试工具
cmake .. -DGTCRN_BUILD_DEBUG_TOOLS=ON

# 如果需要流式调试输出，额外启用：
cmake .. -DGTCRN_BUILD_DEBUG_TOOLS=ON -DGTCRN_STREAM_DEBUG_ENABLE=ON

# 编译
cmake --build . --config Release
```

编译后的调试工具位于 `build/Release/` 目录。

---

## 主要演示程序

### gtcrn_demo - 离线处理演示

处理完整音频文件（一次性处理）：

```bash
gtcrn_demo.exe <权重文件> <输入WAV> <输出WAV>

# 示例
gtcrn_demo.exe weights/gtcrn_weights.bin input.wav output.wav
```

### gtcrn_stream_demo - 流式处理演示

逐帧流式处理音频（实时模式）：

```bash
gtcrn_stream_demo.exe <权重文件> <输入WAV> <输出WAV>

# 示例
gtcrn_stream_demo.exe weights/gtcrn_weights.bin input.wav output_stream.wav
```

输出包含：
- 处理进度
- 帧处理时间统计
- 实时因子（RTF）
- 如果启用 `GTCRN_STREAM_DEBUG_ENABLE`，还会输出第6帧的中间值

---

## 调试工具详解

### 1. debug_engt2 - EnGT2 中间值调试

**用途**：比较 C 和 Python 在 EnGT2（编码器第一个GTConvBlock）的中间值。

```bash
debug_engt2.exe <权重文件> <输入WAV>

# 示例
debug_engt2.exe weights/gtcrn_weights.bin test_wavs/noisy_16k/test.wav
```

**输出示例**：
```
=== C Frame 6 (Python frame 5) ===

EnConv0 output abs_sum: 151.607496
EnConv1 output abs_sum: 112.684705
EnGT2 output abs_sum: 142.484015

--- EnGT2 per-channel sums ---
  ch0: 16.708643
  ch1: 26.233355
  ...
```

**对应 Python 脚本**：`gtcrn/debug_engt2_hooks.py`

---

### 2. test_stft_match - STFT 一致性测试

**用途**：验证 C 的 STFT 实现与 Python 匹配。

```bash
test_stft_match.exe <权重文件> <输入WAV>
```

**输出**：比较 C 和 Python 的 STFT 频谱值。

---

### 3. test_stft_istft_roundtrip - STFT/ISTFT 往返测试

**用途**：验证 STFT -> ISTFT 能正确重建信号。

```bash
test_stft_istft_roundtrip.exe
```

**预期**：输入信号和重建信号应高度相关（相关系数 > 0.99）。

---

### 4. test_fft_roundtrip - FFT 往返测试

**用途**：验证基本 FFT/IFFT 实现的正确性。

```bash
test_fft_roundtrip.exe
```

---

### 5. test_stream_debug - 流式处理调试

**用途**：调试流式处理的帧对齐和状态管理。

```bash
test_stream_debug.exe <权重文件> <输入WAV>
```

---

### 6. test_frame0_debug - 第一帧调试

**用途**：专门调试第一帧（frame 0）的处理，检查初始状态。

```bash
test_frame0_debug.exe <权重文件> <输入WAV>
```

---

### 7. test_stream_spectrum - 流式频谱输出

**用途**：保存流式处理的频谱输出，用于与 Python 比较。

```bash
test_stream_spectrum.exe <权重文件> <输入WAV>
```

---

### 8. test_save_spectrum - 保存频谱数据

**用途**：将 STFT 频谱保存到文件，供 Python 脚本加载比较。

```bash
test_save_spectrum.exe <权重文件> <输入WAV>
```

---

### 9. test_stft_frame0 - STFT 第一帧测试

**用途**：专门测试 STFT 第一帧的计算。

```bash
test_stft_frame0.exe <权重文件> <输入WAV>
```

---

### 10. test_intermediate - 中间层输出

**用途**：输出网络各层的中间结果，用于逐层对比。

```bash
test_intermediate.exe <权重文件> <输入WAV>
```

---

### 11. debug_forward - 前向传播调试

**用途**：调试离线模式的完整前向传播。

```bash
debug_forward.exe <权重文件>
```

---

### 12. debug_enconv0 - EnConv0 调试

**用途**：专门调试编码器第一个卷积层。

```bash
debug_enconv0.exe <权重文件>
```

---

### 13. debug_full_forward - 完整前向传播调试

**用途**：调试完整的离线前向传播流程。

```bash
debug_full_forward.exe <权重文件>
```

---

### 14. debug_gtconv - GTConvBlock 调试

**用途**：专门调试 GTConvBlock 模块。

```bash
debug_gtconv.exe <权重文件>
```

---

### 15. debug_tra - TRA 模块调试

**用途**：调试 Temporal Recurrent Attention 模块。

```bash
debug_tra.exe <权重文件>
```

---

## Python 比较脚本

这些脚本位于 `gtcrn/` 目录下：

### compare_stream_simple.py

**用途**：比较 C 和 Python 流式输出的时域信号。

```bash
cd gtcrn
python compare_stream_simple.py
```

**输出**：
- 相关系数
- 能量比
- SNR

**注意**：需要先运行 `gtcrn_stream_demo` 生成 C 输出。

---

### debug_engt2_hooks.py

**用途**：使用 PyTorch forward hooks 追踪 EnGT2 的所有中间值。

```bash
cd gtcrn
python debug_engt2_hooks.py
```

**输出**：
- 每个子模块（SFE, PointConv1, BN, PReLU, DepthConv 等）的 abs_sum
- 前10个元素值
- 每通道求和

---

### compare_spectrum.py

**用途**：在频谱级别比较 C 和 Python 输出，避免 ISTFT 差异。

```bash
cd gtcrn
python compare_spectrum.py
```

---

## 常见调试场景

### 场景1：验证 C 流式输出与 Python 匹配

1. 生成 C 输出：
   ```bash
   gtcrn_stream_demo.exe weights/gtcrn_weights.bin input.wav c_output.wav
   ```

2. 运行比较脚本：
   ```bash
   python compare_stream_simple.py
   ```

3. 预期结果：
   - 相关系数 > 0.85
   - 能量比接近 1.0

---

### 场景2：定位中间层差异

1. 启用详细调试输出重新编译：
   ```bash
   cmake .. -DGTCRN_BUILD_DEBUG_TOOLS=ON -DGTCRN_STREAM_DEBUG_ENABLE=ON
   cmake --build . --config Release
   ```

2. 运行 C 调试工具：
   ```bash
   debug_engt2.exe weights/gtcrn_weights.bin input.wav > c_debug.txt
   ```

3. 运行 Python 调试脚本：
   ```bash
   python debug_engt2_hooks.py > python_debug.txt
   ```

4. 比较两个输出文件中的 abs_sum 值。

---

### 场景3：验证 STFT 实现

1. 运行往返测试：
   ```bash
   test_stft_istft_roundtrip.exe
   ```

2. 运行 STFT 匹配测试：
   ```bash
   test_stft_match.exe weights/gtcrn_weights.bin input.wav
   ```

---

### 场景4：调试特定帧

1. 修改 `gtcrn_stream.c` 中的 `GTCRN_STREAM_DEBUG` 宏：
   ```c
   #define GTCRN_STREAM_DEBUG 1
   ```

2. 修改调试帧号（默认为帧6）：
   ```c
   if (g_stream_frame_count == 6)  // 改为目标帧号
   ```

3. 重新编译并运行。

---

## 帧对齐说明

**重要**：C 和 Python 的帧索引有 1 的偏移：

- Python 帧索引从 0 开始
- C 的 `g_stream_frame_count` 从 1 开始（第一帧处理后为 1）

对应关系：
| Python 帧 | C frame_count |
|-----------|---------------|
| 0 | 1 |
| 1 | 2 |
| 5 | 6 |

Python 脚本通常在处理前对音频进行 256 样本的零填充，这与 C 的流式处理行为一致。

---

## 关键调试数值参考（Frame 5 / frame_count=6）

以下是正确实现时的参考值：

| 层 | abs_sum |
|----|---------|
| EnConv0 | 151.607 |
| EnConv1 | 112.685 |
| EnGT2 SFE | 156.276 |
| EnGT2 PointConv1 | 216.311 |
| EnGT2 BN1 | 391.342 |
| EnGT2 PReLU1 | 279.167 |
| EnGT2 DepthConv | 234.182 |
| EnGT2 DepthBN | 544.952 |
| EnGT2 DepthPReLU | 443.881 |
| EnGT2 PointConv2 | 265.316 |
| EnGT2 PointBN2 | 182.289 |
| EnGT2 TRA | 82.925 |
| EnGT2 output | 142.484 |
| EnGT3 output | 64.331 |
| EnGT4 output | 35.517 |

如果某一层的值与参考值差异较大（> 1%），说明该层或之前的层存在 bug。

---

## 已修复的重要 Bug 记录

### 1. DepthConv 缓存大小 Bug

**问题**：`cache_t` 计算公式错误
- 错误：`cache_t = (kernel_t - 1) * dilation_t`
- 正确：`cache_t = (kernel_t - 1) * dilation_t + 1`

**影响**：EnGT2/3/4 和 DeGT0/1/2 的 DepthConv 输出错误

**修复文件**：`gtcrn_stream.c`, `gtcrn_model.h`

### 2. 缓冲区重叠 Bug

**问题**：`buf2` 和 `buf3` 大小分配不足导致重叠

**修复**：正确计算各缓冲区所需大小

### 3. ConvTranspose2d 权重索引 Bug

**问题**：转置卷积的权重索引计算错误

**修复文件**：`gtcrn_stream.c`

---

## 联系与贡献

如有问题或建议，请提交 Issue 或 Pull Request。
