# GTCRN 流式处理编译指南

## 修改完成 ✅

已成功将 `gtcrn_streaming_optimized.c` 修改为使用真正的流式处理实现！

### 修改内容

#### 1. 添加必要的头文件和声明
```c
#include "stream_conv.h"      // 流式卷积
#include <math.h>

// 外部函数声明
extern void dpgrnn_forward_stream(...);
extern void gtconvblock_forward_stream(...);
```

#### 2. 修改 `encoder_forward_streaming()`
- ✅ Layer 1-2: 使用普通 ConvBlock（无需缓存）
- ✅ Layer 3-5: 使用 `gtconvblock_forward_stream()` 进行流式处理
- ✅ 正确传递卷积缓存和 TRA 缓存

#### 3. 修改 `decoder_forward_streaming()`
- ✅ Layer 1-3: 使用 `gtconvblock_forward_stream()` 进行流式处理
- ✅ Layer 4-5: 使用普通 ConvBlock
- ✅ 正确处理 skip connections

#### 4. 修改 DPGRNN 处理
- ✅ 创建 `dpgrnn_forward_streaming_wrapper()` 调用完整实现
- ✅ 使用 `dpgrnn_forward_stream()` 进行真正的流式处理
- ✅ 正确管理 Inter-RNN 状态缓存

---

## 编译方法

### 方法 1: 完整编译（推荐）

编译所有必需的源文件：

```bash
cd Unit_C

gcc -o example_realtime_denoise \
    example_realtime_denoise.c \
    gtcrn_streaming_optimized.c \
    gtcrn_streaming.c \
    gtcrn_streaming_impl.c \
    gtcrn_model.c \
    gtcrn_modules.c \
    stream_conv.c \
    stft.c \
    weight_loader.c \
    GRU.c \
    conv2d.c \
    batchnorm2d.c \
    nn_layers.c \
    layernorm.c \
    -lm -O2
```

### 方法 2: 分步编译

```bash
# 1. 编译基础模块
gcc -c conv2d.c batchnorm2d.c nn_layers.c layernorm.c GRU.c -O2

# 2. 编译 GTCRN 模块
gcc -c gtcrn_modules.c gtcrn_model.c -O2

# 3. 编译流式处理模块
gcc -c stream_conv.c gtcrn_streaming_impl.c gtcrn_streaming.c gtcrn_streaming_optimized.c -O2

# 4. 编译 STFT 和权重加载
gcc -c stft.c weight_loader.c -O2

# 5. 链接所有目标文件
gcc -o example_realtime_denoise \
    example_realtime_denoise.c \
    *.o \
    -lm -O2
```

### 方法 3: 使用 Makefile（如果存在）

```bash
make example_realtime_denoise
```

---

## 运行方法

### 基本用法

```bash
./example_realtime_denoise input.wav output.wav weights/
```

### 示例

```bash
# 使用测试音频
./example_realtime_denoise test_wavs/noisy_48k_sample2.wav test_wavs/enhanced.wav checkpoints/model_trained_on_dns3/

# 查看帮助
./example_realtime_denoise
```

---

## 关键文件依赖关系

```
example_realtime_denoise.c
    ↓
gtcrn_streaming_optimized.c
    ├─ gtcrn_streaming.h (缓存结构定义)
    ├─ stream_conv.c (流式卷积实现) ✓ 新增
    ├─ gtcrn_streaming_impl.c (流式实现) ✓ 新增
    │   ├─ dpgrnn_forward_stream() ✓
    │   └─ gtconvblock_forward_stream() ✓
    ├─ gtcrn_model.c (模型定义)
    ├─ gtcrn_modules.c (TRA, SFE, ERB)
    │   └─ tra_forward_stream() ✓
    ├─ GRU.c (GRU 实现)
    ├─ conv2d.c, batchnorm2d.c
    ├─ nn_layers.c, layernorm.c
    ├─ stft.c (STFT/iSTFT)
    └─ weight_loader.c (权重加载)
```

---

## 验证修改

### 1. 检查编译是否成功

```bash
gcc -o example_realtime_denoise \
    example_realtime_denoise.c \
    gtcrn_streaming_optimized.c \
    gtcrn_streaming.c \
    gtcrn_streaming_impl.c \
    gtcrn_model.c \
    gtcrn_modules.c \
    stream_conv.c \
    stft.c \
    weight_loader.c \
    GRU.c \
    conv2d.c \
    batchnorm2d.c \
    nn_layers.c \
    layernorm.c \
    -lm -O2 -Wall

# 应该没有错误或警告
```

### 2. 运行测试

```bash
# 运行示例程序
./example_realtime_denoise test_wavs/noisy_48k_sample2.wav test_wavs/enhanced.wav checkpoints/

# 检查输出
# 应该看到:
# - "GTCRN Streaming created"
# - "Processing X chunks..."
# - "Processing complete!"
# - "Real-time factor: X.XXX"
```

### 3. 验证流式处理是否工作

查看程序输出，确认：
- ✅ 没有 "Warning: Failed to load weights" 之外的警告
- ✅ 处理速度合理（RTF < 1.0 表示实时）
- ✅ 输出音频文件生成成功

---

## 常见编译问题

### 问题 1: 找不到头文件

```
error: stream_conv.h: No such file or directory
```

**解决方案**: 确保所有 `.h` 文件都在同一目录下，或使用 `-I` 指定头文件路径：

```bash
gcc -I./Unit_C -o example_realtime_denoise ...
```

### 问题 2: 未定义的引用

```
undefined reference to `dpgrnn_forward_stream'
```

**解决方案**: 确保包含了 `gtcrn_streaming_impl.c`：

```bash
gcc ... gtcrn_streaming_impl.c ...
```

### 问题 3: 链接数学库错误

```
undefined reference to `sqrtf'
```

**解决方案**: 添加 `-lm` 链接数学库：

```bash
gcc ... -lm
```

---

## 性能优化建议

### 编译优化选项

```bash
# 基本优化
gcc ... -O2 -lm

# 高级优化（更快但编译时间更长）
gcc ... -O3 -march=native -lm

# 调试版本（用于开发）
gcc ... -g -O0 -Wall -Wextra -lm
```

### 运行时优化

1. **使用实际权重**: 导出 PyTorch 模型权重
   ```bash
   python export_weights.py
   ```

2. **调整 chunk_size**: 在 `example_realtime_denoise.c` 中修改
   ```c
   int chunk_size = 768;  // 16ms @ 48kHz
   ```

3. **预分配缓冲区**: 减少内存分配（未来优化）

---

## 下一步

### 1. 导出模型权重

```bash
cd ..
python export_weights.py --model checkpoints/model_trained_on_dns3.pth --output Unit_C/weights/
```

### 2. 运行完整测试

```bash
cd Unit_C
./example_realtime_denoise test_wavs/noisy_48k_sample2.wav test_wavs/enhanced.wav weights/
```

### 3. 性能测试

```bash
# 测试实时因子
time ./example_realtime_denoise test_wavs/noisy_48k_sample2.wav test_wavs/enhanced.wav weights/

# 应该看到 RTF < 1.0 (实时处理)
```

---

## 技术细节

### 流式处理流程

```
音频输入 (768 samples @ 48kHz)
    ↓
STFT (1536 FFT, 768 hop) → (1, 769, 1, 2)
    ↓
gtcrn_streaming_process_frame_optimized()
    ├─ ERB 压缩: 769 → 385 bins
    ├─ SFE: 3 → 9 channels
    ├─ encoder_forward_streaming() ✓ 使用流式 GTConvBlock
    │   ├─ ConvBlock x2 (普通)
    │   └─ GTConvBlock x3 (流式 + 缓存)
    ├─ dpgrnn_forward_streaming_wrapper() ✓ 使用流式 DPGRNN
    │   ├─ Intra-RNN (双向)
    │   └─ Inter-RNN (单向 + 缓存)
    ├─ decoder_forward_streaming() ✓ 使用流式 GTConvBlock
    │   ├─ GTConvBlock x3 (流式 + 缓存)
    │   └─ ConvBlock x2 (普通)
    ├─ ERB 恢复: 385 → 769 bins
    └─ 复数掩码应用
    ↓
iSTFT → 增强音频 (768 samples)
```

### 状态缓存

- **卷积缓存**: 保存历史帧用于膨胀卷积
- **TRA 缓存**: 保存 GRU 隐藏状态
- **Inter-RNN 缓存**: 保存时间维度的 GRU 状态

---

## 总结

✅ **已完成的修改**:
1. 添加流式卷积支持 (`stream_conv.c`)
2. 集成流式 GTConvBlock (`gtconvblock_forward_stream`)
3. 集成流式 DPGRNN (`dpgrnn_forward_stream`)
4. 集成流式 TRA (`tra_forward_stream`)
5. 修改 Encoder/Decoder 使用流式处理

🎯 **结果**:
- 真正的帧级流式处理
- 正确的状态缓存和传递
- 保证时间连续性和因果性
- 低延迟实时降噪

📝 **注意事项**:
- 需要导出 PyTorch 模型权重才能获得实际降噪效果
- 当前使用随机初始化权重仅用于测试流程
- 建议使用 `-O2` 或 `-O3` 优化编译以获得最佳性能
