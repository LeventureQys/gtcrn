# GTCRN_C 项目技术文档

## 项目概述

GTCRN_C 是 GTCRN（Grouped Temporal Convolutional Recurrent Network）语音增强模型的纯 C 语言实现版本。这个项目从零开始，将原本基于 PyTorch 的深度学习模型完整移植到了 C 语言环境，实现了从模型权重导出、网络前向传播、到流式推理的完整链路。整个项目耗时约一个半月，涉及深度学习模型理解、C 语言底层实现、数值计算优化、流式处理架构设计等多个技术领域。

## 项目背景与目标

### 为什么需要 C 语言实现？

虽然 PyTorch 等深度学习框架提供了便捷的训练和推理接口，但在实际部署场景中，特别是嵌入式设备、实时音频处理、以及需要严格控制内存和延迟的应用中，C 语言实现具有不可替代的优势：

1. **零依赖部署**：不需要 Python 运行时和庞大的深度学习框架库，只需要标准 C 库和数学库
2. **极致性能**：直接操作内存，无框架开销，可以针对特定硬件进行优化
3. **内存可控**：精确控制内存分配和释放，适合资源受限环境
4. **跨平台兼容**：C 代码可以在几乎所有平台上编译运行，包括嵌入式 ARM、DSP 等
5. **实时性保证**：流式处理架构设计，支持逐帧推理，延迟可控制在 16ms 以内

### 项目目标

- 完整实现 GTCRN 网络的所有组件，确保与 PyTorch 版本数值一致
- 支持离线批处理和实时流式两种推理模式
- 代码结构清晰，易于维护和扩展
- 提供完整的调试工具链，便于验证实现正确性

## 网络架构深度解析

### GTCRN 整体结构

GTCRN 是一个轻量级的语音降噪模型，参数量仅 48.2K，采用 U-Net 风格的编码器-解码器架构。整个网络处理流程如下：

```
输入音频 (16kHz) 
  ↓ STFT (FFT=512, Hop=256)
复数频谱图 (257 频点 × T 帧)
  ↓ 特征提取 [mag, real, imag]
特征张量 (3, 257, T)
  ↓ ERB 压缩
ERB 特征 (3, 129, T)
  ↓ SFE (子带特征提取)
扩展特征 (9, 129, T)
  ↓ 编码器下采样
编码器特征 (16, 33, T)
  ↓ GTConvBlocks (膨胀卷积 + TRA)
编码器输出 (16, 33, T)
  ↓ DPGRNN (双路径分组 RNN)
增强特征 (16, 33, T)
  ↓ 解码器上采样 + GTConvBlocks
解码器输出 (2, 129, T)
  ↓ ERB 扩展
掩膜 (2, 257, T)
  ↓ 复数掩膜应用
增强频谱 (2, 257, T)
  ↓ ISTFT
输出音频 (16kHz)
```

### 核心模块详解

#### 1. ERB 滤波器组（Equivalent Rectangular Bandwidth）

ERB 滤波器组是 GTCRN 的一个重要创新点，用于在频率域进行智能压缩和扩展。

**设计动机**：人耳对频率的感知不是线性的，而是遵循 ERB 尺度。在语音增强任务中，高频信息往往包含更多噪声，而低频包含更多语音能量。通过 ERB 压缩，可以将 257 个频点压缩到 129 个，减少计算量，同时保持感知上重要的信息。

**实现细节**：
- **压缩阶段（ERB BM）**：前 65 个低频 bin 直接保留，后 192 个高频 bin 通过可学习的线性变换压缩到 64 个 ERB 频带
  - 权重矩阵：`erb_fc_weight[64 × 192]`，每个 ERB 频带是 192 个原始频点的加权组合
- **扩展阶段（ERB BS）**：解码器输出后，将 64 个 ERB 频带扩展回 192 个原始频点
  - 权重矩阵：`ierb_fc_weight[192 × 64]`，实现反向映射

**代码实现位置**：
- `src/gtcrn_forward.c`: `erb_bm()`, `erb_bs()` 函数
- `src/gtcrn_stream.c`: `erb_bm_stream()`, `erb_bs_stream()` 流式版本

#### 2. SFE（Subband Feature Extraction）

SFE 模块通过局部频率上下文提取子带特征，将输入通道数扩展 3 倍。

**工作原理**：对每个频率 bin，提取其左邻、中心、右邻三个位置的值，形成局部上下文窗口。这相当于在频率维度上使用 kernel=3、padding=1 的 unfold 操作。

**数学表达**：
```
对于输入 (B, C, T, F)，输出 (B, 3C, T, F)
output[b, 3c+0, t, f] = input[b, c, t, f-1]  (左邻，边界为0)
output[b, 3c+1, t, f] = input[b, c, t, f]    (中心)
output[b, 3c+2, t, f] = input[b, c, t, f+1]  (右邻，边界为0)
```

**实现细节**：
- 在编码器输入阶段，将 3 通道特征扩展为 9 通道
- 在 GTConvBlock 内部，将 8 通道特征扩展为 24 通道，为后续点卷积提供更丰富的特征表示

**代码实现位置**：
- `src/gtcrn_forward.c`: `sfe_forward()` 函数
- `src/gtcrn_stream.c`: `sfe_stream()` 流式版本

#### 3. GTConvBlock（Grouped Temporal Convolution Block）

GTConvBlock 是 GTCRN 的核心卷积模块，结合了分组卷积、膨胀卷积和时序注意力机制。

**模块结构**：
```
输入 (B, 16, T, F)
  ↓ 通道分割
x1 (B, 8, T, F)  |  x2 (B, 8, T, F)
  ↓ SFE          |  (保持不变)
x1_sfe (B, 24, T, F)
  ↓ Point Conv 1 + BN + PReLU
h1 (B, 16, T, F)
  ↓ 因果填充 + Depthwise Conv (膨胀卷积) + BN + PReLU
dc_out (B, 16, T, F)
  ↓ Point Conv 2 + BN
h1_out (B, 8, T, F)
  ↓ TRA 注意力
h1_att (B, 8, T, F)
  ↓ 通道混洗
输出 (B, 16, T, F) = [h1_att, x2]
```

**关键设计点**：

1. **分组卷积**：Point Conv 1 将 24 通道压缩到 16 通道，Point Conv 2 进一步压缩到 8 通道，减少参数量

2. **深度可分离卷积**：Depthwise Conv 使用 groups=16，每个通道独立进行 3×3 卷积，参数量从 16×16×3×3=2304 减少到 16×1×3×3=144

3. **膨胀卷积（Dilated Convolution）**：编码器使用膨胀率 [1, 2, 5]，解码器使用 [5, 2, 1]（反向顺序），扩大感受野而不增加参数量
   - 膨胀率 1：感受野 3 帧
   - 膨胀率 2：感受野 5 帧
   - 膨胀率 5：感受野 11 帧

4. **因果填充（Causal Padding）**：编码器在时间维度前填充 `(kernel_t - 1) * dilation` 帧，确保输出不依赖未来信息，这对流式处理至关重要

5. **通道混洗（Channel Shuffle）**：将处理后的 x1 和未处理的 x2 交错排列，保持信息流动

**代码实现位置**：
- `src/gtcrn_forward.c`: `gtconv_block_forward()` 函数
- `src/gtcrn_stream.c`: 流式版本，使用缓存机制处理历史帧

#### 4. TRA（Temporal Recurrent Attention）

TRA 是 GTCRN 的时序注意力机制，基于 GRU 实现，用于对时序特征进行自适应加权。

**工作原理**：
```
输入 (B, 8, T, F)
  ↓ 转置为 (B, T, 8, F)
  ↓ 重塑为 (B*T, F, 8)
  ↓ 转置为 (B*T, 8, F) -> 按频率 bin 处理
对每个频率 bin:
  ↓ GRU(input=8, hidden=16)
GRU 输出 (B*T, F, 16)
  ↓ 全连接层 (16 -> 8)
注意力权重 (B*T, F, 8)
  ↓ Sigmoid 激活
归一化权重 (B*T, F, 8)
  ↓ 广播乘法
加权特征 (B, 8, T, F)
```

**设计思想**：GRU 能够学习时序依赖关系，根据历史信息动态调整当前帧的权重。全连接层将 GRU 的 16 维隐藏状态映射回 8 维，与输入通道数匹配。

**实现细节**：
- GRU 在每个频率 bin 上独立运行，处理该频率的时序信息
- 使用双向处理：先按时间顺序处理，再应用注意力权重
- 注意力权重通过 Sigmoid 归一化到 [0, 1]，作为门控机制

**代码实现位置**：
- `src/gtcrn_forward.c`: `tra_forward()` 函数
- `src/gtcrn_stream.c`: `tra_gru_step()` 单步流式版本

#### 5. DPGRNN（Dual-Path Grouped Recurrent Neural Network）

DPGRNN 是 GTCRN 的核心特征增强模块，采用双路径设计，分别处理频率维度和时间维度的依赖关系。

**双路径架构**：

1. **帧内路径（Intra-frame Path）**：在频率维度上处理，使用双向分组 GRU
   - 输入：`(B*T, F, C)` - 将每个时间帧视为独立序列
   - 处理：对每个时间帧，在频率维度上使用双向 GRU 处理
   - 输出：`(B*T, F, C)` - 增强的频率域特征

2. **帧间路径（Inter-frame Path）**：在时间维度上处理，使用单向分组 GRU
   - 输入：`(B, T, F, C)` - 将每个频率 bin 视为独立序列
   - 处理：对每个频率 bin，在时间维度上使用单向 GRU 处理
   - 输出：`(B, T, F, C)` - 增强的时序特征

**分组机制**：
- 将 16 个通道分成两组，每组 8 个通道
- 每组使用独立的 GRU 处理，减少参数量和计算量
- 两个 GRU 的输出拼接后通过全连接层融合

**具体实现**：

**帧内双向 RNN**：
```
输入 (B*T, F, 16)
  ↓ 通道分割
前半部分 (B*T, F, 8) -> RNN1 (双向 GRU, hidden=4) -> [fwd, bwd] (B*T, F, 8)
后半部分 (B*T, F, 8) -> RNN2 (双向 GRU, hidden=4) -> [fwd, bwd] (B*T, F, 8)
  ↓ 拼接
(B*T, F, 16) = [RNN1_fwd, RNN1_bwd, RNN2_fwd, RNN2_bwd]
  ↓ 全连接层 (16 -> 16)
(B*T, F, 16)
  ↓ LayerNorm + 残差连接
输出 (B*T, F, 16)
```

**帧间单向 RNN**：
```
输入 (B, T, F, 16)
对每个频率 bin f:
  提取时间序列 (B, T, 16)
    ↓ 通道分割
  前半部分 (B, T, 8) -> RNN1 (单向 GRU, hidden=8) -> (B, T, 8)
  后半部分 (B, T, 8) -> RNN2 (单向 GRU, hidden=8) -> (B, T, 8)
    ↓ 拼接
  (B, T, 16) = [RNN1_out, RNN2_out]
    ↓ 全连接层 (16 -> 16)
  (B, T, 16)
    ↓ LayerNorm + 残差连接
  输出 (B, T, 16)
```

**为什么需要双路径？**

- **频率域依赖**：语音信号在频率域有很强的相关性，相邻频率 bin 往往包含相似的信息。帧内路径能够利用这种频率相关性，增强特征表示。
- **时序依赖**：语音信号是时序信号，当前帧的信息依赖于历史帧。帧间路径能够建模长时依赖关系，提高降噪效果。
- **互补性**：两个路径从不同维度处理信息，相互补充，形成更强大的特征表示。

**代码实现位置**：
- `src/gtcrn_forward.c`: `intra_grnn_forward()`, `inter_grnn_forward()`, `dpgrnn_forward()` 函数
- `src/gtcrn_stream.c`: `intra_grnn_bidirectional()` 流式版本，需要维护 RNN 隐藏状态

#### 6. 编码器-解码器架构

GTCRN 采用对称的编码器-解码器结构，通过跳跃连接保持细节信息。

**编码器**：
```
输入 (9, 129, T)
  ↓ Conv0: (9, 16, 1×5, stride=2) -> (16, 65, T)
  ↓ Conv1: (16, 16, 1×5, stride=2, groups=2) -> (16, 33, T)
  ↓ GTConvBlock (dilation=1, 2, 5)
编码器输出 (16, 33, T)
```

**解码器**：
```
输入 (16, 33, T)
  ↓ GTConvBlock (dilation=5, 2, 1) + 跳跃连接
  ↓ ConvTranspose3: (16, 16, 1×5, stride=2, groups=2) -> (16, 65, T)
  ↓ ConvTranspose4: (16, 2, 1×5, stride=2) -> (2, 129, T)
解码器输出 (2, 129, T)
```

**跳跃连接**：编码器的 Conv0 输出直接连接到解码器的对应层，保持细节信息不丢失。

**转置卷积**：解码器使用 ConvTranspose2d 实现上采样，通过 stride=2 将频率维度从 33 恢复到 65，再恢复到 129。

## 实现细节与技术难点

### 1. 数值精度保证

从 PyTorch 到 C 语言的移植，最大的挑战是保证数值一致性。PyTorch 使用 32 位浮点数，但具体的计算顺序、舍入方式可能与 C 实现不同。

**解决方案**：
- 严格按照 PyTorch 的计算顺序实现，包括循环嵌套顺序、矩阵乘法顺序等
- 使用相同的数值常量（如 BatchNorm 的 eps=1e-5）
- 实现完整的调试工具链，逐层对比输出

**调试工具**：
- `demo/debug_forward.c`: 完整前向传播调试
- `demo/debug_enconv0.c`: 编码器第一层调试
- `demo/debug_gtconv.c`: GTConvBlock 调试
- `demo/debug_tra.c`: TRA 模块调试
- `scripts/compare_c_pytorch.py`: Python 脚本对比 C 和 PyTorch 输出

### 2. 内存管理优化

C 语言实现需要手动管理内存，如何高效使用内存是一个重要问题。

**工作空间复用**：
- 前向传播过程中，大量使用临时缓冲区
- 通过精心设计缓冲区布局，实现内存复用，减少分配次数
- 例如，在 `gtcrn_forward_complete_with_workspace()` 中，使用 4 个大缓冲区循环复用

**缓冲区布局示例**（来自 `gtcrn_forward.c`）：
```c
size_t buf_size = 16 * time * freq_in;
gtcrn_float* buf1 = workspace;           // 特征张量、SFE 输出
gtcrn_float* buf2 = buf1 + buf_size;      // ERB 输出、编码器中间结果
gtcrn_float* buf3 = buf2 + buf_size;       // 编码器输出缓存（跳跃连接）
gtcrn_float* scratch = buf3 + buf_size;    // 临时工作空间
```

### 3. 流式处理架构

流式处理是实时应用的核心需求，需要维护网络状态，逐帧处理。

**状态管理**：
- **卷积缓存**：存储膨胀卷积所需的历史帧，每个 GTConvBlock 需要 `(kernel_t - 1) * dilation` 帧历史
- **RNN 隐藏状态**：TRA 和 DPGRNN 的 GRU 需要维护隐藏状态
- **ISTFT 重叠相加缓冲区**：STFT 的逆变换需要重叠相加，需要缓存历史帧

**流式处理流程**（来自 `gtcrn_stream.c`）：
```c
// 1. 更新卷积缓存（左移 1 帧，追加新帧）
memmove(cache, cache + frame_size, (cache_t - 1) * frame_size * sizeof(gtcrn_float));
memcpy(cache + (cache_t - 1) * frame_size, input, frame_size * sizeof(gtcrn_float));

// 2. 单帧卷积计算
conv2d_stream_frame(weight, bias, cache, input, output, ...);

// 3. 更新 RNN 隐藏状态
tra_gru_step(..., hidden_state, ...);
```

**延迟控制**：
- 帧长：256 样本（16ms @ 16kHz）
- 目标处理时间：< 16ms
- 实际性能：Intel i5-12400 约 0.8ms，RTF（实时因子）约 0.05

### 4. FFT/STFT 实现

STFT（短时傅里叶变换）是音频处理的基础，需要高效的 FFT 实现。

**实现方案**：
- 使用 Cooley-Tukey 基 2 FFT 算法
- 预计算旋转因子表和位反转表，避免运行时计算
- 支持复数 FFT 和实数 FFT（利用对称性优化）

**STFT 流程**：
```c
// 1. 窗函数应用（Hanning 窗）
apply_window(input, window, frame_size);

// 2. 实数 FFT（512 点 -> 257 频点）
gtcrn_rfft_forward(plan, windowed, real, imag);

// 3. ISTFT 逆变换
gtcrn_istft_forward(plan, real, imag, output, overlap_buffer);
```

### 5. 权重导出与加载

从 PyTorch 模型导出权重到 C 语言可用的二进制格式。

**导出流程**（`scripts/export_weights.py`）：
1. 加载 PyTorch 模型检查点
2. 遍历模型的所有层，提取权重和偏置
3. 按照 C 结构体的内存布局顺序写入二进制文件
4. 验证权重数量和大小

**加载流程**（`src/gtcrn_model.c`）：
1. 打开权重文件，读取二进制数据
2. 按照 `gtcrn_weights_s` 结构体布局解析
3. 验证文件大小和魔数（可选）

**权重文件格式**：
- 纯二进制格式，按行主序（row-major）存储
- 每个浮点数 4 字节（float32）
- 总大小约 192KB（48.2K 参数 × 4 字节）

## 代码组织与架构设计

### 目录结构

```
GTCRN_C/
├── include/              # 头文件
│   ├── gtcrn_types.h     # 类型定义、常量
│   ├── gtcrn_math.h      # 数学工具函数
│   ├── gtcrn_layers.h    # 神经网络层接口
│   ├── gtcrn_fft.h       # FFT/STFT 接口
│   └── gtcrn_model.h     # 主模型 API
├── src/                  # 源文件
│   ├── gtcrn_math.c      # 向量/矩阵运算
│   ├── gtcrn_layers.c    # 层实现（Conv2d, BN, GRU, PReLU）
│   ├── gtcrn_fft.c       # FFT/STFT 实现
│   ├── gtcrn_model.c     # 模型创建、权重加载
│   ├── gtcrn_forward.c   # 离线前向传播
│   └── gtcrn_stream.c    # 流式前向传播
├── demo/                 # 示例程序
│   ├── main.c            # 离线模式示例
│   ├── main_stream.c     # 流式模式示例
│   ├── wav_io.c/h        # WAV 文件 I/O
│   └── debug_*.c         # 调试工具
├── scripts/              # 工具脚本
│   ├── export_weights.py # 权重导出
│   └── compare_*.py       # 对比验证脚本
└── CMakeLists.txt        # 构建配置
```

### 核心数据结构

**模型结构体**（`gtcrn_t`）：
```c
typedef struct {
    gtcrn_weights_t* weights;      // 模型权重
    gtcrn_stream_state_t* state;   // 流式状态（可选）
    gtcrn_fft_plan_t* fft_plan;    // FFT 计划
    // ... 其他状态
} gtcrn_t;
```

**权重结构体**（`gtcrn_weights_s`）：
- 包含所有层的权重、偏置、BatchNorm 参数等
- 总大小约 192KB，所有参数平铺在连续内存中

**流式状态结构体**（`gtcrn_stream_state_s`）：
- 卷积缓存：`conv_cache[层数][通道数][缓存帧数][频率]`
- RNN 隐藏状态：`tra_h[层数][隐藏维度]`, `dpgrnn_h[路径][RNN索引][隐藏维度]`
- ISTFT 重叠缓冲区：`istft_overlap[频率]`

### API 设计

**离线模式**：
```c
gtcrn_t* model = gtcrn_create();
gtcrn_load_weights(model, "weights/gtcrn_weights.bin");
gtcrn_process(model, input_audio, input_len, output_audio, &output_len);
gtcrn_destroy(model);
```

**流式模式**：
```c
gtcrn_t* model = gtcrn_create();
gtcrn_load_weights(model, "weights/gtcrn_weights.bin");
gtcrn_reset_state(model);  // 重置状态

for (each_frame) {
    gtcrn_process_frame(model, input_frame, output_frame);
}
gtcrn_destroy(model);
```

## 开发过程与工作量

### 第一阶段：模型理解与架构设计（1 周）

- 深入阅读 GTCRN 论文和 PyTorch 源码
- 理解每个模块的作用和实现细节
- 设计 C 语言实现的整体架构
- 确定数据结构和 API 接口

### 第二阶段：基础组件实现（2 周）

- **数学库**（`gtcrn_math.c`）：向量运算、矩阵乘法、激活函数
- **神经网络层**（`gtcrn_layers.c`）：
  - Conv2d / ConvTranspose2d：支持分组、膨胀、填充
  - BatchNorm2d：推理模式，使用 running_mean/var
  - PReLU：参数化 ReLU 激活
  - GRU：完整的门控循环单元实现
  - LayerNorm：层归一化
- **FFT/STFT**（`gtcrn_fft.c`）：Cooley-Tukey FFT、STFT、ISTFT

### 第三阶段：网络模块实现（2 周）

- **ERB 滤波器组**：压缩和扩展实现
- **SFE**：子带特征提取
- **GTConvBlock**：包括点卷积、深度卷积、TRA 注意力
- **DPGRNN**：帧内双向 RNN 和帧间单向 RNN
- **完整前向传播**：整合所有模块，实现端到端推理

### 第四阶段：流式处理实现（1.5 周）

- **状态管理**：设计流式状态结构体
- **缓存机制**：实现卷积缓存和 RNN 状态更新
- **逐帧处理**：将批处理代码改造为流式处理
- **ISTFT 重叠相加**：实现音频重建

### 第五阶段：调试与验证（1 周）

- **调试工具开发**：实现逐层对比工具
- **数值验证**：与 PyTorch 输出逐层对比，修复精度问题
- **边界情况处理**：处理填充、边界、单帧等特殊情况
- **性能优化**：优化内存使用和计算效率

### 第六阶段：工具链与文档（0.5 周）

- **权重导出脚本**：Python 脚本从 PyTorch 导出权重
- **构建系统**：CMake 配置，支持 Windows/Linux/macOS
- **示例程序**：离线模式和流式模式示例
- **代码注释**：完整的中文注释，使用 VS 风格 XML 文档

## 技术亮点

### 1. 完整的数值一致性

通过逐层调试和对比，确保 C 实现与 PyTorch 版本的输出误差在可接受范围内（通常 < 1e-5）。

### 2. 高效的内存管理

通过工作空间复用，将内存占用控制在合理范围内。离线模式约需 4-8MB 工作空间，流式模式约需 100-200KB 状态空间。

### 3. 灵活的架构设计

支持单精度和双精度浮点（编译时选择），支持不同的 FFT 大小和帧长（通过宏定义配置）。

### 4. 完善的调试工具

提供多个调试程序，可以单独测试每个模块，快速定位问题。

### 5. 跨平台兼容

使用标准 C99，仅依赖标准库和数学库，可以在几乎所有平台上编译运行。

## 性能指标

### 计算复杂度

- **参数量**：48.2K（约 192KB）
- **FLOPs**：单帧（256 样本）约 2-3M FLOPs
- **内存占用**：
  - 离线模式：工作空间约 4-8MB（取决于音频长度）
  - 流式模式：状态约 100-200KB

### 实时性能

- **帧处理时间**（Intel i5-12400）：
  - 单帧：约 0.8ms
  - RTF（实时因子）：约 0.05（远小于 1.0，可实时处理）
- **延迟**：
  - 算法延迟：16ms（1 帧）
  - 总延迟：约 20-30ms（包括 I/O 和系统开销）

## 未来改进方向

1. **SIMD 优化**：使用 SSE/AVX 指令集加速矩阵运算
2. **定点量化**：实现 INT8 量化，进一步减少内存和计算量
3. **多线程并行**：利用多核 CPU 并行处理不同频率 bin
4. **GPU 加速**：使用 CUDA/OpenCL 实现 GPU 版本
5. **模型压缩**：知识蒸馏、剪枝等技术进一步压缩模型

## 总结

GTCRN_C 项目是一个完整的深度学习模型 C 语言移植案例，涉及从模型理解、架构设计、底层实现、到调试验证的全流程。通过一个半月的开发，成功实现了与 PyTorch 版本功能等价、性能优异的 C 语言实现，为语音增强模型在嵌入式设备和实时应用中的部署提供了可行的解决方案。

项目的成功不仅在于实现了功能，更在于建立了完整的工具链和调试方法，为后续类似项目的开发积累了宝贵经验。

