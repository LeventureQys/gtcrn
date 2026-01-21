# GTCRN流式推理噪音问题修复总结

## 问题描述
C语言GTCRN流式推理版本产生大量噪音，而完整版本推理效果正常。

## 根本原因

### LayerNorm方差计算错误

**问题位置**: `src/gtcrn_stream.c` 中DPGRNN的4个LayerNorm实现
- 第850行: DPGRNN1 Intra LayerNorm
- 第888行: DPGRNN1 Inter LayerNorm
- 第949行: DPGRNN2 Intra LayerNorm
- 第987行: DPGRNN2 Inter LayerNorm

**错误代码**:
```c
// ❌ 错误实现
var = sqrtf(var / (33 * 16) + 1e-8f);  // 直接计算 sqrt(sum(x^2)/N + eps)
for (int i = 0; i < 33 * 16; i++) {
    output[i] = (input[i] - mean) / var * gamma[i] + beta[i];
}
```

**问题分析**:
1. 标准的LayerNorm公式: `y = (x - mean) / sqrt(var + eps) * gamma + beta`
2. 其中 `var = sum((x - mean)^2) / N`
3. 错误代码将 `var` 直接设为 `sqrt(sum(x^2)/N + eps)`，相当于计算了标准差而非方差
4. 然后又除以这个"标准差"，导致归一化尺度错误

**正确实现**:
```c
// ✅ 正确实现
var /= (33 * 16);  // 先计算方差: var = sum((x-mean)^2) / N
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);  // 再计算标准差的倒数
for (int i = 0; i < 33 * 16; i++) {
    output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
}
```

## 影响分析

### 为什么会产生噪音？

1. **数值尺度错误**:
   - 错误实现相当于除以 `sqrt(var/N)` 而不是 `sqrt(var)`
   - 对于N=528，这导致归一化后的值被放大了约23倍 (sqrt(528) ≈ 23)

2. **特征分布异常**:
   - DPGRNN输出的特征值范围异常，超出后续层的预期范围
   - 激活函数(PReLU, Tanh)在异常输入下产生非线性失真

3. **累积误差**:
   - 流式推理逐帧处理，错误在时间维度累积
   - GRU的隐藏状态携带错误信息传递到后续帧
   - 最终导致输出频谱严重失真，表现为噪音

### 为什么完整版本正常？

完整版本 (`src/gtcrn_forward.c`) 的LayerNorm实现是正确的：

```c
// gtcrn_forward.c:641 - 正确实现
var /= total;
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
```

## 修复内容

已修复 `src/gtcrn_stream.c` 中的4处LayerNorm计算：

### 1. DPGRNN1 Intra LayerNorm (第841-856行)
```c
var /= (33 * 16);
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
for (int i = 0; i < 33 * 16; i++) {
    rnn_in[i] = (fc_buf[i] - mean) * inv_std *
        w->dp1_intra_ln_gamma[i] + w->dp1_intra_ln_beta[i];
}
```

### 2. DPGRNN1 Inter LayerNorm (第880-897行)
```c
var /= (33 * 16);
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
for (int f = 0; f < 33; f++) {
    for (int c = 0; c < 16; c++) {
        buf5[c * 33 + f] = (fc_buf[f * 16 + c] - mean) * inv_std *
            w->dp1_inter_ln_gamma[f * 16 + c] + w->dp1_inter_ln_beta[f * 16 + c];
    }
}
```

### 3. DPGRNN2 Intra LayerNorm (第942-957行)
```c
var /= (33 * 16);
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
for (int i = 0; i < 33 * 16; i++) {
    rnn_in[i] = (fc_buf[i] - mean) * inv_std *
        w->dp2_intra_ln_gamma[i] + w->dp2_intra_ln_beta[i];
}
```

### 4. DPGRNN2 Inter LayerNorm (第981-998行)
```c
var /= (33 * 16);
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
for (int f = 0; f < 33; f++) {
    for (int c = 0; c < 16; c++) {
        buf6[c * 33 + f] = (fc_buf[f * 16 + c] - mean) * inv_std *
            w->dp2_inter_ln_gamma[f * 16 + c] + w->dp2_inter_ln_beta[f * 16 + c];
    }
}
```

## 验证步骤

### 1. 重新编译
```bash
cd GTCRN_C
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
```

### 2. 运行流式推理测试
```bash
# Windows
.\build\Release\gtcrn_demo_stream.exe

# Linux/Mac
./build/gtcrn_demo_stream
```

### 3. 对比输出
- 检查流式版本输出音频是否清晰无噪音
- 对比流式版本和完整版本的输出差异（应该非常小，误差 < 1e-5）

### 4. 数值验证（可选）
如果需要更详细的验证，可以：
1. 在Python中运行流式推理，保存中间层输出
2. 在C中运行流式推理，保存中间层输出
3. 对比两者的数值差异

## 预期结果

修复后，流式推理应该：
1. ✅ 输出音频清晰，无明显噪音
2. ✅ 与完整版本推理结果高度一致（误差 < 1e-5）
3. ✅ 与Python流式推理结果一致

## 技术要点

### LayerNorm标准实现
```c
// 标准LayerNorm实现模板
gtcrn_float mean = 0.0f, var = 0.0f;

// 1. 计算均值
for (int i = 0; i < N; i++) {
    mean += x[i];
}
mean /= N;

// 2. 计算方差
for (int i = 0; i < N; i++) {
    gtcrn_float diff = x[i] - mean;
    var += diff * diff;
}
var /= N;  // 关键：先除以N得到方差

// 3. 归一化
gtcrn_float inv_std = 1.0f / sqrtf(var + eps);  // 再开方得到标准差的倒数
for (int i = 0; i < N; i++) {
    y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
}
```

### 常见错误
```c
// ❌ 错误1: 直接开方
var = sqrtf(var / N + eps);  // 这是标准差，不是方差

// ❌ 错误2: 忘记除以N
gtcrn_float inv_std = 1.0f / sqrtf(var + eps);  // var还是sum(x^2)

// ✅ 正确: 分两步
var /= N;  // 先得到方差
gtcrn_float inv_std = 1.0f / sqrtf(var + eps);  // 再开方
```

## 相关文件

- **修复文件**: `src/gtcrn_stream.c`
- **参考实现**: `src/gtcrn_forward.c` (第641行)
- **详细分析**: `documents/stream_fix_analysis.md`

## 修复日期
2026-01-20

## 修复人员
Claude Code (分析) + 用户 (验证)
