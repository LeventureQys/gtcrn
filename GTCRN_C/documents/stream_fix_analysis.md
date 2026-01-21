# GTCRN流式推理噪音问题分析与修复方案

## 问题描述
C语言流式版本推理产生大量噪音，而完整版本效果正常。

## 根本原因

### 1. LayerNorm方差计算错误（主要问题）

**位置**: `gtcrn_stream.c` 第850, 888, 949, 987行

**错误代码**:
```c
// DPGRNN1 Intra LayerNorm (第850行)
var = sqrtf(var / (33 * 16) + 1e-8f);  // ❌ 错误：直接开方
for (int i = 0; i < 33 * 16; i++) {
    rnn_in[i] = (fc_buf[i] - mean) / var * ...  // 除以sqrt(var/N)
}
```

**正确实现** (参考 `gtcrn_forward.c:641`):
```c
var /= total;
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
// 然后使用 inv_std
```

**影响**:
- 归一化尺度错误，导致数值范围异常
- 梯度传播受影响，特征分布偏移
- 累积误差在流式推理中逐帧放大

### 2. 方差计算公式错误

**当前实现**:
```c
for (int i = 0; i < 33 * 16; i++) {
    gtcrn_float diff = fc_buf[i] - mean;
    var += diff * diff;
}
var = sqrtf(var / (33 * 16) + 1e-8f);  // 计算 sqrt(sum(x^2)/N)
```

**应该是**:
```c
for (int i = 0; i < 33 * 16; i++) {
    gtcrn_float diff = fc_buf[i] - mean;
    var += diff * diff;
}
var /= (33 * 16);  // 先计算方差 var = sum(x^2)/N
gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);  // 再计算标准差的倒数
```

## 修复方案

### 修复1: DPGRNN1 Intra LayerNorm (第842-855行)

**替换**:
```c
/* LayerNorm over entire (33, 16) = 528 elements */
{
    gtcrn_float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
    mean /= (33 * 16);
    for (int i = 0; i < 33 * 16; i++) {
        gtcrn_float diff = fc_buf[i] - mean;
        var += diff * diff;
    }
    var = sqrtf(var / (33 * 16) + 1e-8f);  // ❌ 错误
    for (int i = 0; i < 33 * 16; i++) {
        rnn_in[i] = (fc_buf[i] - mean) / var *
            w->dp1_intra_ln_gamma[i] + w->dp1_intra_ln_beta[i];
    }
}
```

**修改为**:
```c
/* LayerNorm over entire (33, 16) = 528 elements */
{
    gtcrn_float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
    mean /= (33 * 16);
    for (int i = 0; i < 33 * 16; i++) {
        gtcrn_float diff = fc_buf[i] - mean;
        var += diff * diff;
    }
    var /= (33 * 16);  // ✅ 先计算方差
    gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);  // ✅ 再计算标准差倒数
    for (int i = 0; i < 33 * 16; i++) {
        rnn_in[i] = (fc_buf[i] - mean) * inv_std *
            w->dp1_intra_ln_gamma[i] + w->dp1_intra_ln_beta[i];
    }
}
```

### 修复2: DPGRNN1 Inter LayerNorm (第879-895行)

**替换**:
```c
/* LayerNorm over entire (33, 16) = 528 elements */
{
    gtcrn_float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
    mean /= (33 * 16);
    for (int i = 0; i < 33 * 16; i++) {
        gtcrn_float diff = fc_buf[i] - mean;
        var += diff * diff;
    }
    var = sqrtf(var / (33 * 16) + 1e-8f);  // ❌ 错误
    for (int f = 0; f < 33; f++) {
        for (int c = 0; c < 16; c++) {
            buf5[c * 33 + f] = (fc_buf[f * 16 + c] - mean) / var *
                w->dp1_inter_ln_gamma[f * 16 + c] + w->dp1_inter_ln_beta[f * 16 + c];
        }
    }
}
```

**修改为**:
```c
/* LayerNorm over entire (33, 16) = 528 elements */
{
    gtcrn_float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
    mean /= (33 * 16);
    for (int i = 0; i < 33 * 16; i++) {
        gtcrn_float diff = fc_buf[i] - mean;
        var += diff * diff;
    }
    var /= (33 * 16);  // ✅ 先计算方差
    gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);  // ✅ 再计算标准差倒数
    for (int f = 0; f < 33; f++) {
        for (int c = 0; c < 16; c++) {
            buf5[c * 33 + f] = (fc_buf[f * 16 + c] - mean) * inv_std *
                w->dp1_inter_ln_gamma[f * 16 + c] + w->dp1_inter_ln_beta[f * 16 + c];
        }
    }
}
```

### 修复3: DPGRNN2 Intra LayerNorm (第940-954行)

同样的修复模式。

### 修复4: DPGRNN2 Inter LayerNorm (第978-994行)

同样的修复模式。

## 验证方法

1. **数值验证**: 对比Python和C版本的中间层输出
2. **音频验证**: 处理测试音频，检查噪音是否消除
3. **一致性验证**: 确保流式版本与完整版本输出一致（误差 < 1e-5）

## 其他潜在问题

1. **缓存初始化**: 确保所有缓存在第一帧前初始化为零
2. **ConvTranspose2d padding**: 检查解码器转置卷积的padding处理是否与PyTorch一致
3. **数值精度**: 检查是否有其他地方使用float而非double导致精度损失

## 修复优先级

1. **高优先级**: LayerNorm方差计算（4处）- 立即修复
2. **中优先级**: 验证缓存初始化
3. **低优先级**: 检查其他数值计算细节
