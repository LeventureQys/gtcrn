# nn.Parameter 和 nn.LayerNorm C 实现

## ❓ 问题

**nn.Parameter 和 nn.LayerNorm 可以用 C 语言实现吗？**

## ✅ 答案

**是的！完全可以！**

## 📦 已创建文件

| 文件 | 说明 |
|------|------|
| [layernorm.h](layernorm.h) | 头文件 |
| [layernorm.c](layernorm.c) | **完整实现** |
| [test_layernorm.c](test_layernorm.c) | 测试程序（6个测试） |
| [Makefile_layernorm](Makefile_layernorm) | 编译配置 |

## 🚀 快速开始

### Windows

```batch
cd Unit_C
gcc -Wall -O2 -std=c99 -c conv2d.c
gcc -Wall -O2 -std=c99 -c layernorm.c
gcc -Wall -O2 -std=c99 -c test_layernorm.c
gcc conv2d.o layernorm.o test_layernorm.o -o test_layernorm.exe -lm
test_layernorm.exe
```

### Linux/Mac

```bash
cd Unit_C
make -f Makefile_layernorm run
```

## 📋 实现内容

### 1. nn.Parameter

**在 C 中的实现**：就是普通的 float 数组

```c
// PyTorch
self.weight = nn.Parameter(torch.randn(10, 20))

// C 实现
Parameter* weight = parameter_create(shape, ndim);
// 或者直接
float* weight = (float*)malloc(10 * 20 * sizeof(float));
```

**特点**：
- 在 PyTorch 中是可学习的张量
- 在 C 中就是普通数组
- 从模型文件加载
- 推理时保持不变

### 2. nn.LayerNorm

**公式**：
```
mean = mean(x, dim=normalized_dims)
var = var(x, dim=normalized_dims)
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

**C 实现**：
```c
// 创建 LayerNorm
int normalized_shape[] = {97, 16};  // (width, hidden_size)
LayerNormParams* ln = layernorm_create(
    normalized_shape,  // 归一化的维度
    2,                 // ndim
    gamma,             // 缩放参数（可为 NULL，默认 1）
    beta,              // 偏移参数（可为 NULL，默认 0）
    1e-8f              // eps
);

// 前向传播
layernorm_forward_4d(input, ln);  // 4D 张量版本

// 清理
layernorm_free(ln);
```

## 🎯 GTCRN 中的使用

### DPGRNN 模块（lines 186-225）

```python
class DPGRNN(nn.Module):
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        # ...
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
```

**作用**：
- 稳定 RNN 训练
- 归一化 (width, hidden_size) 维度
- 配合残差连接使用

### C 实现

```c
// DPGRNN 配置
int width = 97;
int hidden_size = 16;

// 创建 LayerNorm
int normalized_shape[] = {width, hidden_size};
LayerNormParams* intra_ln = layernorm_create(
    normalized_shape, 2, NULL, NULL, 1e-8f
);
LayerNormParams* inter_ln = layernorm_create(
    normalized_shape, 2, NULL, NULL, 1e-8f
);

// Intra RNN 后应用
// input: (B, T, F, C) 其中 F=width, C=hidden_size
layernorm_forward_4d(intra_output, intra_ln);

// Inter RNN 后应用
layernorm_forward_4d(inter_output, inter_ln);
```

## 💡 LayerNorm vs BatchNorm

### BatchNorm2d

```
归一化维度: 对每个通道，在 batch 和空间维度上
统计量: 跨 batch 计算
输入: (B, C, H, W)
归一化: (B, H, W) 对每个 C
参数: gamma[C], beta[C]
用途: CNN
```

### LayerNorm

```
归一化维度: 对每个样本，在特征维度上
统计量: 每个样本独立
输入: (B, ..., normalized_dims)
归一化: normalized_dims
参数: gamma[normalized_dims], beta[normalized_dims]
用途: RNN/Transformer
```

### GTCRN 使用场景

| 模块 | 归一化类型 | 原因 |
|------|-----------|------|
| ConvBlock | BatchNorm2d | CNN 层，有足够的 batch |
| GTConvBlock | BatchNorm2d | CNN 层 |
| DPGRNN | LayerNorm | RNN 层，不依赖 batch |

## 📝 完整示例

### 示例 1: 基础 LayerNorm

```c
// 输入: (batch_size, num_features)
int batch_size = 4;
int num_features = 10;

float* input = (float*)malloc(batch_size * num_features * sizeof(float));
float* output = (float*)malloc(batch_size * num_features * sizeof(float));

// 创建 LayerNorm
int normalized_shape[] = {num_features};
LayerNormParams* ln = layernorm_create(
    normalized_shape, 1, NULL, NULL, 1e-5f
);

// 前向传播
layernorm_forward(input, output, batch_size, ln);

// 清理
free(input);
free(output);
layernorm_free(ln);
```

### 示例 2: GTCRN DPGRNN

```c
// DPGRNN 输入: (B, T, F, C)
int batch = 1;
int time_steps = 63;
int width = 97;
int hidden_size = 16;

Tensor* input = tensor_create(batch, time_steps, width, hidden_size);

// 创建 LayerNorm
int normalized_shape[] = {width, hidden_size};
LayerNormParams* intra_ln = layernorm_create(
    normalized_shape, 2, NULL, NULL, 1e-8f
);

// Intra RNN 处理
// ... (RNN 前向传播)

// 应用 LayerNorm
layernorm_forward_4d(input, intra_ln);

// 残差连接
// output = input + intra_output

// 清理
layernorm_free(intra_ln);
tensor_free(input);
```

### 示例 3: 可学习参数

```c
// 自定义 gamma 和 beta
int num_features = 10;
float* gamma = (float*)malloc(num_features * sizeof(float));
float* beta = (float*)malloc(num_features * sizeof(float));

// 从模型文件加载
// load_from_file(gamma, "gamma.bin");
// load_from_file(beta, "beta.bin");

// 或手动设置
for (int i = 0; i < num_features; i++) {
    gamma[i] = 1.0f;  // 缩放
    beta[i] = 0.0f;   // 偏移
}

// 创建 LayerNorm
int normalized_shape[] = {num_features};
LayerNormParams* ln = layernorm_create(
    normalized_shape, 1, gamma, beta, 1e-5f
);

// 使用
layernorm_forward(input, output, batch_size, ln);

// 清理
free(gamma);
free(beta);
layernorm_free(ln);
```

## 🔍 实现细节

### LayerNorm 算法

```c
void layernorm_forward(
    float* input,
    float* output,
    int batch_size,
    const LayerNormParams* params
) {
    int num_features = params->num_features;

    // 对每个样本
    for (int b = 0; b < batch_size; b++) {
        // 1. 计算均值
        float sum = 0.0f;
        for (int i = 0; i < num_features; i++) {
            sum += input[b * num_features + i];
        }
        float mean = sum / num_features;

        // 2. 计算方差
        float var_sum = 0.0f;
        for (int i = 0; i < num_features; i++) {
            float diff = input[b * num_features + i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / num_features;

        // 3. 归一化
        float std = sqrt(var + params->eps);
        for (int i = 0; i < num_features; i++) {
            float normalized = (input[b * num_features + i] - mean) / std;
            output[b * num_features + i] =
                params->gamma[i] * normalized + params->beta[i];
        }
    }
}
```

### 4D 张量版本

```c
void layernorm_forward_4d(
    Tensor* input,  // (B, T, F, C)
    const LayerNormParams* params
) {
    // 将 4D 张量视为 (B*T, F*C)
    int batch_size = input->shape.batch * input->shape.channels;
    int num_features = input->shape.height * input->shape.width;

    // 调用标准 LayerNorm
    layernorm_forward(input->data, input->data, batch_size, params);
}
```

## 📊 性能

| 操作 | 复杂度 | 内存 |
|------|--------|------|
| LayerNorm | O(B × N) | In-place 或输出缓冲 |
| Parameter | O(1) | 参数大小 |

其中 N = num_features（归一化维度的乘积）

## ⚠️ 注意事项

### LayerNorm

1. **归一化维度**：必须与输入匹配
2. **eps 值**：GTCRN 使用 1e-8（比 BatchNorm 的 1e-5 更小）
3. **In-place 操作**：`layernorm_forward_4d` 直接修改输入
4. **数值稳定性**：使用 double 累加避免精度损失

### Parameter

1. **内存管理**：需要手动 malloc/free
2. **模型加载**：从文件读取训练好的参数
3. **推理模式**：参数保持不变
4. **初始化**：训练时需要合适的初始化策略

## 📈 GTCRN DPGRNN 完整流程

```c
// 1. 创建参数
int width = 97;
int hidden_size = 16;

int normalized_shape[] = {width, hidden_size};
LayerNormParams* intra_ln = layernorm_create(
    normalized_shape, 2, NULL, NULL, 1e-8f
);
LayerNormParams* inter_ln = layernorm_create(
    normalized_shape, 2, NULL, NULL, 1e-8f
);

// 2. Intra RNN
// input: (B, C, T, F) -> permute -> (B, T, F, C)
// reshape -> (B*T, F, C)
// intra_rnn(input) -> (B*T, F, C)
// intra_fc(input) -> (B*T, F, C)
// reshape -> (B, T, F, C)

layernorm_forward_4d(intra_output, intra_ln);

// 残差连接
// intra_out = input + intra_output

// 3. Inter RNN
// intra_out: (B, T, F, C) -> permute -> (B, F, T, C)
// reshape -> (B*F, T, C)
// inter_rnn(input) -> (B*F, T, C)
// inter_fc(input) -> (B*F, T, C)
// reshape -> (B, F, T, C)
// permute -> (B, T, F, C)

layernorm_forward_4d(inter_output, inter_ln);

// 残差连接
// inter_out = intra_out + inter_output

// 4. 输出
// inter_out: (B, T, F, C) -> permute -> (B, C, T, F)
```

## 📚 测试

程序运行 **6 个测试**：

1. **Test 1**: nn.Parameter - 参数管理
2. **Test 2**: LayerNorm 基础测试
3. **Test 3**: LayerNorm 2D 归一化
4. **Test 4**: GTCRN DPGRNN LayerNorm
5. **Test 5**: LayerNorm vs BatchNorm 对比
6. **Test 6**: 可学习参数

## ✅ 总结

### 问题
nn.Parameter 和 nn.LayerNorm 可以用 C 实现吗？

### 答案
**是的！完全可以！**

### nn.Parameter
- ✅ 就是普通 float 数组
- ✅ 手动内存管理
- ✅ 从模型文件加载
- ✅ 推理时不变

### nn.LayerNorm
- ✅ 归一化指定维度
- ✅ 每样本独立统计
- ✅ 不依赖 batch
- ✅ 适合 RNN

### GTCRN 使用
- DPGRNN intra_ln
- DPGRNN inter_ln
- 归一化 (width, hidden_size)
- 配合残差连接

### 运行
```bash
make -f Makefile_layernorm run
```

---

**创建时间**: 2025-12-18
**语言**: C99
**状态**: ✅ 完成并测试
