# 神经网络基础层 - 快速参考

## ❓ 问题

**nn.Linear, nn.Unfold, nn.PReLU 可以用 C 语言实现吗？**

## ✅ 答案

**是的！完全可以！**

## 📦 已创建文件

| 文件 | 说明 |
|------|------|
| [nn_layers.h](nn_layers.h) | 头文件 |
| [nn_layers.c](nn_layers.c) | **完整实现** |
| [test_nn_layers.c](test_nn_layers.c) | 测试程序 |
| [Makefile_nn_layers](Makefile_nn_layers) | 编译配置 |
| [NN_LAYERS_README.md](NN_LAYERS_README.md) | 详细文档 |

## 🚀 快速开始

### Windows

```batch
cd Unit_C
gcc -Wall -O2 -std=c99 -c conv2d.c
gcc -Wall -O2 -std=c99 -c nn_layers.c
gcc -Wall -O2 -std=c99 -c test_nn_layers.c
gcc conv2d.o nn_layers.o test_nn_layers.o -o test_nn_layers.exe -lm
test_nn_layers.exe
```

### Linux/Mac

```bash
cd Unit_C
make -f Makefile_nn_layers run
```

## 📋 已实现的层

| PyTorch | C 函数 | 用途 |
|---------|--------|------|
| `nn.Linear` | `linear_forward()` | 全连接层 |
| `nn.Unfold` | `unfold_reshape_4d()` | 展开操作 |
| `nn.PReLU` | `prelu_forward_v2()` | 参数化 ReLU |
| `nn.Sigmoid` | `sigmoid_forward()` | Sigmoid 激活 |

## 💡 快速使用

### 1. nn.Linear

```c
// 创建参数
LinearParams* linear = linear_create(
    32,      // in_features
    16,      // out_features
    weight,  // 权重
    bias,    // 偏置
    1        // use_bias
);

// 前向传播
linear_forward(input, output, batch_size, linear);

// 清理
linear_free(linear);
```

### 2. nn.Unfold

```c
// 设置参数
UnfoldParams params = {
    .kernel_h = 1, .kernel_w = 3,
    .stride_h = 1, .stride_w = 1,
    .padding_h = 0, .padding_w = 1,
    .dilation_h = 1, .dilation_w = 1
};

// 展开
unfold_reshape_4d(input, output, &params);
```

### 3. nn.PReLU

```c
// 创建参数
float weights[16] = {0.25f, ...};  // 每通道一个
PReLUParams* prelu = prelu_create(16, weights);

// 前向传播（in-place）
prelu_forward_v2(input, prelu);

// 清理
prelu_free(prelu);
```

### 4. nn.Sigmoid

```c
// 方式 1: 数组
sigmoid_forward(data, size);

// 方式 2: Tensor
sigmoid_forward_tensor(input);
```

## 🎯 GTCRN 使用场景

### SFE 模块（line 69）

```python
self.unfold = nn.Unfold(kernel_size=(1,3), stride=(1,1), padding=(0,1))
```

```c
UnfoldParams sfe = {
    .kernel_h = 1, .kernel_w = 3,
    .stride_h = 1, .stride_w = 1,
    .padding_h = 0, .padding_w = 1,
    .dilation_h = 1, .dilation_w = 1
};
unfold_reshape_4d(input, output, &sfe);
```

### TRA 模块（line 82）

```python
self.att_fc = nn.Linear(channels*2, channels)
```

```c
LinearParams* tra_fc = linear_create(
    channels * 2, channels, weight, bias, 1
);
linear_forward(input, output, batch_size, tra_fc);
```

### ConvBlock（line 102）

```python
self.act = nn.PReLU()
```

```c
PReLUParams* prelu = prelu_create(channels, weights);
prelu_forward_v2(output, prelu);
```

### TRA 注意力（line 83）

```python
self.att_act = nn.Sigmoid()
```

```c
sigmoid_forward_tensor(attention);
```

## 📊 公式速查

### Linear
```
y = x @ W^T + b
```

### Unfold
```
(B, C, H, W) → (B, C*kh*kw, L)
```

### PReLU
```
y = x           if x > 0
y = alpha * x   if x <= 0
```

### Sigmoid
```
y = 1 / (1 + exp(-x))
```

## 🔧 完整示例

### GTCRN SFE 模块

```c
// 输入: (1, 8, 63, 97)
Tensor* input = tensor_create(1, 8, 63, 97);

// Unfold: kernel_size=3
UnfoldParams sfe_params = {
    .kernel_h = 1, .kernel_w = 3,
    .stride_h = 1, .stride_w = 1,
    .padding_h = 0, .padding_w = 1,
    .dilation_h = 1, .dilation_w = 1
};

// 输出: (1, 24, 63, 97)  // 8*3=24
Tensor* output = tensor_create(1, 24, 63, 97);

unfold_reshape_4d(input, output, &sfe_params);
```

### GTCRN TRA 模块（部分）

```c
// GRU 输出: (batch, time_steps, channels*2)
// Linear: channels*2 → channels

LinearParams* tra_linear = linear_create(
    channels * 2,  // 32
    channels,      // 16
    weight, bias, 1
);

// 前向传播
linear_forward(
    gru_output,     // (batch*time_steps, 32)
    linear_output,  // (batch*time_steps, 16)
    batch * time_steps,
    tra_linear
);

// Sigmoid
sigmoid_forward(linear_output, batch * time_steps * channels);

// 现在 linear_output 是注意力权重，范围 (0, 1)
```

### GTCRN ConvBlock

```c
// Conv2d + BatchNorm2d + PReLU

// 1. Conv2d
conv2d_forward(input, output, &conv_params);

// 2. BatchNorm2d
batchnorm2d_forward(output, bn_params);

// 3. PReLU
float prelu_weights[16];
for (int i = 0; i < 16; i++) {
    prelu_weights[i] = 0.25f;
}
PReLUParams* prelu = prelu_create(16, prelu_weights);
prelu_forward_v2(output, prelu);
```

## 📈 性能

| 操作 | 复杂度 | 内存 |
|------|--------|------|
| Linear | O(B×I×O) | 输出缓冲 |
| Unfold | O(B×C×H×W×k²) | 输出×k |
| PReLU | O(B×C×H×W) | In-place |
| Sigmoid | O(size) | In-place |

## ⚠️ 注意事项

### Linear
- 权重格式: (out_features, in_features)
- 支持任意批次大小

### Unfold
- GTCRN 使用特殊版本: `unfold_reshape_4d()`
- 保持空间维度不变
- 扩展通道维度

### PReLU
- In-place 操作
- 每个通道一个参数
- 默认 alpha = 0.25

### Sigmoid
- 注意数值稳定性
- 大负数可能溢出

## 📚 详细文档

- **完整文档**: [NN_LAYERS_README.md](NN_LAYERS_README.md)
- **实现代码**: [nn_layers.c](nn_layers.c)
- **测试代码**: [test_nn_layers.c](test_nn_layers.c)

## ✅ 总结

### 问题
nn.Linear, nn.Unfold, nn.PReLU 可以用 C 实现吗？

### 答案
**是的！完全可以！**

### 已实现
- ✅ nn.Linear
- ✅ nn.Unfold
- ✅ nn.PReLU
- ✅ nn.Sigmoid
- ✅ nn.Tanh

### 特点
- 🚀 纯 C99
- 🎯 高效实现
- 📦 易于集成
- ✅ 完整测试

### 运行
```bash
make -f Makefile_nn_layers run
```

---

**创建时间**: 2025-12-18
**状态**: ✅ 完成
