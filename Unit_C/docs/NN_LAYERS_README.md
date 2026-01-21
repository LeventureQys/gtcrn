# 神经网络基础层 C 实现

## 📋 问题

**nn.Linear, nn.Unfold, nn.PReLU 可以用 C 语言实现吗？**

## ✅ 答案

**是的！完全可以！而且实现简单高效！**

## 📦 已实现的层

| PyTorch 层 | C 函数 | 用途 |
|-----------|--------|------|
| `nn.Linear` | `linear_forward()` | 全连接层 |
| `nn.Unfold` | `unfold_forward()` | 展开操作（im2col） |
| `nn.PReLU` | `prelu_forward_v2()` | 参数化 ReLU |
| `nn.Sigmoid` | `sigmoid_forward()` | Sigmoid 激活 |
| `nn.Tanh` | `tanh_forward()` | Tanh 激活 |

## 🎯 GTCRN 中的使用

### 1. nn.Unfold - SFE 模块

```python
# gtcrn1.py lines 64-74
class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.unfold = nn.Unfold(
            kernel_size=(1,kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size-1)//2)
        )
```

**作用**: 提取子带特征，将频率邻域展开为通道

### 2. nn.Linear - TRA 模块

```python
# gtcrn1.py lines 77-93
class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)  # ← Linear
        self.att_act = nn.Sigmoid()
```

**作用**: 注意力机制中的全连接层

### 3. nn.PReLU - 激活函数

```python
# gtcrn1.py lines 102, 119, 125
self.act = nn.PReLU()  # ConvBlock
self.point_act = nn.PReLU()  # GTConvBlock
self.depth_act = nn.PReLU()  # GTConvBlock
```

**作用**: 所有卷积块的激活函数

### 4. nn.Sigmoid - 注意力权重

```python
# gtcrn1.py line 83
self.att_act = nn.Sigmoid()  # TRA
```

**作用**: 生成 (0,1) 范围的注意力权重

## 💻 C 实现

### 文件结构

```
Unit_C/
├── nn_layers.h              ← 头文件
├── nn_layers.c              ← 实现
├── test_nn_layers.c         ← 测试
└── Makefile_nn_layers       ← 编译配置
```

### 1. nn.Linear

#### 公式
```
y = x @ W^T + b
```

#### C 实现

```c
// 创建 Linear 参数
LinearParams* linear_params = linear_create(
    in_features,    // 输入特征数
    out_features,   // 输出特征数
    weight,         // 权重 (out_features, in_features)
    bias,           // 偏置 (out_features)
    use_bias        // 是否使用偏置
);

// 前向传播
linear_forward(
    input,          // 输入 (batch_size, in_features)
    output,         // 输出 (batch_size, out_features)
    batch_size,     // 批次大小
    linear_params   // 参数
);

// 清理
linear_free(linear_params);
```

#### 示例

```c
// TRA 模块: Linear(channels*2, channels)
int in_features = 32;   // channels * 2
int out_features = 16;  // channels
int batch_size = 63;    // time_steps

LinearParams* linear = linear_create(
    in_features, out_features, weight, bias, 1
);

linear_forward(input, output, batch_size, linear);
```

### 2. nn.Unfold

#### 公式
```
将 (B, C, H, W) 展开为 (B, C*kh*kw, L)
其中 L = output_h * output_w
```

#### C 实现

```c
// 设置 Unfold 参数
UnfoldParams unfold_params = {
    .kernel_h = 1,
    .kernel_w = 3,
    .stride_h = 1,
    .stride_w = 1,
    .padding_h = 0,
    .padding_w = 1,
    .dilation_h = 1,
    .dilation_w = 1
};

// 展开并 reshape 为 4D（GTCRN SFE 使用方式）
unfold_reshape_4d(
    input,          // (B, C, T, F)
    output,         // (B, C*kernel_size, T, F)
    &unfold_params
);
```

#### 示例

```c
// SFE 模块: Unfold(kernel_size=(1,3), stride=(1,1), padding=(0,1))
Tensor* input = tensor_create(1, 8, 63, 97);   // (B, C, T, F)
Tensor* output = tensor_create(1, 24, 63, 97); // (B, C*3, T, F)

UnfoldParams params = {
    .kernel_h = 1, .kernel_w = 3,
    .stride_h = 1, .stride_w = 1,
    .padding_h = 0, .padding_w = 1,
    .dilation_h = 1, .dilation_w = 1
};

unfold_reshape_4d(input, output, &params);
```

### 3. nn.PReLU

#### 公式
```
y = x           if x > 0
y = alpha * x   if x <= 0
```

#### C 实现

```c
// 创建 PReLU 参数
float prelu_weights[16];
for (int i = 0; i < 16; i++) {
    prelu_weights[i] = 0.25f;  // PyTorch 默认值
}

PReLUParams* prelu = prelu_create(
    num_channels,   // 参数数量（通常等于通道数）
    prelu_weights   // 每个通道的负斜率
);

// 前向传播（in-place）
prelu_forward_v2(
    input,          // (B, C, H, W)
    prelu
);

// 清理
prelu_free(prelu);
```

#### 示例

```c
// ConvBlock: PReLU()
Tensor* input = tensor_create(1, 16, 63, 97);

float weights[16];
for (int i = 0; i < 16; i++) {
    weights[i] = 0.25f;
}

PReLUParams* prelu = prelu_create(16, weights);
prelu_forward_v2(input, prelu);
```

### 4. nn.Sigmoid

#### 公式
```
y = 1 / (1 + exp(-x))
```

#### C 实现

```c
// 方式 1: 直接操作数组
sigmoid_forward(
    data,           // 数据指针
    size            // 数据大小
);

// 方式 2: 操作 Tensor
sigmoid_forward_tensor(
    input           // (B, C, H, W)
);
```

#### 示例

```c
// TRA 模块: Sigmoid()
Tensor* attention = tensor_create(1, 16, 63, 1);

// 计算注意力权重
sigmoid_forward_tensor(attention);

// 现在 attention 的值在 (0, 1) 范围内
```

## 🚀 编译和运行

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
make -f Makefile_nn_layers
./test_nn_layers
```

## 📊 测试输出

程序运行 6 个测试：

### Test 1: nn.Linear
测试全连接层的矩阵乘法

### Test 2: nn.Unfold
测试展开操作（im2col）

### Test 3: nn.PReLU
测试参数化 ReLU 激活

### Test 4: nn.Sigmoid
测试 Sigmoid 激活函数

### Test 5: GTCRN SFE 模块
完整的 SFE 模块实现

### Test 6: GTCRN TRA 注意力
TRA 模块的 Linear + Sigmoid 部分

## 🔍 实现细节

### nn.Linear 实现

```c
void linear_forward(
    const float* input,
    float* output,
    int batch_size,
    const LinearParams* params
) {
    int in_features = params->in_features;
    int out_features = params->out_features;

    // 对每个批次样本
    for (int b = 0; b < batch_size; b++) {
        // 对每个输出特征
        for (int o = 0; o < out_features; o++) {
            float sum = 0.0f;

            // 矩阵乘法
            for (int i = 0; i < in_features; i++) {
                sum += input[b * in_features + i] *
                       params->weight[o * in_features + i];
            }

            // 加偏置
            if (params->use_bias) {
                sum += params->bias[o];
            }

            output[b * out_features + o] = sum;
        }
    }
}
```

### nn.Unfold 实现（GTCRN 特化版本）

```c
void unfold_reshape_4d(
    const Tensor* input,
    Tensor* output,
    const UnfoldParams* params
) {
    // 对每个位置
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            // 对每个卷积核位置
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    // 计算输入位置（考虑 padding）
                    int ih = h * stride_h - padding_h + kh * dilation_h;
                    int iw = w * stride_w - padding_w + kw * dilation_w;

                    // 读取值（边界外为 0）
                    float val = 0.0f;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        val = input->data[...];
                    }

                    // 写入输出（展开为新通道）
                    int out_c = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    output->data[...] = val;
                }
            }
        }
    }
}
```

### nn.PReLU 实现

```c
void prelu_forward_v2(
    Tensor* input,
    const PReLUParams* params
) {
    // 对每个通道
    for (int c = 0; c < channels; c++) {
        float alpha = params->weight[c];

        // 对每个空间位置
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = ((b * channels + c) * height + h) * width + w;

                // PReLU
                if (input->data[idx] < 0) {
                    input->data[idx] *= alpha;
                }
            }
        }
    }
}
```

## 📈 性能特点

### nn.Linear
- **复杂度**: O(batch_size × in_features × out_features)
- **优化**: 可使用 BLAS 库加速矩阵乘法

### nn.Unfold
- **复杂度**: O(B × C × H × W × kh × kw)
- **内存**: 输出大小 = 输入大小 × kernel_size
- **优化**: 可并行化

### nn.PReLU
- **复杂度**: O(B × C × H × W)
- **内存**: In-place 操作，无额外内存
- **优化**: 可 SIMD 向量化

### nn.Sigmoid
- **复杂度**: O(size)
- **内存**: In-place 操作
- **优化**: 可使用查找表或快速近似

## 🎓 GTCRN 模块对应

### SFE (Subband Feature Extraction)

```c
// gtcrn1.py lines 64-74
// Input: (B, C, T, F)
// Output: (B, C*3, T, F)

UnfoldParams sfe_params = {
    .kernel_h = 1, .kernel_w = 3,
    .stride_h = 1, .stride_w = 1,
    .padding_h = 0, .padding_w = 1,
    .dilation_h = 1, .dilation_w = 1
};

unfold_reshape_4d(input, output, &sfe_params);
```

### TRA (Temporal Recurrent Attention)

```c
// gtcrn1.py lines 77-93
// 1. GRU (需要单独实现)
// 2. Linear
LinearParams* tra_linear = linear_create(
    channels * 2, channels, weight, bias, 1
);
linear_forward(gru_output, linear_output, batch * time_steps, tra_linear);

// 3. Sigmoid
sigmoid_forward(linear_output, batch * time_steps * channels);

// 4. 应用注意力
// output = input * attention_weights
```

### ConvBlock

```c
// gtcrn1.py lines 96-104
// Conv2d + BatchNorm2d + PReLU

// 1. Conv2d (或 ConvTranspose2d)
conv2d_forward(input, output, &conv_params);

// 2. BatchNorm2d
batchnorm2d_forward(output, bn_params);

// 3. PReLU
prelu_forward_v2(output, prelu_params);
```

### GTConvBlock

```c
// gtcrn1.py lines 107-153
// SFE + Point Conv + Depth Conv + Point Conv + TRA

// 1. SFE
unfold_reshape_4d(x1, sfe_output, &sfe_params);

// 2. Point Conv + BN + PReLU
conv2d_forward(sfe_output, h1, &point_conv1_params);
batchnorm2d_forward(h1, bn1_params);
prelu_forward_v2(h1, prelu_params);

// 3. Depth Conv + BN + PReLU
conv2d_forward(h1, h2, &depth_conv_params);
batchnorm2d_forward(h2, bn2_params);
prelu_forward_v2(h2, prelu_params);

// 4. Point Conv + BN
conv2d_forward(h2, h3, &point_conv2_params);
batchnorm2d_forward(h3, bn3_params);

// 5. TRA (需要 GRU + Linear + Sigmoid)
// ...
```

## ⚠️ 注意事项

### 内存管理
- Linear: 需要分配输出缓冲区
- Unfold: 输出大小 = 输入大小 × kernel_size
- PReLU: In-place 操作
- Sigmoid: In-place 操作

### 数值稳定性
- Sigmoid: 对于大的负数，exp(-x) 可能溢出
  - 解决: 使用 `1 / (1 + exp(-x))` 或查找表
- Linear: 权重初始化很重要
  - 建议: Xavier 或 He 初始化

### 性能优化
- Linear: 使用 BLAS 库（如 OpenBLAS）
- Unfold: 并行化外层循环
- PReLU: SIMD 向量化
- Sigmoid: 查找表或多项式近似

## 📚 相关文件

### 实现文件
- [nn_layers.h](nn_layers.h) - 头文件
- [nn_layers.c](nn_layers.c) - 实现
- [conv2d.h](conv2d.h) - Conv2d 头文件（依赖）
- [conv2d.c](conv2d.c) - Conv2d 实现（依赖）

### 测试文件
- [test_nn_layers.c](test_nn_layers.c) - 完整测试

### 构建文件
- [Makefile_nn_layers](Makefile_nn_layers) - 编译配置

## ✅ 总结

### 问题
nn.Linear, nn.Unfold, nn.PReLU 可以用 C 语言实现吗？

### 答案
**是的！完全可以！**

### 已实现
- ✅ nn.Linear - 全连接层
- ✅ nn.Unfold - 展开操作
- ✅ nn.PReLU - 参数化 ReLU
- ✅ nn.Sigmoid - Sigmoid 激活
- ✅ nn.Tanh - Tanh 激活

### 特点
- ✅ 纯 C99 实现
- ✅ 无外部依赖（仅 math.h）
- ✅ 高效实现
- ✅ 易于集成
- ✅ 完整测试

### GTCRN 应用
- SFE 模块: Unfold
- TRA 模块: Linear + Sigmoid
- ConvBlock: PReLU
- GTConvBlock: Unfold + PReLU

### 使用
```bash
make -f Makefile_nn_layers run
```

---

**创建时间**: 2025-12-18
**语言**: C99
**平台**: 跨平台
**状态**: ✅ 完成并测试
