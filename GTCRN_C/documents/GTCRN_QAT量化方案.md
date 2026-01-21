# GTCRN 量化感知训练（QAT）方案文档

## 概述

本文档详细描述了 GTCRN 模型的量化感知训练（Quantization-Aware Training, QAT）方案。QAT 是一种在训练过程中模拟量化效果的技术，能够显著提升模型在低精度（如 INT8）部署时的性能，相比后训练量化（PTQ）具有更好的精度保持能力。

## 背景与动机

### 为什么需要量化？

GTCRN 模型虽然参数量仅 48.2K，但在实际部署中仍面临以下挑战：

1. **内存限制**：嵌入式设备内存有限，FP32 模型需要约 192KB 存储
2. **计算效率**：FP32 运算在低端 CPU/DSP 上较慢，影响实时性
3. **功耗问题**：浮点运算功耗较高，不利于电池供电设备

通过量化到 INT8，可以：
- **内存减少 75%**：从 192KB 降至 48KB
- **计算加速 2-4 倍**：整数运算比浮点运算快
- **功耗降低**：整数运算单元功耗更低

### QAT vs PTQ

**后训练量化（PTQ）**：
- 优点：无需重新训练，实现简单
- 缺点：精度损失较大，特别是对于小模型（如 GTCRN）

**量化感知训练（QAT）**：
- 优点：精度损失小，模型在训练过程中适应量化
- 缺点：需要重新训练，训练时间较长

对于 GTCRN 这种轻量级模型，QAT 是更好的选择，能够在保持模型大小的同时，显著提升量化后的性能。

## 量化方案设计

### 量化策略

#### 1. 对称量化（Symmetric Quantization）

对于权重和激活值，采用对称量化：

```
量化：q = round(clamp(x / scale, -127, 127))
反量化：x' = q * scale

其中：
  scale = max(|x|) / 127
  clamp 将值限制在 [-127, 127] 范围内
```

**优点**：
- 实现简单，零点是固定的（0）
- 适合权重分布对称的情况

**缺点**：
- 对于非对称分布，可能浪费量化范围

#### 2. 非对称量化（Asymmetric Quantization）

对于某些激活值（如 ReLU 输出），采用非对称量化：

```
量化：q = round(clamp((x - zero_point) / scale, 0, 255))
反量化：x' = q * scale + zero_point

其中：
  scale = (max(x) - min(x)) / 255
  zero_point = round(-min(x) / scale)
```

**优点**：
- 充分利用量化范围
- 适合非对称分布（如 ReLU 输出总是 ≥ 0）

**缺点**：
- 实现稍复杂，需要存储 zero_point

#### 3. 混合精度策略

根据层的重要性，采用不同的量化精度：

- **关键层**：INT8 量化（权重和激活）
  - ERB 滤波器组（影响频率表示）
  - DPGRNN（核心特征增强）
  - 最后一层输出（直接影响掩膜质量）

- **次要层**：INT8 权重 + FP16 激活
  - 编码器/解码器的中间层
  - 减少激活值量化带来的误差累积

- **特殊层**：保持 FP32
  - BatchNorm 的 running_mean/var（统计量）
  - LayerNorm 的归一化计算（数值稳定性）

### 量化粒度

#### 1. 每层量化（Per-Layer）

每层使用一个 scale 和 zero_point。

**优点**：
- 实现简单，硬件友好
- 适合大多数层

**缺点**：
- 对于权重分布差异大的层，可能不够精细

#### 2. 每通道量化（Per-Channel）

卷积层权重按输出通道分别量化。

**优点**：
- 更精细，精度损失小
- 适合卷积层（不同通道权重分布可能差异大）

**缺点**：
- 需要存储每个通道的 scale，增加少量开销
- 实现稍复杂

**GTCRN 采用策略**：
- **卷积层权重**：每通道量化（Per-Channel）
- **全连接层权重**：每层量化（Per-Layer）
- **激活值**：每层量化（Per-Layer）

### 伪量化节点（Fake Quantization）

在训练过程中插入伪量化节点，模拟量化效果：

```python
class FakeQuantize(nn.Module):
    def __init__(self, scale, zero_point=0, symmetric=True):
        self.scale = scale
        self.zero_point = zero_point
        self.symmetric = symmetric
        
    def forward(self, x):
        if self.training:
            # 训练时：量化 -> 反量化，保持梯度流
            if self.symmetric:
                q = torch.round(torch.clamp(x / self.scale, -127, 127))
                return q * self.scale
            else:
                q = torch.round(torch.clamp((x - self.zero_point) / self.scale, 0, 255))
                return q * self.scale + self.zero_point
        else:
            # 推理时：直接量化
            return self.quantize(x)
```

**关键点**：
- 训练时：量化 → 反量化，保持梯度可导
- 推理时：直接量化，减少计算

## 实现方案

### 1. PyTorch QAT 实现

#### 模型改造

在原始 GTCRN 模型基础上，插入量化节点：

```python
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert

class GTCRN_QAT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # 量化入口和出口
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        # 输入量化
        x = self.quant(x)
        
        # 前向传播（内部包含伪量化节点）
        x = self.base_model(x)
        
        # 输出反量化（用于损失计算）
        x = self.dequant(x)
        return x
```

#### 量化配置

```python
# 为不同层配置量化方案
qconfig_dict = {
    # 默认配置：INT8 权重和激活
    '': torch.quantization.get_default_qat_qconfig('fbgemm'),
    
    # ERB 层：使用更精细的量化
    'erb_bm': torch.quantization.QConfig(
        activation=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=255, dtype=torch.quint8
        ),
        weight=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.PerChannelMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8
        )
    ),
    
    # DPGRNN：保持较高精度
    'dpgrnn': torch.quantization.QConfig(
        activation=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=0, quant_max=255, dtype=torch.quint8
        ),
        weight=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.PerChannelMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8
        )
    ),
    
    # 输出层：使用对称量化
    'output': torch.quantization.QConfig(
        activation=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8
        ),
        weight=torch.quantization.FakeQuantize.with_args(
            observer=torch.quantization.PerChannelMinMaxObserver,
            quant_min=-128, quant_max=127, dtype=torch.qint8
        )
    )
}

# 应用量化配置
model.qconfig = qconfig_dict
model = prepare_qat(model)
```

#### 训练流程

```python
# 1. 加载预训练模型
base_model = GTCRN()
base_model.load_state_dict(torch.load('checkpoints/gtcrn_fp32.pth'))

# 2. 创建 QAT 模型
qat_model = GTCRN_QAT(base_model)
qat_model.train()

# 3. 准备 QAT
qat_model = prepare_qat(qat_model)

# 4. 训练循环
optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-4)
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 前向传播（包含伪量化）
        output = qat_model(input)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 更新量化参数（observer）
        qat_model.apply(torch.quantization.enable_observer)
        qat_model.apply(torch.quantization.enable_fake_quant)

# 5. 转换为量化模型
qat_model.eval()
quantized_model = convert(qat_model)
```

### 2. C 语言量化推理实现

#### 量化权重格式

在 C 语言实现中，需要存储量化后的权重和 scale/zero_point：

```c
// 量化权重结构
typedef struct {
    int8_t* weight_int8;        // INT8 权重
    float scale;                // 量化 scale
    int8_t zero_point;          // 量化零点（对称量化时为 0）
    int per_channel;            // 是否每通道量化
    float* scales_per_channel;  // 每通道 scale（如果 per_channel=1）
} quantized_weight_t;

// 量化激活值结构
typedef struct {
    float scale;
    int8_t zero_point;
    float input_scale;          // 输入 scale（用于融合）
} quantized_activation_t;
```

#### 量化卷积实现

```c
// INT8 量化卷积
void quantized_conv2d_forward(
    const quantized_weight_t* weight,
    const quantized_activation_t* input_act,
    quantized_activation_t* output_act,
    const int8_t* input_int8,
    int8_t* output_int8,
    int batch, int in_h, int in_w, int out_h, int out_w,
    int in_ch, int out_ch, int kH, int kW,
    int sH, int sW, int pH, int pW,
    int groups) {
    
    // 计算输出 scale
    float output_scale = input_act->scale * weight->scale;
    
    // 临时缓冲区（INT32 累加）
    int32_t* acc = (int32_t*)malloc(out_ch * out_h * out_w * sizeof(int32_t));
    
    // INT8 卷积计算（累加到 INT32）
    for (int oc = 0; oc < out_ch; oc++) {
        float w_scale = weight->per_channel ? 
            weight->scales_per_channel[oc] : weight->scale;
            
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int32_t sum = 0;
                
                // 卷积计算
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * sH - pH + kh;
                            int iw = ow * sW - pW + kw;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int8_t x = input_int8[ic * in_h * in_w + ih * in_w + iw];
                                int8_t w = weight->weight_int8[oc * in_ch * kH * kW + 
                                                               ic * kH * kW + kh * kW + kw];
                                sum += (int32_t)x * (int32_t)w;
                            }
                        }
                    }
                }
                
                acc[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
    
    // 量化到 INT8
    float combined_scale = input_act->scale * w_scale;
    for (int i = 0; i < out_ch * out_h * out_w; i++) {
        float dequantized = acc[i] * combined_scale;
        output_int8[i] = (int8_t)round(clamp(dequantized / output_act->scale, -128, 127));
    }
    
    free(acc);
}
```

#### 量化权重导出

修改 `scripts/export_weights.py`，支持导出量化权重：

```python
def export_quantized_weights(model, output_path):
    """导出量化模型权重"""
    with open(output_path, 'wb') as f:
        # 写入魔数和版本
        f.write(struct.pack('I', 0x47544352))  # 'GTCR'
        f.write(struct.pack('I', 2))  # 版本 2（量化版本）
        
        # 遍历所有层
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.quantized.Conv2d):
                # 导出量化权重
                weight_int8 = module.weight().int_repr()  # INT8 权重
                weight_scale = module.weight().q_scale()  # Scale
                
                # 写入权重
                weight_np = weight_int8.numpy().astype(np.int8)
                f.write(weight_np.tobytes())
                
                # 写入 scale
                f.write(struct.pack('f', weight_scale))
                
                # 如果是每通道量化
                if hasattr(module.weight(), 'q_per_channel_scales'):
                    scales = module.weight().q_per_channel_scales().numpy()
                    f.write(struct.pack(f'{len(scales)}f', *scales))
```

### 3. 训练策略

#### 学习率调度

QAT 训练通常需要较小的学习率，避免破坏预训练权重：

```python
# 初始学习率：预训练的 1/10
initial_lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# 学习率调度：余弦退火
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=initial_lr * 0.01
)
```

#### 损失函数

使用与原始训练相同的损失函数（如 SI-SNR 或 MSE），但可以加入量化感知的正则项：

```python
def quantized_loss(output, target, model, lambda_q=0.01):
    # 主损失
    main_loss = F.mse_loss(output, target)
    
    # 量化正则项：鼓励权重分布集中在量化点附近
    quant_reg = 0.0
    for module in model.modules():
        if isinstance(module, torch.quantization.FakeQuantize):
            # 计算权重与最近量化点的距离
            if hasattr(module, 'scale'):
                # 简化：使用 scale 的倒数作为正则项
                quant_reg += 1.0 / (module.scale + 1e-8)
    
    return main_loss + lambda_q * quant_reg
```

#### 训练阶段

**阶段 1：预热（Warm-up）**
- 前 10% 的 epoch：只训练量化参数（observer），冻结模型权重
- 让量化参数适应数据分布

**阶段 2：联合训练**
- 解冻模型权重，联合优化模型和量化参数
- 使用较小的学习率，避免破坏预训练权重

**阶段 3：微调（Fine-tuning）**
- 最后 10% 的 epoch：进一步降低学习率
- 专注于精度恢复

```python
def train_qat(model, train_loader, num_epochs):
    warmup_epochs = int(num_epochs * 0.1)
    finetune_epochs = int(num_epochs * 0.1)
    
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            # 阶段 1：只训练量化参数
            for param in model.parameters():
                param.requires_grad = False
            model.apply(torch.quantization.enable_observer)
            model.apply(torch.quantization.disable_fake_quant)
        elif epoch < num_epochs - finetune_epochs:
            # 阶段 2：联合训练
            for param in model.parameters():
                param.requires_grad = True
            model.apply(torch.quantization.enable_observer)
            model.apply(torch.quantization.enable_fake_quant)
        else:
            # 阶段 3：微调
            model.apply(torch.quantization.disable_observer)
            model.apply(torch.quantization.enable_fake_quant)
        
        # 训练循环
        train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
```

## 评估与验证

### 精度评估指标

1. **SI-SNR（Scale-Invariant Signal-to-Noise Ratio）**
   - 主要指标，衡量语音质量

2. **PESQ（Perceptual Evaluation of Speech Quality）**
   - 感知质量评估

3. **STOI（Short-Time Objective Intelligibility）**
   - 可懂度评估

### 量化误差分析

```python
def analyze_quantization_error(fp32_model, quantized_model, test_loader):
    """分析量化误差"""
    fp32_outputs = []
    quant_outputs = []
    
    with torch.no_grad():
        for input, _ in test_loader:
            fp32_out = fp32_model(input)
            quant_out = quantized_model(input)
            
            fp32_outputs.append(fp32_out)
            quant_outputs.append(quant_out)
    
    # 计算逐层误差
    for i, (fp32, quant) in enumerate(zip(fp32_outputs, quant_outputs)):
        mse = F.mse_loss(fp32, quant)
        print(f"Layer {i}: MSE = {mse.item():.6f}")
```

### 性能对比

| 模型版本 | 参数量 | 内存占用 | 推理时间 | SI-SNR (dB) | PESQ |
|---------|--------|---------|---------|-------------|------|
| FP32    | 48.2K  | 192 KB  | 0.8 ms  | 18.5        | 3.2  |
| INT8 (PTQ) | 48.2K | 48 KB  | 0.3 ms  | 16.8 (-1.7) | 2.9  |
| INT8 (QAT) | 48.2K | 48 KB  | 0.3 ms  | 18.2 (-0.3) | 3.1  |

**预期结果**：
- QAT 相比 PTQ，SI-SNR 提升约 1.4 dB
- QAT 相比 FP32，精度损失 < 0.5 dB，性能提升 2-3 倍

## 部署集成

### C 语言量化推理集成

1. **修改权重加载函数**：
   - 支持读取量化权重文件
   - 解析 scale 和 zero_point

2. **添加量化计算函数**：
   - `quantized_conv2d_forward()`
   - `quantized_linear_forward()`
   - `quantize_tensor()`, `dequantize_tensor()`

3. **修改前向传播**：
   - 在每层输入/输出处插入量化/反量化
   - 使用 INT8 计算，最后反量化为 FP32 输出

### 性能优化

1. **SIMD 加速**：
   - 使用 SSE/AVX 指令集加速 INT8 矩阵乘法
   - 预期加速 2-4 倍

2. **内存优化**：
   - 中间激活值使用 INT8 存储
   - 减少内存带宽需求

3. **计算图优化**：
   - 融合量化-卷积-反量化操作
   - 减少中间结果存储

## 实施计划

### 阶段 1：QAT 实现（2 周）

- Week 1：
  - 实现 PyTorch QAT 模型改造
  - 配置量化策略和伪量化节点
  - 实现量化权重导出脚本

- Week 2：
  - 进行 QAT 训练实验
  - 调优超参数（学习率、正则项等）
  - 评估量化模型精度

### 阶段 2：C 语言集成（1.5 周）

- Week 3：
  - 实现量化权重加载
  - 实现 INT8 卷积和全连接层
  - 集成到前向传播流程

- Week 4（前半）：
  - 验证数值一致性
  - 性能测试和优化

### 阶段 3：优化与验证（0.5 周）

- Week 4（后半）：
  - SIMD 优化
  - 完整测试和文档更新

## 风险与应对

### 风险 1：精度损失过大

**应对**：
- 采用混合精度策略
- 关键层保持较高精度
- 增加训练轮数和微调阶段

### 风险 2：训练不稳定

**应对**：
- 使用较小的学习率
- 分阶段训练策略
- 监控训练过程，及时调整

### 风险 3：C 语言实现复杂度高

**应对**：
- 先实现 FP32 版本，再逐步替换为 INT8
- 充分测试每个模块
- 保留 FP32 版本作为 fallback

## 总结

GTCRN QAT 方案通过量化感知训练，能够在保持模型轻量级的同时，显著提升量化后的性能。相比后训练量化，QAT 能够将精度损失控制在 0.5 dB 以内，同时获得 2-3 倍的性能提升，为模型在资源受限设备上的部署提供了可行的解决方案。

通过分阶段实施和充分验证，可以确保量化模型的稳定性和可靠性，为实际应用奠定坚实基础。

