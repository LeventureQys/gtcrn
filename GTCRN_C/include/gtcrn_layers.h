/// <file>gtcrn_layers.h</file>
/// <summary>GTCRN神经网络层</summary>
/// <author>江月希 李文轩</author>
/// <remarks>神经网络层实现: Conv2d, GRU, BatchNorm, LayerNorm等</remarks>

#ifndef GTCRN_LAYERS_H
#define GTCRN_LAYERS_H

#include "gtcrn_types.h"
#include "gtcrn_math.h"

#ifdef __cplusplus
extern "C" {
#endif

// Conv2d层

/// <summary>Conv2d层权重</summary>
typedef struct {
    const gtcrn_float* weight;  /* (out_ch, in_ch/groups, kH, kW) */
    const gtcrn_float* bias;    /* (out_ch,) 或 NULL */
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int dilation_h;
    int dilation_w;
    int groups;
} gtcrn_conv2d_t;

/// <summary>应用Conv2d操作</summary>
/// <remarks>
/// 执行二维卷积操作,支持分组卷积、膨胀卷积等。输入和输出按行主序存储,形状为(B, C, H, W)。
/// 输出尺寸必须预先计算正确,否则可能越界。权重布局: (out_ch, in_ch/groups, kH, kW),按行主序。
/// 支持分组卷积(groups > 1),此时in_channels必须能被groups整除。支持膨胀卷积(dilation > 1),会增大感受野。
/// </remarks>
/// <param name="layer">Conv2d层参数结构,包含权重、偏置、卷积参数等</param>
/// <param name="input">输入张量数组,形状为(batch, in_channels, in_h, in_w),按行主序</param>
/// <param name="output">输出张量数组,必须预分配,形状为(batch, out_channels, out_h, out_w)</param>
/// <param name="batch">批次大小,必须大于0</param>
/// <param name="in_h">输入空间高度</param>
/// <param name="in_w">输入空间宽度</param>
/// <param name="out_h">输出空间高度,计算公式: (in_h + 2*padding_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1</param>
/// <param name="out_w">输出空间宽度,计算公式: (in_w + 2*padding_w - dilation_w*(kernel_w-1) - 1) / stride_w + 1</param>
void gtcrn_conv2d_forward(const gtcrn_conv2d_t* layer,
                          const gtcrn_float* input,
                          gtcrn_float* output,
                          int batch, int in_h, int in_w,
                          int out_h, int out_w);
void gtcrn_conv2d_forward(const gtcrn_conv2d_t* layer,
                          const gtcrn_float* input,
                          gtcrn_float* output,
                          int batch, int in_h, int in_w,
                          int out_h, int out_w);

/// <summary>应用ConvTranspose2d操作</summary>
/// <remarks>执行转置卷积(反卷积)操作,用于上采样。与普通卷积相反,用于将特征图尺寸增大。输出尺寸必须预先计算正确。常用于解码器中的上采样操作。权重布局与普通卷积相同,但计算过程相反。</remarks>
/// <param name="layer">Conv2d层参数结构,权重布局与普通卷积相同</param>
/// <param name="input">输入张量数组,形状为(batch, in_channels, in_h, in_w)</param>
/// <param name="output">输出张量数组,必须预分配,形状为(batch, out_channels, out_h, out_w)</param>
/// <param name="batch">批次大小</param>
/// <param name="in_h">输入空间高度</param>
/// <param name="in_w">输入空间宽度</param>
/// <param name="out_h">输出空间高度,计算公式: (in_h - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + 1</param>
/// <param name="out_w">输出空间宽度,计算公式: (in_w - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + 1</param>
void gtcrn_conv_transpose2d_forward(const gtcrn_conv2d_t* layer,
                                    const gtcrn_float* input,
                                    gtcrn_float* output,
                                    int batch, int in_h, int in_w,
                                    int out_h, int out_w);

// BatchNorm2d层

/// <summary>BatchNorm2d层权重(推理模式)</summary>
typedef struct {
    const gtcrn_float* gamma;       /* 缩放参数(num_features,) */
    const gtcrn_float* beta;        /* 偏置参数(num_features,) */
    const gtcrn_float* running_mean;
    const gtcrn_float* running_var;
    int num_features;
    gtcrn_float eps;
} gtcrn_batchnorm2d_t;

/// <summary>应用BatchNorm2d(推理模式)</summary>
/// <remarks>对二维特征图进行批量归一化,使用训练时统计的running_mean和running_var。这是原地操作,直接修改输入张量。使用推理模式公式: y = gamma * (x - mean) / sqrt(var + eps) + beta。running_mean和running_var是训练时统计的全局均值和方差。归一化在通道维度上进行,每个通道独立归一化。</remarks>
/// <param name="layer">BatchNorm2d参数结构,包含gamma、beta、running_mean、running_var等</param>
/// <param name="x">输入/输出张量数组,形状为(batch, num_features, height, width),按行主序,函数执行后此数组包含归一化后的结果</param>
/// <param name="batch">批次大小</param>
/// <param name="height">空间高度</param>
/// <param name="width">空间宽度</param>
void gtcrn_batchnorm2d_forward(const gtcrn_batchnorm2d_t* layer,
                               gtcrn_float* x,
                               int batch, int height, int width);

// LayerNorm层

/// <summary>LayerNorm层权重</summary>
typedef struct {
    const gtcrn_float* gamma;   /* 缩放参数(normalized_shape) */
    const gtcrn_float* beta;    /* 偏置参数(normalized_shape) */
    int normalized_size;        /* normalized_shape维度的乘积 */
    gtcrn_float eps;
} gtcrn_layernorm_t;

/// <summary>应用LayerNorm</summary>
/// <remarks>对张量的最后normalized_size个维度进行层归一化。与BatchNorm不同,LayerNorm在样本内归一化,不依赖批次统计。这是原地操作,输入数据会被覆盖。归一化公式: y = gamma * (x - mean) / sqrt(var + eps) + beta。每个样本的最后normalized_size个元素独立计算均值和方差。常用于RNN和Transformer中。</remarks>
/// <param name="layer">LayerNorm参数结构,包含gamma、beta和归一化尺寸</param>
/// <param name="x">输入/输出张量数组,按行主序存储,对每个独立样本的最后normalized_size个元素进行归一化</param>
/// <param name="batch">独立归一化的样本数量,等于前导维度的乘积,例如: 形状为(2, 16, 33)时,batch=2*16=32</param>
void gtcrn_layernorm_forward(const gtcrn_layernorm_t* layer,
                             gtcrn_float* x,
                             int batch);

// GRU层

/// <summary>GRU层权重</summary>
/// <remarks>PyTorch GRU权重布局: weight_ih, weight_hh, bias_ih, bias_hh。每个weight_ih: (3*hidden_size, input_size)对应[r, z, n],每个weight_hh: (3*hidden_size, hidden_size)对应[r, z, n]</remarks>
typedef struct {
    const gtcrn_float* weight_ih;   /* (3*hidden, input) */
    const gtcrn_float* weight_hh;   /* (3*hidden, hidden) */
    const gtcrn_float* bias_ih;     /* (3*hidden,) 或 NULL */
    const gtcrn_float* bias_hh;     /* (3*hidden,) 或 NULL */
    int input_size;
    int hidden_size;
    int bidirectional;
    /* 反向权重(如果双向) */
    const gtcrn_float* weight_ih_reverse;
    const gtcrn_float* weight_hh_reverse;
    const gtcrn_float* bias_ih_reverse;
    const gtcrn_float* bias_hh_reverse;
} gtcrn_gru_t;

/// <summary>GRU单向前向传播</summary>
/// <remarks>执行门控循环单元(GRU)的前向传播,处理序列数据。GRU包含重置门(r)、更新门(z)和新门(n),用于控制信息流动。权重布局遵循PyTorch格式: weight_ih: (3*hidden_size, input_size)对应[r, z, n]门, weight_hh: (3*hidden_size, hidden_size)对应[r, z, n]门。h_0可以为NULL,此时初始化为零。workspace可以复用,但每次调用都会覆盖内容。输出维度是hidden_size,输入维度是input_size。</remarks>
/// <param name="layer">GRU层参数结构,包含权重矩阵和偏置</param>
/// <param name="input">输入序列数组,形状为(batch, seq_len, input_size),按行主序,访问第b个样本第t个时间步: input[(b*seq_len + t)*input_size + i]</param>
/// <param name="h_0">初始隐藏状态数组,形状为(batch, hidden_size),可以为NULL,此时使用零初始化</param>
/// <param name="output">输出序列数组,必须预分配,形状为(batch, seq_len, hidden_size),每个时间步的输出隐藏状态</param>
/// <param name="h_n">最终隐藏状态数组,必须预分配,形状为(batch, hidden_size),存储最后一个时间步的隐藏状态,可用于下一段序列</param>
/// <param name="batch">批次大小,必须大于0</param>
/// <param name="seq_len">序列长度(时间步数),必须大于0</param>
/// <param name="workspace">临时工作空间数组,必须至少分配6 * hidden_size个gtcrn_float,用于存储中间计算结果,函数执行后内容未定义</param>
void gtcrn_gru_forward(const gtcrn_gru_t* layer,
                       const gtcrn_float* input,
                       const gtcrn_float* h_0,
                       gtcrn_float* output,
                       gtcrn_float* h_n,
                       int batch, int seq_len,
                       gtcrn_float* workspace);

/// <summary>双向GRU前向传播</summary>
/// <remarks>执行双向GRU的前向传播,同时处理正向和反向序列。输出维度是单向的2倍(hidden_size * 2)。输出维度是hidden_size * 2(正向和反向拼接)。反向序列从最后一个时间步开始处理。需要提供反向权重,否则行为未定义。</remarks>
/// <param name="layer">GRU层参数结构,必须设置bidirectional=1,必须提供反向权重(weight_ih_reverse等)</param>
/// <param name="input">输入序列数组,形状为(batch, seq_len, input_size)</param>
/// <param name="h_0">初始隐藏状态数组,形状为(batch, hidden_size * 2),前hidden_size个是正向初始状态,后hidden_size个是反向初始状态</param>
/// <param name="output">输出序列数组,必须预分配,形状为(batch, seq_len, hidden_size * 2),每时间步: [正向输出, 反向输出]</param>
/// <param name="h_n">最终隐藏状态数组,必须预分配,形状为(batch, hidden_size * 2),正向和反向的最终状态拼接</param>
/// <param name="batch">批次大小</param>
/// <param name="seq_len">序列长度</param>
/// <param name="workspace">临时工作空间,必须至少6 * hidden_size * 2个浮点数</param>
void gtcrn_gru_bidirectional_forward(const gtcrn_gru_t* layer,
                                     const gtcrn_float* input,
                                     const gtcrn_float* h_0,
                                     gtcrn_float* output,
                                     gtcrn_float* h_n,
                                     int batch, int seq_len,
                                     gtcrn_float* workspace);

// PReLU激活

/// <summary>PReLU激活参数</summary>
typedef struct {
    const gtcrn_float* alpha;   /* 可学习参数,大小 = num_parameters */
    int num_parameters;         /* 1表示共享,或num_channels */
} gtcrn_prelu_t;

/// <summary>应用PReLU激活</summary>
/// <remarks>执行参数化ReLU激活函数: f(x) = max(0, x) + alpha * min(0, x)。当alpha=0时退化为ReLU,当alpha=1时退化为线性。这是原地操作,输入数据会被覆盖。alpha可以是单个共享参数(num_parameters=1)或每通道独立参数。当num_parameters=1时,所有通道共享同一个alpha值。当num_parameters=channels时,每个通道有独立的alpha值。</remarks>
/// <param name="layer">PReLU参数结构,包含alpha参数数组</param>
/// <param name="x">输入/输出张量数组,形状为(batch, channels, height, width),按行主序,函数执行后此数组包含激活后的结果</param>
/// <param name="batch">批次大小</param>
/// <param name="channels">通道数</param>
/// <param name="spatial">空间大小,等于height * width</param>
void gtcrn_prelu_forward(const gtcrn_prelu_t* layer,
                         gtcrn_float* x,
                         int batch, int channels, int spatial);

// 线性层

/// <summary>线性层权重</summary>
typedef struct {
    const gtcrn_float* weight;  /* (out_features, in_features) */
    const gtcrn_float* bias;    /* (out_features,) 或 NULL */
    int in_features;
    int out_features;
} gtcrn_linear_t;

/// <summary>应用线性层</summary>
/// <remarks>执行全连接层(线性变换): output = input * weight^T + bias。等价于矩阵乘法加上偏置。权重矩阵布局: (out_features, in_features),按行主序,访问权重: weight[i * in_features + j] 表示第i个输出对第j个输入的权重。bias可以为NULL,此时不添加偏置。计算: output[b, i] = sum(input[b, j] * weight[i, j]) + bias[i]。</remarks>
/// <param name="layer">线性层参数结构,包含权重矩阵和偏置向量</param>
/// <param name="input">输入数组,形状为(batch, in_features),按行主序,访问第b个样本: input[b * in_features + i]</param>
/// <param name="output">输出数组,必须预分配,形状为(batch, out_features)</param>
/// <param name="batch">批次大小,必须大于0</param>
void gtcrn_linear_forward(const gtcrn_linear_t* layer,
                          const gtcrn_float* input,
                          gtcrn_float* output,
                          int batch);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_LAYERS_H */
