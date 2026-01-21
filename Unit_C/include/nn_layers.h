#ifndef NN_LAYERS_H
#define NN_LAYERS_H

#include "conv2d.h"

/*
 * 神经网络基础层的C实现
 * 包括: Linear, Unfold, PReLU, Sigmoid, Tanh
 */

// ============================================================================
// nn.Linear - 全连接层
// ============================================================================

typedef struct {
    int in_features;
    int out_features;
    float* weight;      // (out_features, in_features)
    float* bias;        // (out_features)
    int use_bias;
} LinearParams;

/*
 * Linear forward: y = x @ W^T + b
 * Input:  (*, in_features)
 * Output: (*, out_features)
 *
 * 支持任意维度的输入，最后一维是 in_features
 */
void linear_forward(
    const float* input,     // 输入数据
    float* output,          // 输出数据
    int batch_size,         // 批次大小（所有前导维度的乘积）
    const LinearParams* params
);

/*
 * 创建 Linear 参数
 */
LinearParams* linear_create(
    int in_features,
    int out_features,
    const float* weight,
    const float* bias,
    int use_bias
);

/*
 * 释放 Linear 参数
 */
void linear_free(LinearParams* params);


// ============================================================================
// nn.Unfold - 展开操作（im2col）
// ============================================================================

typedef struct {
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int dilation_h;
    int dilation_w;
} UnfoldParams;

/*
 * Unfold forward: 将图像块展开成列
 *
 * 用于 SFE (Subband Feature Extraction)
 * 从 gtcrn1.py line 69:
 *   nn.Unfold(kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,(kernel_size-1)//2))
 *
 * Input:  (B, C, H, W)
 * Output: (B, C*kernel_h*kernel_w, L)
 *   其中 L = output_h * output_w
 */
void unfold_forward(
    const Tensor* input,
    float* output,          // (B, C*kh*kw, L)
    const UnfoldParams* params,
    int* out_length         // 输出 L 的值
);

/*
 * Unfold 并 reshape 为 4D 张量
 * 这是 GTCRN SFE 的实际使用方式
 *
 * Input:  (B, C, T, F)
 * Output: (B, C*kernel_size, T, F)
 */
void unfold_reshape_4d(
    const Tensor* input,
    Tensor* output,
    const UnfoldParams* params
);


// ============================================================================
// nn.PReLU - Parametric ReLU
// ============================================================================

typedef struct {
    int num_parameters;     // 参数数量（通常等于通道数）
    float* weight;          // 负斜率参数
} PReLUParams;

/*
 * PReLU forward:
 *   y = x           if x > 0
 *   y = weight * x  if x <= 0
 *
 * 从 gtcrn1.py line 102, 119, 125:
 *   self.act = nn.PReLU()
 *
 * Input/Output: (B, C, H, W)
 * weight: (C,) - 每个通道一个参数
 */
void prelu_forward_v2(
    Tensor* input,
    const PReLUParams* params
);

/*
 * 创建 PReLU 参数
 */
PReLUParams* prelu_create(
    int num_parameters,
    const float* weight
);

/*
 * 释放 PReLU 参数
 */
void prelu_free(PReLUParams* params);


// ============================================================================
// nn.Sigmoid - Sigmoid 激活
// ============================================================================

/*
 * Sigmoid forward: y = 1 / (1 + exp(-x))
 *
 * 从 gtcrn1.py line 83:
 *   self.att_act = nn.Sigmoid()
 *
 * Input/Output: 任意形状
 */
void sigmoid_forward(
    float* data,
    int size
);

/*
 * Sigmoid forward (Tensor 版本)
 */
void sigmoid_forward_tensor(Tensor* input);


// ============================================================================
// 辅助函数
// ============================================================================

/*
 * 计算 Unfold 输出的长度
 */
int calculate_unfold_output_length(
    int input_h,
    int input_w,
    const UnfoldParams* params
);

/*
 * 打印张量统计信息（用于调试）
 */
void print_tensor_stats_v2(const char* name, const Tensor* tensor);

#endif
