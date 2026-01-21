#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "conv2d.h"

/*
 * LayerNorm 和 Parameter 的 C 实现
 */

// ============================================================================
// nn.Parameter - 可学习参数
// ============================================================================

/*
 * nn.Parameter 在 C 中就是普通的 float 数组
 * 在 PyTorch 中，Parameter 是一个特殊的 Tensor，会被自动注册为模型参数
 * 在 C 中，我们只需要管理这些参数的内存
 *
 * 例如：
 *   PyTorch: self.weight = nn.Parameter(torch.randn(10, 20))
 *   C:       float* weight = (float*)malloc(10 * 20 * sizeof(float));
 */

typedef struct {
    float* data;
    int* shape;
    int ndim;
    int total_size;
} Parameter;

/*
 * 创建参数
 */
Parameter* parameter_create(int* shape, int ndim);

/*
 * 从数据创建参数
 */
Parameter* parameter_from_data(float* data, int* shape, int ndim);

/*
 * 释放参数
 */
void parameter_free(Parameter* param);


// ============================================================================
// nn.LayerNorm - 层归一化
// ============================================================================

typedef struct {
    int* normalized_shape;  // 归一化的维度
    int ndim;               // normalized_shape 的维度数
    float* gamma;           // 缩放参数（可学习）
    float* beta;            // 偏移参数（可学习）
    float eps;              // 数值稳定性常数
    int num_features;       // 特征总数（normalized_shape 的乘积）
} LayerNormParams;

/*
 * LayerNorm forward
 *
 * 从 gtcrn1.py line 196, 200:
 *   self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
 *   self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)
 *
 * 在 DPGRNN 中使用:
 *   Input: (B, T, F, C) 其中 F=width, C=hidden_size
 *   归一化最后两个维度: (F, C)
 *
 * 公式:
 *   mean = mean(x, dim=normalized_dims)
 *   var = var(x, dim=normalized_dims)
 *   y = gamma * (x - mean) / sqrt(var + eps) + beta
 */
void layernorm_forward(
    float* input,           // 输入数据
    float* output,          // 输出数据
    int batch_size,         // 批次大小（所有非归一化维度的乘积）
    const LayerNormParams* params
);

/*
 * LayerNorm forward (Tensor 版本)
 * 用于 4D 张量，归一化最后两个维度
 *
 * Input: (B, T, F, C)
 * 归一化: (F, C)
 */
void layernorm_forward_4d(
    Tensor* input,          // (B, T, F, C) - 会被 reshape
    const LayerNormParams* params
);

/*
 * 创建 LayerNorm 参数
 */
LayerNormParams* layernorm_create(
    int* normalized_shape,  // 例如: [97, 16] 表示归一化最后两个维度
    int ndim,               // normalized_shape 的维度数
    const float* gamma,     // 缩放参数（如果为 NULL，初始化为 1）
    const float* beta,      // 偏移参数（如果为 NULL，初始化为 0）
    float eps               // 数值稳定性常数（通常 1e-5 或 1e-8）
);

/*
 * 释放 LayerNorm 参数
 */
void layernorm_free(LayerNormParams* params);


// ============================================================================
// 辅助函数
// ============================================================================

/*
 * 计算数组乘积
 */
int product(const int* arr, int n);

/*
 * 打印参数信息
 */
void print_parameter_info(const char* name, const Parameter* param);

/*
 * 打印 LayerNorm 信息
 */
void print_layernorm_info(const char* name, const LayerNormParams* params);

#endif
