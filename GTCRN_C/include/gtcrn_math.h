/// <file>gtcrn_math.h</file>
/// <summary>GTCRN数学工具</summary>
/// <author>江月希 李文轩</author>
/// <remarks>GTCRN库中使用的基本数学运算</remarks>

#ifndef GTCRN_MATH_H
#define GTCRN_MATH_H

#include "gtcrn_types.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 基本数学常量 */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GTCRN_EPS 1e-12f

/* 激活函数 */

/// <summary>Sigmoid激活: 1 / (1 + exp(-x))</summary>
static inline gtcrn_float gtcrn_sigmoid(gtcrn_float x) {
    return 1.0f / (1.0f + expf(-x));
}

/// <summary>Tanh激活</summary>
static inline gtcrn_float gtcrn_tanh(gtcrn_float x) {
    return tanhf(x);
}

/// <summary>PReLU激活: max(0, x) + alpha * min(0, x)</summary>
static inline gtcrn_float gtcrn_prelu(gtcrn_float x, gtcrn_float alpha) {
    return x > 0 ? x : alpha * x;
}

/// <summary>ReLU激活: max(0, x)</summary>
static inline gtcrn_float gtcrn_relu(gtcrn_float x) {
    return x > 0 ? x : 0;
}

/* 向量操作 */

/// <summary>向量加法: y = a + b</summary>
/// <remarks>计算两个向量的逐元素加法,结果存储到输出向量。执行: y[i] = a[i] + b[i] for i in [0, n)。输入和输出可以重叠(原地操作),但建议使用不同缓冲区。</remarks>
/// <param name="a">第一个输入向量,长度为n</param>
/// <param name="b">第二个输入向量,长度为n</param>
/// <param name="y">输出向量,必须预分配,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_add(const gtcrn_float* a, const gtcrn_float* b,
                   gtcrn_float* y, int n);

/// <summary>向量减法: y = a - b</summary>
/// <remarks>计算两个向量的逐元素减法。执行: y[i] = a[i] - b[i] for i in [0, n)。</remarks>
/// <param name="a">被减数向量,长度为n</param>
/// <param name="b">减数向量,长度为n</param>
/// <param name="y">输出向量,必须预分配,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_sub(const gtcrn_float* a, const gtcrn_float* b,
                   gtcrn_float* y, int n);

/// <summary>逐元素向量乘法: y = a * b</summary>
/// <remarks>计算两个向量的逐元素乘法(Hadamard积)。执行: y[i] = a[i] * b[i] for i in [0, n)。不同于点积,这是逐元素乘法。</remarks>
/// <param name="a">第一个输入向量,长度为n</param>
/// <param name="b">第二个输入向量,长度为n</param>
/// <param name="y">输出向量,必须预分配,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_mul(const gtcrn_float* a, const gtcrn_float* b,
                   gtcrn_float* y, int n);

/// <summary>标量-向量乘法: y = alpha * x</summary>
/// <remarks>将向量乘以标量,结果存储到输出向量。执行: y[i] = alpha * x[i] for i in [0, n)。可用于缩放向量。</remarks>
/// <param name="x">输入向量,长度为n</param>
/// <param name="alpha">标量乘数</param>
/// <param name="y">输出向量,必须预分配,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_scale(const gtcrn_float* x, gtcrn_float alpha,
                     gtcrn_float* y, int n);

/// <summary>向量复制: y = x</summary>
/// <remarks>将输入向量复制到输出向量。执行: y[i] = x[i] for i in [0, n)。等价于memcpy,但类型安全。</remarks>
/// <param name="x">源向量,长度为n</param>
/// <param name="y">目标向量,必须预分配,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_copy(const gtcrn_float* x, gtcrn_float* y, int n);

/// <summary>将向量置零</summary>
/// <remarks>将向量的所有元素设置为零。执行: x[i] = 0.0 for i in [0, n)。等价于memset(x, 0, n * sizeof(gtcrn_float))。</remarks>
/// <param name="x">目标向量,长度为n,函数执行后全为零</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_zero(gtcrn_float* x, int n);

/// <summary>将向量设为常数值</summary>
/// <remarks>将向量的所有元素设置为指定的常数值。执行: x[i] = val for i in [0, n)。</remarks>
/// <param name="x">目标向量,长度为n,函数执行后所有元素为val</param>
/// <param name="val">要设置的常数值</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_set(gtcrn_float* x, gtcrn_float val, int n);

/// <summary>向量点积: sum(a * b)</summary>
/// <remarks>计算两个向量的内积(点积),返回标量结果。用于计算向量相似度、投影等。如果n=0,返回0.0。</remarks>
/// <param name="a">第一个向量,长度为n</param>
/// <param name="b">第二个向量,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
/// <returns>点积结果: sum(a[i] * b[i] for i in [0, n))</returns>
gtcrn_float gtcrn_vec_dot(const gtcrn_float* a, const gtcrn_float* b, int n);

/// <summary>向量求和: sum(x)</summary>
/// <remarks>计算向量所有元素的和。用于统计计算。如果n=0,返回0.0。</remarks>
/// <param name="x">输入向量,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
/// <returns>所有元素的和</returns>
gtcrn_float gtcrn_vec_sum(const gtcrn_float* x, int n);

/// <summary>向量均值: sum(x) / n</summary>
/// <remarks>计算向量所有元素的算术平均值。如果n=0,行为未定义(除零)。</remarks>
/// <param name="x">输入向量,长度为n</param>
/// <param name="n">向量长度,必须大于0</param>
/// <returns>均值: sum(x) / n</returns>
gtcrn_float gtcrn_vec_mean(const gtcrn_float* x, int n);

/// <summary>向量方差: sum((x - mean)^2) / n</summary>
/// <remarks>计算向量的方差,需要预先知道均值。需要预先计算均值,避免重复计算。如果n=0,行为未定义。</remarks>
/// <param name="x">输入向量,长度为n</param>
/// <param name="mean">向量的均值,通常通过gtcrn_vec_mean()计算</param>
/// <param name="n">向量长度,必须大于0</param>
/// <returns>方差: sum((x[i] - mean)^2) / n</returns>
gtcrn_float gtcrn_vec_var(const gtcrn_float* x, gtcrn_float mean, int n);

/// <summary>原地应用sigmoid到向量</summary>
/// <remarks>对向量的每个元素应用sigmoid激活函数: f(x) = 1 / (1 + exp(-x))。这是原地操作,直接修改输入向量。执行: x[i] = 1 / (1 + exp(-x[i])) for i in [0, n)。输出范围在(0, 1)。常用于二分类问题的输出层。</remarks>
/// <param name="x">输入/输出向量,长度为n,函数执行后包含sigmoid结果</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_sigmoid(gtcrn_float* x, int n);

/// <summary>原地应用tanh到向量</summary>
/// <remarks>对向量的每个元素应用双曲正切激活函数: f(x) = tanh(x)。这是原地操作,直接修改输入向量。执行: x[i] = tanh(x[i]) for i in [0, n)。输出范围在(-1, 1)。常用于RNN的激活函数。</remarks>
/// <param name="x">输入/输出向量,长度为n,函数执行后包含tanh结果</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_tanh(gtcrn_float* x, int n);

/// <summary>原地应用PReLU到向量</summary>
/// <remarks>对向量的每个元素应用参数化ReLU激活函数: f(x) = max(0, x) + alpha * min(0, x)。这是原地操作,直接修改输入向量。执行: x[i] = x[i] > 0 ? x[i] : alpha[i] * x[i]。如果alpha长度为1,所有元素共享同一个alpha值。如果alpha长度为n,每个元素有独立的alpha值。</remarks>
/// <param name="x">输入/输出向量,长度为n,函数执行后包含PReLU结果</param>
/// <param name="alpha">alpha参数数组,长度必须为n(每元素独立)或1(共享)</param>
/// <param name="n">向量长度,必须大于0</param>
void gtcrn_vec_prelu(gtcrn_float* x, const gtcrn_float* alpha, int n);

/* 矩阵操作 */

/// <summary>矩阵-向量乘法: y = A * x + bias</summary>
/// <remarks>计算矩阵与向量的乘积并加上偏置,用于线性变换。计算: y[i] = sum(A[i, j] * x[j] for j in [0, N)) + bias[i]。如果bias为NULL,不添加偏置项。矩阵按行主序存储: 第i行的元素在A[i*N : (i+1)*N]。</remarks>
/// <param name="A">输入矩阵,形状为(M, N),按行主序存储,访问第i行第j列: A[i * N + j]</param>
/// <param name="x">输入向量,长度为N</param>
/// <param name="bias">偏置向量,长度为M,可以为NULL(不添加偏置)</param>
/// <param name="y">输出向量,必须预分配,长度为M</param>
/// <param name="M">矩阵行数(输出维度),必须大于0</param>
/// <param name="N">矩阵列数(输入维度),必须大于0</param>
void gtcrn_matvec(const gtcrn_float* A, const gtcrn_float* x,
                  const gtcrn_float* bias, gtcrn_float* y,
                  int M, int N);

/// <summary>通用矩阵乘法: C = A * B</summary>
/// <remarks>计算两个矩阵的乘积,结果存储到输出矩阵。计算: C[i, j] = sum(A[i, k] * B[k, j] for k in [0, K))。所有矩阵按行主序存储。复杂度: O(M * K * N)。</remarks>
/// <param name="A">第一个矩阵,形状为(M, K),按行主序</param>
/// <param name="B">第二个矩阵,形状为(K, N),按行主序</param>
/// <param name="C">输出矩阵,必须预分配,形状为(M, N),按行主序</param>
/// <param name="M">A的行数(也是C的行数),必须大于0</param>
/// <param name="K">A的列数(也是B的行数),必须大于0</param>
/// <param name="N">B的列数(也是C的列数),必须大于0</param>
void gtcrn_matmul(const gtcrn_float* A, const gtcrn_float* B,
                  gtcrn_float* C, int M, int K, int N);

/// <summary>线性层的批量矩阵-向量乘法</summary>
/// <remarks>对批量输入执行线性变换,等价于对每个样本执行矩阵-向量乘法。这是神经网络线性层的核心操作。对每个样本计算: output[b, i] = sum(input[b, j] * weight[i, j]) + bias[i]。等价于对每个样本调用gtcrn_matvec()。权重矩阵布局与PyTorch Linear层一致。</remarks>
/// <param name="weight">权重矩阵,形状为(out_features, in_features),按行主序,访问: weight[i * in_features + j] 表示第i个输出对第j个输入的权重</param>
/// <param name="bias">偏置向量,长度为out_features,可以为NULL(不添加偏置)</param>
/// <param name="input">输入数组,形状为(batch, in_features),按行主序,访问第b个样本: input[b * in_features + i]</param>
/// <param name="output">输出数组,必须预分配,形状为(batch, out_features)</param>
/// <param name="batch">批次大小,必须大于0</param>
/// <param name="in_features">输入特征维度,必须大于0</param>
/// <param name="out_features">输出特征维度,必须大于0</param>
void gtcrn_linear(const gtcrn_float* weight, const gtcrn_float* bias,
                  const gtcrn_float* input, gtcrn_float* output,
                  int batch, int in_features, int out_features);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_MATH_H */
