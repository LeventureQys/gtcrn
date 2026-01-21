/// <file>gtcrn_math.c</file>
/// <summary>GTCRN数学工具实现</summary>
/// <author>江月希 李文轩</author>

#include "gtcrn_math.h"
#include <string.h>

/* 向量操作 */

void gtcrn_vec_add(const gtcrn_float* a, const gtcrn_float* b,
                   gtcrn_float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] + b[i];
    }
}

void gtcrn_vec_sub(const gtcrn_float* a, const gtcrn_float* b,
                   gtcrn_float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] - b[i];
    }
}

void gtcrn_vec_mul(const gtcrn_float* a, const gtcrn_float* b,
                   gtcrn_float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = a[i] * b[i];
    }
}

void gtcrn_vec_scale(const gtcrn_float* x, gtcrn_float alpha,
                     gtcrn_float* y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i];
    }
}

void gtcrn_vec_copy(const gtcrn_float* x, gtcrn_float* y, int n) {
    memcpy(y, x, n * sizeof(gtcrn_float));
}

void gtcrn_vec_zero(gtcrn_float* x, int n) {
    memset(x, 0, n * sizeof(gtcrn_float));
}

void gtcrn_vec_set(gtcrn_float* x, gtcrn_float val, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = val;
    }
}

gtcrn_float gtcrn_vec_dot(const gtcrn_float* a, const gtcrn_float* b, int n) {
    gtcrn_float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

gtcrn_float gtcrn_vec_sum(const gtcrn_float* x, int n) {
    gtcrn_float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

gtcrn_float gtcrn_vec_mean(const gtcrn_float* x, int n) {
    return gtcrn_vec_sum(x, n) / (gtcrn_float)n;
}

gtcrn_float gtcrn_vec_var(const gtcrn_float* x, gtcrn_float mean, int n) {
    gtcrn_float var = 0.0f;
    for (int i = 0; i < n; i++) {
        gtcrn_float diff = x[i] - mean;
        var += diff * diff;
    }
    return var / (gtcrn_float)n;
}

void gtcrn_vec_sigmoid(gtcrn_float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = gtcrn_sigmoid(x[i]);
    }
}

void gtcrn_vec_tanh(gtcrn_float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = gtcrn_tanh(x[i]);
    }
}

void gtcrn_vec_prelu(gtcrn_float* x, const gtcrn_float* alpha, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] > 0 ? x[i] : alpha[0] * x[i];
    }
}

/* 矩阵操作 */

void gtcrn_matvec(const gtcrn_float* A, const gtcrn_float* x,
                  const gtcrn_float* bias, gtcrn_float* y,
                  int M, int N) {
    for (int i = 0; i < M; i++) {
        gtcrn_float sum = bias ? bias[i] : 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

void gtcrn_matmul(const gtcrn_float* A, const gtcrn_float* B,
                  gtcrn_float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            gtcrn_float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void gtcrn_linear(const gtcrn_float* weight, const gtcrn_float* bias,
                  const gtcrn_float* input, gtcrn_float* output,
                  int batch, int in_features, int out_features) {
    for (int b = 0; b < batch; b++) {
        gtcrn_matvec(weight,
                     input + b * in_features,
                     bias,
                     output + b * out_features,
                     out_features, in_features);
    }
}
