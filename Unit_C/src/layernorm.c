#include "layernorm.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// 辅助函数
// ============================================================================

int product(const int* arr, int n) {
    int result = 1;
    for (int i = 0; i < n; i++) {
        result *= arr[i];
    }
    return result;
}

void print_parameter_info(const char* name, const Parameter* param) {
    printf("%s: shape=(", name);
    for (int i = 0; i < param->ndim; i++) {
        printf("%d", param->shape[i]);
        if (i < param->ndim - 1) printf(", ");
    }
    printf("), total_size=%d\n", param->total_size);
}

void print_layernorm_info(const char* name, const LayerNormParams* params) {
    printf("%s: normalized_shape=(", name);
    for (int i = 0; i < params->ndim; i++) {
        printf("%d", params->normalized_shape[i]);
        if (i < params->ndim - 1) printf(", ");
    }
    printf("), eps=%.2e, num_features=%d\n", params->eps, params->num_features);
}


// ============================================================================
// nn.Parameter 实现
// ============================================================================

Parameter* parameter_create(int* shape, int ndim) {
    Parameter* param = (Parameter*)malloc(sizeof(Parameter));
    if (!param) return NULL;

    param->ndim = ndim;
    param->shape = (int*)malloc(ndim * sizeof(int));
    if (!param->shape) {
        free(param);
        return NULL;
    }

    memcpy(param->shape, shape, ndim * sizeof(int));
    param->total_size = product(shape, ndim);

    param->data = (float*)calloc(param->total_size, sizeof(float));
    if (!param->data) {
        free(param->shape);
        free(param);
        return NULL;
    }

    return param;
}

Parameter* parameter_from_data(float* data, int* shape, int ndim) {
    Parameter* param = parameter_create(shape, ndim);
    if (!param) return NULL;

    memcpy(param->data, data, param->total_size * sizeof(float));
    return param;
}

void parameter_free(Parameter* param) {
    if (param) {
        if (param->data) free(param->data);
        if (param->shape) free(param->shape);
        free(param);
    }
}


// ============================================================================
// nn.LayerNorm 实现
// ============================================================================

LayerNormParams* layernorm_create(
    int* normalized_shape,
    int ndim,
    const float* gamma,
    const float* beta,
    float eps
) {
    LayerNormParams* params = (LayerNormParams*)malloc(sizeof(LayerNormParams));
    if (!params) return NULL;

    params->ndim = ndim;
    params->eps = eps;

    // 复制 normalized_shape
    params->normalized_shape = (int*)malloc(ndim * sizeof(int));
    if (!params->normalized_shape) {
        free(params);
        return NULL;
    }
    memcpy(params->normalized_shape, normalized_shape, ndim * sizeof(int));

    // 计算特征总数
    params->num_features = product(normalized_shape, ndim);

    // 分配 gamma 和 beta
    params->gamma = (float*)malloc(params->num_features * sizeof(float));
    params->beta = (float*)malloc(params->num_features * sizeof(float));

    if (!params->gamma || !params->beta) {
        layernorm_free(params);
        return NULL;
    }

    // 初始化 gamma 和 beta
    if (gamma) {
        memcpy(params->gamma, gamma, params->num_features * sizeof(float));
    } else {
        // 默认 gamma = 1
        for (int i = 0; i < params->num_features; i++) {
            params->gamma[i] = 1.0f;
        }
    }

    if (beta) {
        memcpy(params->beta, beta, params->num_features * sizeof(float));
    } else {
        // 默认 beta = 0
        for (int i = 0; i < params->num_features; i++) {
            params->beta[i] = 0.0f;
        }
    }

    return params;
}

void layernorm_free(LayerNormParams* params) {
    if (params) {
        if (params->normalized_shape) free(params->normalized_shape);
        if (params->gamma) free(params->gamma);
        if (params->beta) free(params->beta);
        free(params);
    }
}

void layernorm_forward(
    float* input,
    float* output,
    int batch_size,
    const LayerNormParams* params
) {
    int num_features = params->num_features;

    // 对每个批次样本
    for (int b = 0; b < batch_size; b++) {
        float* batch_input = input + b * num_features;
        float* batch_output = output + b * num_features;

        // 1. 计算均值
        double sum = 0.0;
        for (int i = 0; i < num_features; i++) {
            sum += batch_input[i];
        }
        float mean = (float)(sum / num_features);

        // 2. 计算方差
        double var_sum = 0.0;
        for (int i = 0; i < num_features; i++) {
            float diff = batch_input[i] - mean;
            var_sum += diff * diff;
        }
        float var = (float)(var_sum / num_features);

        // 3. 归一化
        float std = sqrtf(var + params->eps);

        for (int i = 0; i < num_features; i++) {
            // y = gamma * (x - mean) / sqrt(var + eps) + beta
            float normalized = (batch_input[i] - mean) / std;
            batch_output[i] = params->gamma[i] * normalized + params->beta[i];
        }
    }
}

void layernorm_forward_4d(
    Tensor* input,
    const LayerNormParams* params
) {
    // Input: (B, T, F, C)
    // 归一化最后两个维度: (F, C)

    int batch = input->shape.batch;
    int channels = input->shape.channels;  // 这里实际是 T
    int height = input->shape.height;      // 这里实际是 F
    int width = input->shape.width;        // 这里实际是 C

    // 验证归一化维度
    if (params->ndim != 2) {
        printf("Error: Expected 2D normalized_shape for 4D input\n");
        return;
    }

    if (params->normalized_shape[0] != height || params->normalized_shape[1] != width) {
        printf("Error: normalized_shape mismatch. Expected (%d, %d), got (%d, %d)\n",
               height, width, params->normalized_shape[0], params->normalized_shape[1]);
        return;
    }

    int num_features = params->num_features;  // F * C
    int batch_size = batch * channels;        // B * T

    // 将 4D 张量视为 (B*T, F*C) 进行归一化
    layernorm_forward(input->data, input->data, batch_size, params);
}


// ============================================================================
// 专用版本：GTCRN DPGRNN LayerNorm
// ============================================================================

/*
 * GTCRN DPGRNN 中的 LayerNorm 使用方式：
 *
 * intra_ln: 输入 (B, T, F, C)，归一化 (F, C)
 * inter_ln: 输入 (B, T, F, C)，归一化 (F, C)
 *
 * 这个函数是 layernorm_forward_4d 的别名，为了清晰起见
 */
void dpgrnn_layernorm_forward(
    Tensor* input,          // (B, T, F, C)
    const LayerNormParams* params
) {
    layernorm_forward_4d(input, params);
}
