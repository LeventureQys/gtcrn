#include "nn_layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// nn.Linear 实现
// ============================================================================

LinearParams* linear_create(
    int in_features,
    int out_features,
    const float* weight,
    const float* bias,
    int use_bias
) {
    LinearParams* params = (LinearParams*)malloc(sizeof(LinearParams));
    if (!params) return NULL;

    params->in_features = in_features;
    params->out_features = out_features;
    params->use_bias = use_bias;

    // 分配权重 (out_features, in_features)
    params->weight = (float*)malloc(out_features * in_features * sizeof(float));
    if (!params->weight) {
        free(params);
        return NULL;
    }
    memcpy(params->weight, weight, out_features * in_features * sizeof(float));

    // 分配偏置
    if (use_bias) {
        params->bias = (float*)malloc(out_features * sizeof(float));
        if (!params->bias) {
            free(params->weight);
            free(params);
            return NULL;
        }
        memcpy(params->bias, bias, out_features * sizeof(float));
    } else {
        params->bias = NULL;
    }

    return params;
}

void linear_free(LinearParams* params) {
    if (params) {
        if (params->weight) free(params->weight);
        if (params->bias) free(params->bias);
        free(params);
    }
}

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

            // 矩阵乘法: output[o] = sum(input[i] * weight[o][i])
            for (int i = 0; i < in_features; i++) {
                int weight_idx = o * in_features + i;
                sum += input[b * in_features + i] * params->weight[weight_idx];
            }

            // 加偏置
            if (params->use_bias) {
                sum += params->bias[o];
            }

            output[b * out_features + o] = sum;
        }
    }
}


// ============================================================================
// nn.Unfold 实现
// ============================================================================

int calculate_unfold_output_length(
    int input_h,
    int input_w,
    const UnfoldParams* params
) {
    int out_h = calculate_output_size(input_h, params->kernel_h,
                                       params->stride_h, params->padding_h,
                                       params->dilation_h);
    int out_w = calculate_output_size(input_w, params->kernel_w,
                                       params->stride_w, params->padding_w,
                                       params->dilation_w);
    return out_h * out_w;
}

void unfold_forward(
    const Tensor* input,
    float* output,
    const UnfoldParams* params,
    int* out_length
) {
    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int in_h = input->shape.height;
    int in_w = input->shape.width;

    int kernel_h = params->kernel_h;
    int kernel_w = params->kernel_w;
    int stride_h = params->stride_h;
    int stride_w = params->stride_w;
    int padding_h = params->padding_h;
    int padding_w = params->padding_w;
    int dilation_h = params->dilation_h;
    int dilation_w = params->dilation_w;

    // 计算输出尺寸
    int out_h = calculate_output_size(in_h, kernel_h, stride_h, padding_h, dilation_h);
    int out_w = calculate_output_size(in_w, kernel_w, stride_w, padding_w, dilation_w);
    int L = out_h * out_w;
    *out_length = L;

    int blocks_per_channel = kernel_h * kernel_w;
    int total_blocks = channels * blocks_per_channel;

    // 对每个批次
    for (int b = 0; b < batch; b++) {
        int col_idx = 0;

        // 对每个输出位置
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int block_idx = 0;

                // 对每个通道
                for (int c = 0; c < channels; c++) {
                    // 对每个卷积核位置
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            // 计算输入位置
                            int ih = oh * stride_h - padding_h + kh * dilation_h;
                            int iw = ow * stride_w - padding_w + kw * dilation_w;

                            float val = 0.0f;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                                val = input->data[input_idx];
                            }

                            // 输出格式: (B, C*kh*kw, L)
                            int output_idx = (b * total_blocks + block_idx) * L + col_idx;
                            output[output_idx] = val;

                            block_idx++;
                        }
                    }
                }
                col_idx++;
            }
        }
    }
}

void unfold_reshape_4d(
    const Tensor* input,
    Tensor* output,
    const UnfoldParams* params
) {
    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int in_h = input->shape.height;
    int in_w = input->shape.width;

    int kernel_h = params->kernel_h;
    int kernel_w = params->kernel_w;

    // GTCRN SFE 的特殊情况: kernel_size=(1, k), stride=(1, 1)
    // 输出应该保持空间维度不变
    // 输出: (B, C*kernel_size, T, F)

    // 对每个批次
    for (int b = 0; b < batch; b++) {
        // 对每个通道
        for (int c = 0; c < channels; c++) {
            // 对每个空间位置
            for (int h = 0; h < in_h; h++) {
                for (int w = 0; w < in_w; w++) {
                    // 对每个卷积核位置
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            // 计算输入位置
                            int ih = h * params->stride_h - params->padding_h + kh * params->dilation_h;
                            int iw = w * params->stride_w - params->padding_w + kw * params->dilation_w;

                            float val = 0.0f;
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                                val = input->data[input_idx];
                            }

                            // 输出通道索引
                            int out_c = c * kernel_h * kernel_w + kh * kernel_w + kw;
                            int output_idx = ((b * output->shape.channels + out_c) * in_h + h) * in_w + w;
                            output->data[output_idx] = val;
                        }
                    }
                }
            }
        }
    }
}


// ============================================================================
// nn.PReLU 实现
// ============================================================================

PReLUParams* prelu_create(
    int num_parameters,
    const float* weight
) {
    PReLUParams* params = (PReLUParams*)malloc(sizeof(PReLUParams));
    if (!params) return NULL;

    params->num_parameters = num_parameters;
    params->weight = (float*)malloc(num_parameters * sizeof(float));

    if (!params->weight) {
        free(params);
        return NULL;
    }

    if (weight) {
        memcpy(params->weight, weight, num_parameters * sizeof(float));
    } else {
        // 默认值 0.25
        for (int i = 0; i < num_parameters; i++) {
            params->weight[i] = 0.25f;
        }
    }

    return params;
}

void prelu_free(PReLUParams* params) {
    if (params) {
        if (params->weight) free(params->weight);
        free(params);
    }
}

void prelu_forward_v2(
    Tensor* input,
    const PReLUParams* params
) {
    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int height = input->shape.height;
    int width = input->shape.width;

    // 对每个批次和通道
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float alpha = params->weight[c];

            // 对每个空间位置
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * channels + c) * height + h) * width + w;

                    // PReLU: y = x if x > 0, else alpha * x
                    if (input->data[idx] < 0) {
                        input->data[idx] *= alpha;
                    }
                }
            }
        }
    }
}


// ============================================================================
// nn.Sigmoid 实现
// ============================================================================

void sigmoid_forward(
    float* data,
    int size
) {
    for (int i = 0; i < size; i++) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        data[i] = 1.0f / (1.0f + expf(-data[i]));
    }
}

void sigmoid_forward_tensor(Tensor* input) {
    int total_size = input->shape.batch * input->shape.channels *
                     input->shape.height * input->shape.width;
    sigmoid_forward(input->data, total_size);
}


// ============================================================================
// 辅助函数
// ============================================================================

void print_tensor_stats_v2(const char* name, const Tensor* tensor) {
    int total = tensor->shape.batch * tensor->shape.channels *
                tensor->shape.height * tensor->shape.width;

    float min_val = tensor->data[0];
    float max_val = tensor->data[0];
    double sum = 0.0;

    for (int i = 0; i < total; i++) {
        float val = tensor->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    printf("%s [%d,%d,%d,%d]: min=%.6f, max=%.6f, mean=%.6f\n",
           name,
           tensor->shape.batch, tensor->shape.channels,
           tensor->shape.height, tensor->shape.width,
           min_val, max_val, sum / total);
}
