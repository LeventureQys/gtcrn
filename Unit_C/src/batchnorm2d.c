#include "batchnorm2d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

BatchNorm2dParams* batchnorm2d_create(
    int num_features,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps
) {
    BatchNorm2dParams* params = (BatchNorm2dParams*)malloc(sizeof(BatchNorm2dParams));
    if (!params) return NULL;

    params->num_features = num_features;
    params->eps = eps;

    params->gamma = (float*)malloc(num_features * sizeof(float));
    params->beta = (float*)malloc(num_features * sizeof(float));
    params->running_mean = (float*)malloc(num_features * sizeof(float));
    params->running_var = (float*)malloc(num_features * sizeof(float));

    if (!params->gamma || !params->beta || !params->running_mean || !params->running_var) {
        batchnorm2d_free(params);
        return NULL;
    }

    memcpy(params->gamma, gamma, num_features * sizeof(float));
    memcpy(params->beta, beta, num_features * sizeof(float));
    memcpy(params->running_mean, running_mean, num_features * sizeof(float));
    memcpy(params->running_var, running_var, num_features * sizeof(float));

    return params;
}

void batchnorm2d_free(BatchNorm2dParams* params) {
    if (params) {
        if (params->gamma) free(params->gamma);
        if (params->beta) free(params->beta);
        if (params->running_mean) free(params->running_mean);
        if (params->running_var) free(params->running_var);
        free(params);
    }
}

void batchnorm2d_forward(
    Tensor* input,
    const BatchNorm2dParams* params
) {
    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int height = input->shape.height;
    int width = input->shape.width;

    // For each batch and channel
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float mean = params->running_mean[c];
            float var = params->running_var[c];
            float std = sqrtf(var + params->eps);
            float scale = params->gamma[c];
            float shift = params->beta[c];

            // Normalize all spatial locations for this channel
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * channels + c) * height + h) * width + w;

                    // y = gamma * (x - mean) / sqrt(var + eps) + beta
                    float normalized = (input->data[idx] - mean) / std;
                    input->data[idx] = scale * normalized + shift;
                }
            }
        }
    }
}

void fuse_conv_batchnorm(
    FusedConvBN* fused,
    const Conv2dParams* conv_params,
    const BatchNorm2dParams* bn_params
) {
    // Copy conv parameters
    fused->conv_params = *conv_params;
    fused->is_fused = 1;

    int out_channels = conv_params->out_channels;
    int in_channels = conv_params->in_channels;
    int kernel_h = conv_params->kernel_h;
    int kernel_w = conv_params->kernel_w;
    int groups = conv_params->groups;

    int in_channels_per_group = in_channels / groups;
    int weight_size = out_channels * in_channels_per_group * kernel_h * kernel_w;

    // Allocate fused parameters
    fused->fused_weight = (float*)malloc(weight_size * sizeof(float));
    fused->fused_bias = (float*)malloc(out_channels * sizeof(float));

    if (!fused->fused_weight || !fused->fused_bias) {
        printf("Error: Failed to allocate fused parameters\n");
        return;
    }

    // Fuse weights and bias
    // For each output channel
    for (int oc = 0; oc < out_channels; oc++) {
        float gamma = bn_params->gamma[oc];
        float beta = bn_params->beta[oc];
        float mean = bn_params->running_mean[oc];
        float var = bn_params->running_var[oc];
        float std = sqrtf(var + bn_params->eps);

        // Scale factor for this channel
        float scale = gamma / std;

        // Fuse weights: w_fused = w * gamma / sqrt(var + eps)
        for (int ic = 0; ic < in_channels_per_group; ic++) {
            for (int kh = 0; kh < kernel_h; kh++) {
                for (int kw = 0; kw < kernel_w; kw++) {
                    int weight_idx = ((oc * in_channels_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                    fused->fused_weight[weight_idx] = conv_params->weight[weight_idx] * scale;
                }
            }
        }

        // Fuse bias: b_fused = (b - mean) * gamma / sqrt(var + eps) + beta
        float original_bias = conv_params->use_bias ? conv_params->bias[oc] : 0.0f;
        fused->fused_bias[oc] = (original_bias - mean) * scale + beta;
    }

    // Update conv_params to use fused parameters
    fused->conv_params.weight = fused->fused_weight;
    fused->conv_params.bias = fused->fused_bias;
    fused->conv_params.use_bias = 1;  // Always use bias after fusion
}

void fused_conv_bn_forward(
    const Tensor* input,
    Tensor* output,
    FusedConvBN* fused
) {
    if (!fused->is_fused) {
        printf("Error: Conv+BN not fused yet. Call fuse_conv_batchnorm first.\n");
        return;
    }

    // Use regular conv2d with fused parameters
    conv2d_forward(input, output, &fused->conv_params);
}

void fused_conv_bn_free(FusedConvBN* fused) {
    if (fused->fused_weight) {
        free(fused->fused_weight);
        fused->fused_weight = NULL;
    }
    if (fused->fused_bias) {
        free(fused->fused_bias);
        fused->fused_bias = NULL;
    }
    fused->is_fused = 0;
}
