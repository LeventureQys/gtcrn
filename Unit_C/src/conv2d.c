#include "conv2d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

Tensor* tensor_create(int batch, int channels, int height, int width) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->shape.batch = batch;
    tensor->shape.channels = channels;
    tensor->shape.height = height;
    tensor->shape.width = width;

    int total_size = batch * channels * height * width;
    tensor->data = (float*)calloc(total_size, sizeof(float));

    if (!tensor->data) {
        free(tensor);
        return NULL;
    }

    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (tensor) {
        if (tensor->data) {
            free(tensor->data);
        }
        free(tensor);
    }
}

int calculate_output_size(
    int input_size,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int effective_kernel = dilation * (kernel_size - 1) + 1;
    return (input_size + 2 * padding - effective_kernel) / stride + 1;
}

int calculate_transpose_output_size(
    int input_size,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    int effective_kernel = dilation * (kernel_size - 1) + 1;
    return (input_size - 1) * stride - 2 * padding + effective_kernel;
}

void conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
) {
    int batch = input->shape.batch;
    int in_h = input->shape.height;
    int in_w = input->shape.width;
    int out_h = output->shape.height;
    int out_w = output->shape.width;

    int in_channels_per_group = params->in_channels / params->groups;
    int out_channels_per_group = params->out_channels / params->groups;

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < params->groups; g++) {
            for (int oc = 0; oc < out_channels_per_group; oc++) {
                int out_c = g * out_channels_per_group + oc;

                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        float sum = 0.0f;

                        for (int ic = 0; ic < in_channels_per_group; ic++) {
                            int in_c = g * in_channels_per_group + ic;

                            for (int kh = 0; kh < params->kernel_h; kh++) {
                                for (int kw = 0; kw < params->kernel_w; kw++) {
                                    int ih = oh * params->stride_h - params->padding_h + kh * params->dilation_h;
                                    int iw = ow * params->stride_w - params->padding_w + kw * params->dilation_w;

                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        int input_idx = ((b * input->shape.channels + in_c) * in_h + ih) * in_w + iw;
                                        int weight_idx = ((out_c * in_channels_per_group + ic) * params->kernel_h + kh) * params->kernel_w + kw;

                                        sum += input->data[input_idx] * params->weight[weight_idx];
                                    }
                                }
                            }
                        }

                        if (params->use_bias) {
                            sum += params->bias[out_c];
                        }

                        int output_idx = ((b * output->shape.channels + out_c) * out_h + oh) * out_w + ow;
                        output->data[output_idx] = sum;
                    }
                }
            }
        }
    }
}

void conv2d_transpose_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
) {
    int batch = input->shape.batch;
    int in_h = input->shape.height;
    int in_w = input->shape.width;
    int out_h = output->shape.height;
    int out_w = output->shape.width;

    memset(output->data, 0, batch * output->shape.channels * out_h * out_w * sizeof(float));

    int in_channels_per_group = params->in_channels / params->groups;
    int out_channels_per_group = params->out_channels / params->groups;

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < params->groups; g++) {
            for (int ic = 0; ic < in_channels_per_group; ic++) {
                int in_c = g * in_channels_per_group + ic;

                for (int ih = 0; ih < in_h; ih++) {
                    for (int iw = 0; iw < in_w; iw++) {
                        int input_idx = ((b * input->shape.channels + in_c) * in_h + ih) * in_w + iw;
                        float input_val = input->data[input_idx];

                        for (int oc = 0; oc < out_channels_per_group; oc++) {
                            int out_c = g * out_channels_per_group + oc;

                            for (int kh = 0; kh < params->kernel_h; kh++) {
                                for (int kw = 0; kw < params->kernel_w; kw++) {
                                    int oh = ih * params->stride_h - params->padding_h + kh * params->dilation_h;
                                    int ow = iw * params->stride_w - params->padding_w + kw * params->dilation_w;

                                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                        int output_idx = ((b * output->shape.channels + out_c) * out_h + oh) * out_w + ow;
                                        int weight_idx = ((in_c * out_channels_per_group + oc) * params->kernel_h + kh) * params->kernel_w + kw;

                                        output->data[output_idx] += input_val * params->weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (params->use_bias) {
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < output->shape.channels; c++) {
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        int idx = ((b * output->shape.channels + c) * out_h + h) * out_w + w;
                        output->data[idx] += params->bias[c];
                    }
                }
            }
        }
    }
}

void depthwise_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
) {
    Conv2dParams dw_params = *params;
    dw_params.groups = params->in_channels;
    conv2d_forward(input, output, &dw_params);
}

void pointwise_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
) {
    Conv2dParams pw_params = *params;
    pw_params.kernel_h = 1;
    pw_params.kernel_w = 1;
    pw_params.stride_h = 1;
    pw_params.stride_w = 1;
    pw_params.padding_h = 0;
    pw_params.padding_w = 0;
    pw_params.dilation_h = 1;
    pw_params.dilation_w = 1;

    conv2d_forward(input, output, &pw_params);
}

void batch_norm_2d_forward(
    Tensor* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps
) {
    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int height = input->shape.height;
    int width = input->shape.width;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float mean = running_mean[c];
            float var = running_var[c];
            float std = sqrtf(var + eps);
            float scale = gamma[c];
            float shift = beta[c];

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * channels + c) * height + h) * width + w;
                    float normalized = (input->data[idx] - mean) / std;
                    input->data[idx] = scale * normalized + shift;
                }
            }
        }
    }
}

void prelu_forward(
    Tensor* input,
    const float* weight
) {
    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int height = input->shape.height;
    int width = input->shape.width;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            float alpha = weight[c];

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * channels + c) * height + h) * width + w;
                    if (input->data[idx] < 0) {
                        input->data[idx] *= alpha;
                    }
                }
            }
        }
    }
}

void tanh_forward(Tensor* input) {
    int total_size = input->shape.batch * input->shape.channels *
                     input->shape.height * input->shape.width;

    for (int i = 0; i < total_size; i++) {
        input->data[i] = tanhf(input->data[i]);
    }
}
