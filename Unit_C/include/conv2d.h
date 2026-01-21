#ifndef CONV2D_H
#define CONV2D_H

#include <stdint.h>

typedef struct {
    int batch;
    int channels;
    int height;
    int width;
} TensorShape;

typedef struct {
    float* data;
    TensorShape shape;
} Tensor;

typedef struct {
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int dilation_h;
    int dilation_w;
    int groups;
    int in_channels;
    int out_channels;
    float* weight;
    float* bias;
    int use_bias;
} Conv2dParams;

Tensor* tensor_create(int batch, int channels, int height, int width);
void tensor_free(Tensor* tensor);

void conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);

void conv2d_transpose_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);

void depthwise_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);

void pointwise_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    const Conv2dParams* params
);

void batch_norm_2d_forward(
    Tensor* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps
);

void prelu_forward(
    Tensor* input,
    const float* weight
);

void tanh_forward(Tensor* input);

int calculate_output_size(
    int input_size,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

int calculate_transpose_output_size(
    int input_size,
    int kernel_size,
    int stride,
    int padding,
    int dilation
);

#endif
