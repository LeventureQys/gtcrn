#include "conv2d.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_tensor_info(const char* name, const Tensor* tensor) {
    printf("%s shape: [%d, %d, %d, %d]\n",
           name,
           tensor->shape.batch,
           tensor->shape.channels,
           tensor->shape.height,
           tensor->shape.width);
}

void print_tensor_sample(const char* name, const Tensor* tensor, int max_samples) {
    printf("%s sample values: ", name);
    int total = tensor->shape.batch * tensor->shape.channels *
                tensor->shape.height * tensor->shape.width;
    int samples = (total < max_samples) ? total : max_samples;

    for (int i = 0; i < samples; i++) {
        printf("%.4f ", tensor->data[i]);
    }
    printf("\n");
}

void test_regular_conv2d() {
    printf("\n=== Testing Regular Conv2d ===\n");

    int batch = 1;
    int in_channels = 3;
    int in_height = 32;
    int in_width = 32;
    int out_channels = 16;
    int kernel_h = 3;
    int kernel_w = 3;
    int stride = 1;
    int padding = 1;

    Tensor* input = tensor_create(batch, in_channels, in_height, in_width);
    if (!input) {
        printf("Failed to create input tensor\n");
        return;
    }

    for (int i = 0; i < batch * in_channels * in_height * in_width; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    int out_h = calculate_output_size(in_height, kernel_h, stride, padding, 1);
    int out_w = calculate_output_size(in_width, kernel_w, stride, padding, 1);

    Tensor* output = tensor_create(batch, out_channels, out_h, out_w);
    if (!output) {
        printf("Failed to create output tensor\n");
        tensor_free(input);
        return;
    }

    Conv2dParams params;
    params.kernel_h = kernel_h;
    params.kernel_w = kernel_w;
    params.stride_h = stride;
    params.stride_w = stride;
    params.padding_h = padding;
    params.padding_w = padding;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = 1;
    params.in_channels = in_channels;
    params.out_channels = out_channels;
    params.use_bias = 1;

    int weight_size = out_channels * in_channels * kernel_h * kernel_w;
    params.weight = (float*)malloc(weight_size * sizeof(float));
    params.bias = (float*)malloc(out_channels * sizeof(float));

    for (int i = 0; i < weight_size; i++) {
        params.weight[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < out_channels; i++) {
        params.bias[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    clock_t start = clock();
    conv2d_forward(input, output, &params);
    clock_t end = clock();

    print_tensor_info("Input", input);
    print_tensor_info("Output", output);
    print_tensor_sample("Output", output, 10);
    printf("Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    free(params.weight);
    free(params.bias);
    tensor_free(input);
    tensor_free(output);
}

void test_depthwise_conv2d() {
    printf("\n=== Testing Depthwise Conv2d ===\n");

    int batch = 1;
    int channels = 16;
    int in_height = 32;
    int in_width = 32;
    int kernel_h = 3;
    int kernel_w = 3;
    int stride = 1;
    int padding = 1;

    Tensor* input = tensor_create(batch, channels, in_height, in_width);
    if (!input) {
        printf("Failed to create input tensor\n");
        return;
    }

    for (int i = 0; i < batch * channels * in_height * in_width; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    int out_h = calculate_output_size(in_height, kernel_h, stride, padding, 1);
    int out_w = calculate_output_size(in_width, kernel_w, stride, padding, 1);

    Tensor* output = tensor_create(batch, channels, out_h, out_w);
    if (!output) {
        printf("Failed to create output tensor\n");
        tensor_free(input);
        return;
    }

    Conv2dParams params;
    params.kernel_h = kernel_h;
    params.kernel_w = kernel_w;
    params.stride_h = stride;
    params.stride_w = stride;
    params.padding_h = padding;
    params.padding_w = padding;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = channels;
    params.in_channels = channels;
    params.out_channels = channels;
    params.use_bias = 0;

    int weight_size = channels * 1 * kernel_h * kernel_w;
    params.weight = (float*)malloc(weight_size * sizeof(float));

    for (int i = 0; i < weight_size; i++) {
        params.weight[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    clock_t start = clock();
    depthwise_conv2d_forward(input, output, &params);
    clock_t end = clock();

    print_tensor_info("Input", input);
    print_tensor_info("Output", output);
    print_tensor_sample("Output", output, 10);
    printf("Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    free(params.weight);
    tensor_free(input);
    tensor_free(output);
}

void test_pointwise_conv2d() {
    printf("\n=== Testing Pointwise Conv2d (1x1) ===\n");

    int batch = 1;
    int in_channels = 16;
    int out_channels = 32;
    int in_height = 32;
    int in_width = 32;

    Tensor* input = tensor_create(batch, in_channels, in_height, in_width);
    if (!input) {
        printf("Failed to create input tensor\n");
        return;
    }

    for (int i = 0; i < batch * in_channels * in_height * in_width; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    Tensor* output = tensor_create(batch, out_channels, in_height, in_width);
    if (!output) {
        printf("Failed to create output tensor\n");
        tensor_free(input);
        return;
    }

    Conv2dParams params;
    params.kernel_h = 1;
    params.kernel_w = 1;
    params.stride_h = 1;
    params.stride_w = 1;
    params.padding_h = 0;
    params.padding_w = 0;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = 1;
    params.in_channels = in_channels;
    params.out_channels = out_channels;
    params.use_bias = 1;

    int weight_size = out_channels * in_channels * 1 * 1;
    params.weight = (float*)malloc(weight_size * sizeof(float));
    params.bias = (float*)malloc(out_channels * sizeof(float));

    for (int i = 0; i < weight_size; i++) {
        params.weight[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < out_channels; i++) {
        params.bias[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    clock_t start = clock();
    pointwise_conv2d_forward(input, output, &params);
    clock_t end = clock();

    print_tensor_info("Input", input);
    print_tensor_info("Output", output);
    print_tensor_sample("Output", output, 10);
    printf("Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    free(params.weight);
    free(params.bias);
    tensor_free(input);
    tensor_free(output);
}

void test_conv2d_transpose() {
    printf("\n=== Testing Conv2d Transpose (Deconvolution) ===\n");

    int batch = 1;
    int in_channels = 16;
    int out_channels = 8;
    int in_height = 16;
    int in_width = 16;
    int kernel_h = 4;
    int kernel_w = 4;
    int stride = 2;
    int padding = 1;

    Tensor* input = tensor_create(batch, in_channels, in_height, in_width);
    if (!input) {
        printf("Failed to create input tensor\n");
        return;
    }

    for (int i = 0; i < batch * in_channels * in_height * in_width; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    int out_h = calculate_transpose_output_size(in_height, kernel_h, stride, padding, 1);
    int out_w = calculate_transpose_output_size(in_width, kernel_w, stride, padding, 1);

    Tensor* output = tensor_create(batch, out_channels, out_h, out_w);
    if (!output) {
        printf("Failed to create output tensor\n");
        tensor_free(input);
        return;
    }

    Conv2dParams params;
    params.kernel_h = kernel_h;
    params.kernel_w = kernel_w;
    params.stride_h = stride;
    params.stride_w = stride;
    params.padding_h = padding;
    params.padding_w = padding;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = 1;
    params.in_channels = in_channels;
    params.out_channels = out_channels;
    params.use_bias = 1;

    int weight_size = in_channels * out_channels * kernel_h * kernel_w;
    params.weight = (float*)malloc(weight_size * sizeof(float));
    params.bias = (float*)malloc(out_channels * sizeof(float));

    for (int i = 0; i < weight_size; i++) {
        params.weight[i] = (float)rand() / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < out_channels; i++) {
        params.bias[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    clock_t start = clock();
    conv2d_transpose_forward(input, output, &params);
    clock_t end = clock();

    print_tensor_info("Input", input);
    print_tensor_info("Output", output);
    print_tensor_sample("Output", output, 10);
    printf("Time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    free(params.weight);
    free(params.bias);
    tensor_free(input);
    tensor_free(output);
}

void test_batch_norm_and_activation() {
    printf("\n=== Testing BatchNorm + PReLU + Tanh ===\n");

    int batch = 1;
    int channels = 16;
    int height = 32;
    int width = 32;

    Tensor* input = tensor_create(batch, channels, height, width);
    if (!input) {
        printf("Failed to create input tensor\n");
        return;
    }

    for (int i = 0; i < batch * channels * height * width; i++) {
        input->data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    float* gamma = (float*)malloc(channels * sizeof(float));
    float* beta = (float*)malloc(channels * sizeof(float));
    float* running_mean = (float*)malloc(channels * sizeof(float));
    float* running_var = (float*)malloc(channels * sizeof(float));
    float* prelu_weight = (float*)malloc(channels * sizeof(float));

    for (int i = 0; i < channels; i++) {
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
        running_mean[i] = 0.0f;
        running_var[i] = 1.0f;
        prelu_weight[i] = 0.25f;
    }

    printf("Before BatchNorm: ");
    print_tensor_sample("", input, 5);

    batch_norm_2d_forward(input, gamma, beta, running_mean, running_var, 1e-5f);
    printf("After BatchNorm: ");
    print_tensor_sample("", input, 5);

    prelu_forward(input, prelu_weight);
    printf("After PReLU: ");
    print_tensor_sample("", input, 5);

    tanh_forward(input);
    printf("After Tanh: ");
    print_tensor_sample("", input, 5);

    free(gamma);
    free(beta);
    free(running_mean);
    free(running_var);
    free(prelu_weight);
    tensor_free(input);
}

int main() {
    srand(time(NULL));

    printf("GTCRN Conv2d Implementation Test Suite\n");
    printf("========================================\n");

    test_regular_conv2d();
    test_depthwise_conv2d();
    test_pointwise_conv2d();
    test_conv2d_transpose();
    test_batch_norm_and_activation();

    printf("\n=== All tests completed ===\n");

    return 0;
}
