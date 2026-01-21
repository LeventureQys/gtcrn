#include "conv2d.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * ConvTranspose2d (Transposed Convolution / Deconvolution)
 *
 * This is the C implementation of PyTorch's nn.ConvTranspose2d
 * Used in GTCRN's decoder for upsampling feature maps
 *
 * Key differences from regular Conv2d:
 * 1. Upsamples the input (increases spatial dimensions)
 * 2. Output size = (input_size - 1) * stride - 2*padding + kernel_size
 * 3. Each input pixel spreads its value across multiple output pixels
 */

void print_tensor_2d(const char* name, const Tensor* tensor) {
    printf("\n%s (shape: [%d, %d, %d, %d]):\n",
           name,
           tensor->shape.batch,
           tensor->shape.channels,
           tensor->shape.height,
           tensor->shape.width);

    // Print first channel of first batch
    printf("Channel 0:\n");
    for (int h = 0; h < tensor->shape.height && h < 8; h++) {
        for (int w = 0; w < tensor->shape.width && w < 8; w++) {
            int idx = h * tensor->shape.width + w;
            printf("%6.2f ", tensor->data[idx]);
        }
        printf("\n");
    }
}

void example_1_basic_upsample() {
    printf("\n");
    printf("=================================================================\n");
    printf("Example 1: Basic ConvTranspose2d - 2x Upsampling\n");
    printf("=================================================================\n");
    printf("PyTorch equivalent:\n");
    printf("  nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)\n");
    printf("  Input: [1, 1, 4, 4] -> Output: [1, 1, 8, 8]\n\n");

    // Create small input
    Tensor* input = tensor_create(1, 1, 4, 4);

    // Fill with simple pattern
    for (int h = 0; h < 4; h++) {
        for (int w = 0; w < 4; w++) {
            input->data[h * 4 + w] = (float)(h * 4 + w + 1);
        }
    }

    // Calculate output size
    int out_h = calculate_transpose_output_size(4, 4, 2, 1, 1);
    int out_w = calculate_transpose_output_size(4, 4, 2, 1, 1);
    printf("Calculated output size: %d x %d\n", out_h, out_w);

    Tensor* output = tensor_create(1, 1, out_h, out_w);

    // Setup parameters
    Conv2dParams params;
    params.kernel_h = 4;
    params.kernel_w = 4;
    params.stride_h = 2;
    params.stride_w = 2;
    params.padding_h = 1;
    params.padding_w = 1;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = 1;
    params.in_channels = 1;
    params.out_channels = 1;
    params.use_bias = 0;

    // Simple weights (all 0.25 for averaging)
    int weight_size = 1 * 1 * 4 * 4;
    params.weight = (float*)malloc(weight_size * sizeof(float));
    for (int i = 0; i < weight_size; i++) {
        params.weight[i] = 0.25f;
    }

    // Perform transpose convolution
    conv2d_transpose_forward(input, output, &params);

    print_tensor_2d("Input", input);
    print_tensor_2d("Output (upsampled 2x)", output);

    free(params.weight);
    tensor_free(input);
    tensor_free(output);
}

void example_2_gtcrn_decoder_block() {
    printf("\n");
    printf("=================================================================\n");
    printf("Example 2: GTCRN Decoder ConvTranspose2d Block\n");
    printf("=================================================================\n");
    printf("From gtcrn1.py line 254:\n");
    printf("  ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), \n");
    printf("            groups=2, use_deconv=True)\n");
    printf("  Input: [1, 16, 63, 97] -> Output: [1, 16, 63, 194]\n\n");

    int batch = 1;
    int channels = 16;
    int in_h = 63;
    int in_w = 97;
    int kernel_h = 1;
    int kernel_w = 5;
    int stride_h = 1;
    int stride_w = 2;
    int padding_h = 0;
    int padding_w = 2;
    int groups = 2;

    Tensor* input = tensor_create(batch, channels, in_h, in_w);

    // Random initialization
    srand(42);
    for (int i = 0; i < batch * channels * in_h * in_w; i++) {
        input->data[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Calculate output size
    int out_h = calculate_transpose_output_size(in_h, kernel_h, stride_h, padding_h, 1);
    int out_w = calculate_transpose_output_size(in_w, kernel_w, stride_w, padding_w, 1);

    printf("Input shape: [%d, %d, %d, %d]\n", batch, channels, in_h, in_w);
    printf("Output shape: [%d, %d, %d, %d]\n", batch, channels, out_h, out_w);
    printf("Kernel: (%d, %d), Stride: (%d, %d), Padding: (%d, %d), Groups: %d\n",
           kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups);

    Tensor* output = tensor_create(batch, channels, out_h, out_w);

    // Setup parameters
    Conv2dParams params;
    params.kernel_h = kernel_h;
    params.kernel_w = kernel_w;
    params.stride_h = stride_h;
    params.stride_w = stride_w;
    params.padding_h = padding_h;
    params.padding_w = padding_w;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = groups;
    params.in_channels = channels;
    params.out_channels = channels;
    params.use_bias = 1;

    // Allocate weights and bias
    int channels_per_group = channels / groups;
    int weight_size = channels * channels_per_group * kernel_h * kernel_w;
    params.weight = (float*)malloc(weight_size * sizeof(float));
    params.bias = (float*)malloc(channels * sizeof(float));

    // Initialize with random values
    for (int i = 0; i < weight_size; i++) {
        params.weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < channels; i++) {
        params.bias[i] = 0.0f;
    }

    // Perform transpose convolution
    clock_t start = clock();
    conv2d_transpose_forward(input, output, &params);
    clock_t end = clock();

    printf("\nExecution time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // Print statistics
    float min_val = output->data[0], max_val = output->data[0];
    double sum = 0.0;
    int total = batch * channels * out_h * out_w;

    for (int i = 0; i < total; i++) {
        float val = output->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    printf("Output statistics:\n");
    printf("  Min: %.6f\n", min_val);
    printf("  Max: %.6f\n", max_val);
    printf("  Mean: %.6f\n", sum / total);

    free(params.weight);
    free(params.bias);
    tensor_free(input);
    tensor_free(output);
}

void example_3_final_decoder_layer() {
    printf("\n");
    printf("=================================================================\n");
    printf("Example 3: GTCRN Final Decoder Layer\n");
    printf("=================================================================\n");
    printf("From gtcrn1.py line 255:\n");
    printf("  ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), \n");
    printf("            use_deconv=True, is_last=True)\n");
    printf("  This layer outputs the complex mask (real + imaginary)\n");
    printf("  Input: [1, 16, 63, 194] -> Output: [1, 2, 63, 385]\n\n");

    int batch = 1;
    int in_channels = 16;
    int out_channels = 2;  // Real and imaginary parts
    int in_h = 63;
    int in_w = 194;
    int kernel_h = 1;
    int kernel_w = 5;
    int stride_h = 1;
    int stride_w = 2;
    int padding_h = 0;
    int padding_w = 2;

    Tensor* input = tensor_create(batch, in_channels, in_h, in_w);

    // Random initialization
    srand(123);
    for (int i = 0; i < batch * in_channels * in_h * in_w; i++) {
        input->data[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Calculate output size
    int out_h = calculate_transpose_output_size(in_h, kernel_h, stride_h, padding_h, 1);
    int out_w = calculate_transpose_output_size(in_w, kernel_w, stride_w, padding_w, 1);

    printf("Input shape: [%d, %d, %d, %d]\n", batch, in_channels, in_h, in_w);
    printf("Output shape: [%d, %d, %d, %d]\n", batch, out_channels, out_h, out_w);
    printf("This produces the complex ratio mask for speech enhancement\n");

    Tensor* output = tensor_create(batch, out_channels, out_h, out_w);

    // Setup parameters
    Conv2dParams params;
    params.kernel_h = kernel_h;
    params.kernel_w = kernel_w;
    params.stride_h = stride_h;
    params.stride_w = stride_w;
    params.padding_h = padding_h;
    params.padding_w = padding_w;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = 1;
    params.in_channels = in_channels;
    params.out_channels = out_channels;
    params.use_bias = 1;

    // Allocate weights and bias
    int weight_size = out_channels * in_channels * kernel_h * kernel_w;
    params.weight = (float*)malloc(weight_size * sizeof(float));
    params.bias = (float*)malloc(out_channels * sizeof(float));

    // Initialize
    for (int i = 0; i < weight_size; i++) {
        params.weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < out_channels; i++) {
        params.bias[i] = 0.0f;
    }

    // Perform transpose convolution
    clock_t start = clock();
    conv2d_transpose_forward(input, output, &params);
    clock_t end = clock();

    // Apply Tanh activation (is_last=True uses Tanh)
    tanh_forward(output);

    printf("\nExecution time: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // Print sample values from both channels
    printf("\nSample output values (after Tanh):\n");
    printf("Channel 0 (Real mask): ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", output->data[i]);
    }
    printf("\nChannel 1 (Imag mask): ");
    int offset = out_h * out_w;
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", output->data[offset + i]);
    }
    printf("\n");

    free(params.weight);
    free(params.bias);
    tensor_free(input);
    tensor_free(output);
}

void example_4_stride_comparison() {
    printf("\n");
    printf("=================================================================\n");
    printf("Example 4: Understanding Stride in ConvTranspose2d\n");
    printf("=================================================================\n");
    printf("Comparing different stride values for upsampling\n\n");

    int strides[] = {1, 2, 3, 4};
    int num_strides = 4;

    for (int s = 0; s < num_strides; s++) {
        int stride = strides[s];

        Tensor* input = tensor_create(1, 1, 4, 4);
        for (int i = 0; i < 16; i++) {
            input->data[i] = 1.0f;
        }

        int out_h = calculate_transpose_output_size(4, 3, stride, 1, 1);
        int out_w = calculate_transpose_output_size(4, 3, stride, 1, 1);

        printf("Stride=%d: Input [1,1,4,4] -> Output [1,1,%d,%d] (%.1fx upsampling)\n",
               stride, out_h, out_w, (float)out_h / 4.0f);

        tensor_free(input);
    }
}

int main() {
    printf("\n");
    printf("#################################################################\n");
    printf("#                                                               #\n");
    printf("#  ConvTranspose2d Implementation for GTCRN                     #\n");
    printf("#  C Implementation of PyTorch nn.ConvTranspose2d               #\n");
    printf("#                                                               #\n");
    printf("#################################################################\n");

    example_1_basic_upsample();
    example_2_gtcrn_decoder_block();
    example_3_final_decoder_layer();
    example_4_stride_comparison();

    printf("\n");
    printf("=================================================================\n");
    printf("Key Points about ConvTranspose2d:\n");
    printf("=================================================================\n");
    printf("1. Used for upsampling in decoder (opposite of Conv2d)\n");
    printf("2. Output size = (input_size - 1) * stride - 2*padding + kernel\n");
    printf("3. Each input value spreads to multiple output locations\n");
    printf("4. Supports groups for efficient channel-wise operations\n");
    printf("5. In GTCRN: Used to restore frequency resolution in decoder\n");
    printf("\n");
    printf("GTCRN Decoder uses ConvTranspose2d at:\n");
    printf("  - Line 254: 16->16 channels, groups=2, stride=(1,2)\n");
    printf("  - Line 255: 16->2 channels, final layer with Tanh\n");
    printf("\n");

    return 0;
}
