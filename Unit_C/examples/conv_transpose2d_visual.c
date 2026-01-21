#include "conv2d.h"
#include <stdio.h>
#include <stdlib.h>

/*
 * Visual demonstration of how ConvTranspose2d works
 * Shows step-by-step how input values spread to output
 */

void print_matrix(const char* name, float* data, int h, int w) {
    printf("\n%s (%dx%d):\n", name, h, w);
    for (int i = 0; i < h; i++) {
        printf("  ");
        for (int j = 0; j < w; j++) {
            printf("%6.2f ", data[i * w + j]);
        }
        printf("\n");
    }
}

void visualize_single_input_spread() {
    printf("\n");
    printf("=================================================================\n");
    printf("Visualization: How ONE input pixel spreads in ConvTranspose2d\n");
    printf("=================================================================\n");
    printf("Input: Single pixel with value 1.0\n");
    printf("Kernel: 3x3 with all values = 1.0\n");
    printf("Stride: 2, Padding: 0\n\n");

    // Create 1x1 input with single value
    Tensor* input = tensor_create(1, 1, 1, 1);
    input->data[0] = 1.0f;

    // Calculate output size: (1-1)*2 - 2*0 + 3 = 3
    int out_h = calculate_transpose_output_size(1, 3, 2, 0, 1);
    int out_w = calculate_transpose_output_size(1, 3, 2, 0, 1);
    printf("Output size: %dx%d\n", out_h, out_w);

    Tensor* output = tensor_create(1, 1, out_h, out_w);

    // Setup 3x3 kernel with all 1s
    Conv2dParams params;
    params.kernel_h = 3;
    params.kernel_w = 3;
    params.stride_h = 2;
    params.stride_w = 2;
    params.padding_h = 0;
    params.padding_w = 0;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = 1;
    params.in_channels = 1;
    params.out_channels = 1;
    params.use_bias = 0;

    params.weight = (float*)malloc(9 * sizeof(float));
    for (int i = 0; i < 9; i++) {
        params.weight[i] = 1.0f;
    }

    printf("\nKernel:\n");
    print_matrix("", params.weight, 3, 3);

    conv2d_transpose_forward(input, output, &params);

    printf("\nResult: The single input value spreads to a 3x3 output\n");
    printf("Each output position = input_value * corresponding_kernel_weight\n");
    print_matrix("Output", output->data, out_h, out_w);

    printf("\nExplanation:\n");
    printf("  The input pixel at (0,0) with value 1.0\n");
    printf("  spreads to output positions (0,0) through (2,2)\n");
    printf("  Each output[i,j] = input[0,0] * kernel[i,j] = 1.0 * 1.0 = 1.0\n");

    free(params.weight);
    tensor_free(input);
    tensor_free(output);
}

void visualize_stride_effect() {
    printf("\n");
    printf("=================================================================\n");
    printf("Visualization: Effect of Stride in ConvTranspose2d\n");
    printf("=================================================================\n");
    printf("Input: 2x2 with values [[1,2], [3,4]]\n");
    printf("Kernel: 2x2 with all values = 0.5\n");
    printf("Comparing stride=1 vs stride=2\n\n");

    // Create 2x2 input
    Tensor* input = tensor_create(1, 1, 2, 2);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;
    input->data[3] = 4.0f;

    print_matrix("Input", input->data, 2, 2);

    // Test stride=1
    {
        printf("\n--- Stride = 1 ---\n");
        int out_h = calculate_transpose_output_size(2, 2, 1, 0, 1);
        int out_w = calculate_transpose_output_size(2, 2, 1, 0, 1);
        printf("Output size: %dx%d\n", out_h, out_w);

        Tensor* output = tensor_create(1, 1, out_h, out_w);

        Conv2dParams params;
        params.kernel_h = 2;
        params.kernel_w = 2;
        params.stride_h = 1;
        params.stride_w = 1;
        params.padding_h = 0;
        params.padding_w = 0;
        params.dilation_h = 1;
        params.dilation_w = 1;
        params.groups = 1;
        params.in_channels = 1;
        params.out_channels = 1;
        params.use_bias = 0;

        params.weight = (float*)malloc(4 * sizeof(float));
        for (int i = 0; i < 4; i++) {
            params.weight[i] = 0.5f;
        }

        conv2d_transpose_forward(input, output, &params);
        print_matrix("Output", output->data, out_h, out_w);

        printf("\nNote: Adjacent input pixels create OVERLAPPING outputs\n");
        printf("      Output values are SUMMED where they overlap\n");

        free(params.weight);
        tensor_free(output);
    }

    // Test stride=2
    {
        printf("\n--- Stride = 2 ---\n");
        int out_h = calculate_transpose_output_size(2, 2, 2, 0, 1);
        int out_w = calculate_transpose_output_size(2, 2, 2, 0, 1);
        printf("Output size: %dx%d\n", out_h, out_w);

        Tensor* output = tensor_create(1, 1, out_h, out_w);

        Conv2dParams params;
        params.kernel_h = 2;
        params.kernel_w = 2;
        params.stride_h = 2;
        params.stride_w = 2;
        params.padding_h = 0;
        params.padding_w = 0;
        params.dilation_h = 1;
        params.dilation_w = 1;
        params.groups = 1;
        params.in_channels = 1;
        params.out_channels = 1;
        params.use_bias = 0;

        params.weight = (float*)malloc(4 * sizeof(float));
        for (int i = 0; i < 4; i++) {
            params.weight[i] = 0.5f;
        }

        conv2d_transpose_forward(input, output, &params);
        print_matrix("Output", output->data, out_h, out_w);

        printf("\nNote: With stride=2, outputs do NOT overlap\n");
        printf("      Each 2x2 block comes from a single input pixel\n");

        free(params.weight);
        tensor_free(output);
    }

    tensor_free(input);
}

void visualize_gtcrn_frequency_upsampling() {
    printf("\n");
    printf("=================================================================\n");
    printf("Visualization: GTCRN Frequency Upsampling\n");
    printf("=================================================================\n");
    printf("Simulating how GTCRN decoder restores frequency resolution\n\n");

    // Simulate compressed frequency representation
    printf("Encoder output: 97 frequency bins (compressed)\n");
    printf("Decoder must restore to: 385 frequency bins (original)\n\n");

    // First upsampling stage: 97 -> 194
    {
        printf("Stage 1: 97 bins -> 194 bins (2x upsampling)\n");
        printf("  ConvTranspose2d(16, 16, kernel=(1,5), stride=(1,2), padding=(0,2), groups=2)\n");

        int in_w = 97;
        int out_w = calculate_transpose_output_size(in_w, 5, 2, 2, 1);
        printf("  Calculated output: %d bins\n", out_w);

        // Create small example with 4 frequency bins -> 8 bins
        Tensor* input = tensor_create(1, 1, 1, 4);
        for (int i = 0; i < 4; i++) {
            input->data[i] = (float)(i + 1);
        }

        int example_out = calculate_transpose_output_size(4, 5, 2, 2, 1);
        Tensor* output = tensor_create(1, 1, 1, example_out);

        Conv2dParams params;
        params.kernel_h = 1;
        params.kernel_w = 5;
        params.stride_h = 1;
        params.stride_w = 2;
        params.padding_h = 0;
        params.padding_w = 2;
        params.dilation_h = 1;
        params.dilation_w = 1;
        params.groups = 1;
        params.in_channels = 1;
        params.out_channels = 1;
        params.use_bias = 0;

        params.weight = (float*)malloc(5 * sizeof(float));
        // Interpolation-like weights
        params.weight[0] = 0.1f;
        params.weight[1] = 0.2f;
        params.weight[2] = 0.4f;
        params.weight[3] = 0.2f;
        params.weight[4] = 0.1f;

        conv2d_transpose_forward(input, output, &params);

        printf("\n  Example (4 bins -> %d bins):\n", example_out);
        printf("  Input:  ");
        for (int i = 0; i < 4; i++) {
            printf("%.1f ", input->data[i]);
        }
        printf("\n  Output: ");
        for (int i = 0; i < example_out; i++) {
            printf("%.2f ", output->data[i]);
        }
        printf("\n");

        free(params.weight);
        tensor_free(input);
        tensor_free(output);
    }

    // Second upsampling stage: 194 -> 385
    {
        printf("\nStage 2: 194 bins -> 385 bins (~2x upsampling)\n");
        printf("  ConvTranspose2d(16, 2, kernel=(1,5), stride=(1,2), padding=(0,2))\n");

        int in_w = 194;
        int out_w = calculate_transpose_output_size(in_w, 5, 2, 2, 1);
        printf("  Calculated output: %d bins\n", out_w);
        printf("  Output channels: 2 (real + imaginary mask)\n");
    }

    printf("\nResult: Full frequency resolution restored!\n");
    printf("  97 -> 194 -> 385 frequency bins\n");
    printf("  Ready to apply complex mask to input spectrogram\n");
}

void visualize_step_by_step() {
    printf("\n");
    printf("=================================================================\n");
    printf("Step-by-Step: ConvTranspose2d Computation\n");
    printf("=================================================================\n");
    printf("Input: 2x2, Kernel: 2x2, Stride: 2, Padding: 0\n\n");

    Tensor* input = tensor_create(1, 1, 2, 2);
    input->data[0] = 1.0f;  // Top-left
    input->data[1] = 2.0f;  // Top-right
    input->data[2] = 3.0f;  // Bottom-left
    input->data[3] = 4.0f;  // Bottom-right

    float kernel[4] = {0.1f, 0.2f, 0.3f, 0.4f};

    printf("Input:\n");
    printf("  ┌─────┬─────┐\n");
    printf("  │ 1.0 │ 2.0 │\n");
    printf("  ├─────┼─────┤\n");
    printf("  │ 3.0 │ 4.0 │\n");
    printf("  └─────┴─────┘\n");

    printf("\nKernel:\n");
    printf("  ┌─────┬─────┐\n");
    printf("  │ 0.1 │ 0.2 │\n");
    printf("  ├─────┼─────┤\n");
    printf("  │ 0.3 │ 0.4 │\n");
    printf("  └─────┴─────┘\n");

    int out_h = calculate_transpose_output_size(2, 2, 2, 0, 1);
    int out_w = calculate_transpose_output_size(2, 2, 2, 0, 1);

    printf("\nOutput size: %dx%d\n", out_h, out_w);

    printf("\nStep-by-step computation:\n");
    printf("\n1. Input[0,0]=1.0 at position (0,0):\n");
    printf("   Output position = (0,0)*stride + kernel_offset = (0,0) + (kh,kw)\n");
    printf("   Output[0,0] += 1.0 * 0.1 = 0.1\n");
    printf("   Output[0,1] += 1.0 * 0.2 = 0.2\n");
    printf("   Output[1,0] += 1.0 * 0.3 = 0.3\n");
    printf("   Output[1,1] += 1.0 * 0.4 = 0.4\n");

    printf("\n2. Input[0,1]=2.0 at position (0,1):\n");
    printf("   Output position = (0,1)*stride + kernel_offset = (0,2) + (kh,kw)\n");
    printf("   Output[0,2] += 2.0 * 0.1 = 0.2\n");
    printf("   Output[0,3] += 2.0 * 0.2 = 0.4\n");
    printf("   Output[1,2] += 2.0 * 0.3 = 0.6\n");
    printf("   Output[1,3] += 2.0 * 0.4 = 0.8\n");

    printf("\n3. Input[1,0]=3.0 at position (1,0):\n");
    printf("   Output position = (1,0)*stride + kernel_offset = (2,0) + (kh,kw)\n");
    printf("   Output[2,0] += 3.0 * 0.1 = 0.3\n");
    printf("   Output[2,1] += 3.0 * 0.2 = 0.6\n");
    printf("   Output[3,0] += 3.0 * 0.3 = 0.9\n");
    printf("   Output[3,1] += 3.0 * 0.4 = 1.2\n");

    printf("\n4. Input[1,1]=4.0 at position (1,1):\n");
    printf("   Output position = (1,1)*stride + kernel_offset = (2,2) + (kh,kw)\n");
    printf("   Output[2,2] += 4.0 * 0.1 = 0.4\n");
    printf("   Output[2,3] += 4.0 * 0.2 = 0.8\n");
    printf("   Output[3,2] += 4.0 * 0.3 = 1.2\n");
    printf("   Output[3,3] += 4.0 * 0.4 = 1.6\n");

    // Actually compute it
    Tensor* output = tensor_create(1, 1, out_h, out_w);
    Conv2dParams params;
    params.kernel_h = 2;
    params.kernel_w = 2;
    params.stride_h = 2;
    params.stride_w = 2;
    params.padding_h = 0;
    params.padding_w = 0;
    params.dilation_h = 1;
    params.dilation_w = 1;
    params.groups = 1;
    params.in_channels = 1;
    params.out_channels = 1;
    params.use_bias = 0;
    params.weight = kernel;

    conv2d_transpose_forward(input, output, &params);

    printf("\nFinal Output:\n");
    print_matrix("", output->data, out_h, out_w);

    printf("\nVisualization:\n");
    printf("  ┌──────┬──────┬──────┬──────┐\n");
    printf("  │ 0.10 │ 0.20 │ 0.20 │ 0.40 │  <- From input[0,0] and input[0,1]\n");
    printf("  ├──────┼──────┼──────┼──────┤\n");
    printf("  │ 0.30 │ 0.40 │ 0.60 │ 0.80 │  <- From input[0,0] and input[0,1]\n");
    printf("  ├──────┼──────┼──────┼──────┤\n");
    printf("  │ 0.30 │ 0.60 │ 0.40 │ 0.80 │  <- From input[1,0] and input[1,1]\n");
    printf("  ├──────┼──────┼──────┼──────┤\n");
    printf("  │ 0.90 │ 1.20 │ 1.20 │ 1.60 │  <- From input[1,0] and input[1,1]\n");
    printf("  └──────┴──────┴──────┴──────┘\n");

    tensor_free(input);
    tensor_free(output);
}

int main() {
    printf("\n");
    printf("#################################################################\n");
    printf("#                                                               #\n");
    printf("#  ConvTranspose2d Visual Demonstrations                        #\n");
    printf("#  Understanding How Transposed Convolution Works               #\n");
    printf("#                                                               #\n");
    printf("#################################################################\n");

    visualize_single_input_spread();
    visualize_stride_effect();
    visualize_step_by_step();
    visualize_gtcrn_frequency_upsampling();

    printf("\n");
    printf("=================================================================\n");
    printf("Key Takeaways:\n");
    printf("=================================================================\n");
    printf("1. Each INPUT pixel spreads to MULTIPLE output pixels\n");
    printf("2. Stride controls SPACING between output blocks\n");
    printf("3. When stride < kernel, outputs OVERLAP and SUM\n");
    printf("4. When stride >= kernel, outputs are SEPARATE\n");
    printf("5. In GTCRN: Used to restore frequency resolution in decoder\n");
    printf("\n");

    return 0;
}
