/**
 * test_dpgrnn.c - Test program for DPGRNN implementation
 *
 * This program tests the complete DPGRNN forward pass including:
 * - Intra-RNN (Bidirectional GRNN)
 * - Inter-RNN (Unidirectional GRNN)
 * - Linear layers
 * - LayerNorm
 * - Residual connections
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gtcrn_model.h"
#include "GRU.h"

// 辅助函数: 初始化随机权重
void init_random_weights(float* weights, int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
}

// 辅助函数: 打印张量统计信息
void print_tensor_stats(const char* name, const float* data, int size) {
    float min_val = data[0];
    float max_val = data[0];
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
    }

    float mean = sum / size;

    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = data[i] - mean;
        var += diff * diff;
    }
    var /= size;
    float std = sqrtf(var);

    printf("%s: min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n",
           name, min_val, max_val, mean, std);
}

// 测试1: 基本的DPGRNN前向传播
void test_dpgrnn_forward() {
    printf("\n========================================\n");
    printf("Test 1: DPGRNN Forward Pass\n");
    printf("========================================\n\n");

    // 参数设置
    int B = 1;      // Batch size
    int C = 16;     // Channels
    int T = 10;     // Time steps
    int F = 97;     // Frequency bins

    printf("Input shape: (B=%d, C=%d, T=%d, F=%d)\n", B, C, T, F);
    printf("Total elements: %d\n\n", B * C * T * F);

    // 创建DPGRNN
    DPGRNN* dpgrnn = dpgrnn_create(C, F, C);
    if (!dpgrnn) {
        printf("Failed to create DPGRNN\n");
        return;
    }

    // 初始化GRU权重 (简化测试，使用随机权重)
    printf("Initializing GRU weights...\n");

    // Intra-RNN weights
    init_random_weights(dpgrnn->intra_gru_g1_fwd->W_z,
                       dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->U_z,
                       dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->b_z,
                       dpgrnn->intra_gru_g1_fwd->hidden_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->W_r,
                       dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->U_r,
                       dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->b_r,
                       dpgrnn->intra_gru_g1_fwd->hidden_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->W_h,
                       dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->U_h,
                       dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size);
    init_random_weights(dpgrnn->intra_gru_g1_fwd->b_h,
                       dpgrnn->intra_gru_g1_fwd->hidden_size);

    // 为简化，只初始化group 1，group 2使用相同的权重
    memcpy(dpgrnn->intra_gru_g2_fwd->W_z, dpgrnn->intra_gru_g1_fwd->W_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->U_z, dpgrnn->intra_gru_g1_fwd->U_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->b_z, dpgrnn->intra_gru_g1_fwd->b_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->W_r, dpgrnn->intra_gru_g1_fwd->W_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->U_r, dpgrnn->intra_gru_g1_fwd->U_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->b_r, dpgrnn->intra_gru_g1_fwd->b_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->W_h, dpgrnn->intra_gru_g1_fwd->W_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->U_h, dpgrnn->intra_gru_g1_fwd->U_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_fwd->b_h, dpgrnn->intra_gru_g1_fwd->b_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));

    // Backward weights (same as forward for simplicity)
    memcpy(dpgrnn->intra_gru_g1_bwd->W_z, dpgrnn->intra_gru_g1_fwd->W_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->U_z, dpgrnn->intra_gru_g1_fwd->U_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->b_z, dpgrnn->intra_gru_g1_fwd->b_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->W_r, dpgrnn->intra_gru_g1_fwd->W_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->U_r, dpgrnn->intra_gru_g1_fwd->U_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->b_r, dpgrnn->intra_gru_g1_fwd->b_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->W_h, dpgrnn->intra_gru_g1_fwd->W_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->U_h, dpgrnn->intra_gru_g1_fwd->U_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g1_bwd->b_h, dpgrnn->intra_gru_g1_fwd->b_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));

    memcpy(dpgrnn->intra_gru_g2_bwd->W_z, dpgrnn->intra_gru_g1_fwd->W_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->U_z, dpgrnn->intra_gru_g1_fwd->U_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->b_z, dpgrnn->intra_gru_g1_fwd->b_z,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->W_r, dpgrnn->intra_gru_g1_fwd->W_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->U_r, dpgrnn->intra_gru_g1_fwd->U_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->b_r, dpgrnn->intra_gru_g1_fwd->b_r,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->W_h, dpgrnn->intra_gru_g1_fwd->W_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->input_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->U_h, dpgrnn->intra_gru_g1_fwd->U_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));
    memcpy(dpgrnn->intra_gru_g2_bwd->b_h, dpgrnn->intra_gru_g1_fwd->b_h,
           dpgrnn->intra_gru_g1_fwd->hidden_size * sizeof(float));

    // Inter-RNN weights
    init_random_weights(dpgrnn->inter_gru_g1->W_z,
                       dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->input_size);
    init_random_weights(dpgrnn->inter_gru_g1->U_z,
                       dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->hidden_size);
    init_random_weights(dpgrnn->inter_gru_g1->b_z,
                       dpgrnn->inter_gru_g1->hidden_size);
    init_random_weights(dpgrnn->inter_gru_g1->W_r,
                       dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->input_size);
    init_random_weights(dpgrnn->inter_gru_g1->U_r,
                       dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->hidden_size);
    init_random_weights(dpgrnn->inter_gru_g1->b_r,
                       dpgrnn->inter_gru_g1->hidden_size);
    init_random_weights(dpgrnn->inter_gru_g1->W_h,
                       dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->input_size);
    init_random_weights(dpgrnn->inter_gru_g1->U_h,
                       dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->hidden_size);
    init_random_weights(dpgrnn->inter_gru_g1->b_h,
                       dpgrnn->inter_gru_g1->hidden_size);

    memcpy(dpgrnn->inter_gru_g2->W_z, dpgrnn->inter_gru_g1->W_z,
           dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->input_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->U_z, dpgrnn->inter_gru_g1->U_z,
           dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->hidden_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->b_z, dpgrnn->inter_gru_g1->b_z,
           dpgrnn->inter_gru_g1->hidden_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->W_r, dpgrnn->inter_gru_g1->W_r,
           dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->input_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->U_r, dpgrnn->inter_gru_g1->U_r,
           dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->hidden_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->b_r, dpgrnn->inter_gru_g1->b_r,
           dpgrnn->inter_gru_g1->hidden_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->W_h, dpgrnn->inter_gru_g1->W_h,
           dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->input_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->U_h, dpgrnn->inter_gru_g1->U_h,
           dpgrnn->inter_gru_g1->hidden_size * dpgrnn->inter_gru_g1->hidden_size * sizeof(float));
    memcpy(dpgrnn->inter_gru_g2->b_h, dpgrnn->inter_gru_g1->b_h,
           dpgrnn->inter_gru_g1->hidden_size * sizeof(float));

    printf("GRU weights initialized\n\n");

    // 创建输入张量
    Tensor input = {
        .data = (float*)malloc(B * C * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C, .height = T, .width = F}
    };

    // 初始化输入数据
    for (int i = 0; i < B * C * T * F; i++) {
        input.data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    print_tensor_stats("Input", input.data, B * C * T * F);

    // 创建输出张量
    Tensor output = {
        .data = (float*)malloc(B * C * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C, .height = T, .width = F}
    };

    // 执行前向传播
    printf("\nRunning DPGRNN forward pass...\n");
    dpgrnn_forward(&input, &output, dpgrnn);
    printf("Forward pass completed\n\n");

    // 打印输出统计信息
    print_tensor_stats("Output", output.data, B * C * T * F);

    // 验证输出形状
    printf("\nOutput shape: (B=%d, C=%d, T=%d, F=%d)\n",
           output.shape.batch, output.shape.channels,
           output.shape.height, output.shape.width);

    // 检查是否有NaN或Inf
    int has_nan = 0;
    int has_inf = 0;
    for (int i = 0; i < B * C * T * F; i++) {
        if (isnan(output.data[i])) has_nan = 1;
        if (isinf(output.data[i])) has_inf = 1;
    }

    if (has_nan) {
        printf("WARNING: Output contains NaN values!\n");
    }
    if (has_inf) {
        printf("WARNING: Output contains Inf values!\n");
    }
    if (!has_nan && !has_inf) {
        printf("✓ Output is valid (no NaN or Inf)\n");
    }

    // 清理
    free(input.data);
    free(output.data);
    dpgrnn_free(dpgrnn);

    printf("\nTest 1 completed\n");
}

int main() {
    printf("========================================\n");
    printf("DPGRNN Implementation Test Suite\n");
    printf("========================================\n");

    // 设置随机种子
    srand(42);

    // 运行测试
    test_dpgrnn_forward();

    printf("\n========================================\n");
    printf("All tests completed\n");
    printf("========================================\n");

    return 0;
}
