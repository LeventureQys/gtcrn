/**
 * example_use_complete_modules.c - 演示如何使用完整实现的模块
 *
 * 本示例展示了如何在实际代码中使用：
 * 1. GRU_bidirectional_complete.c 中的双向分组GRU
 * 2. gtconvblock_forward_complete.c 中的完整GTConvBlock
 *
 * 编译:
 *   gcc -o example_complete example_use_complete_modules.c \
 *       gtcrn_model.c gtcrn_modules.c GRU.c conv2d.c batchnorm2d.c \
 *       nn_layers.c layernorm.c stft.c weight_loader.c \
 *       GRU_bidirectional_complete.c gtconvblock_forward_complete.c \
 *       -lm -O3
 *
 * 运行:
 *   ./example_complete
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "gtcrn_model.h"
#include "GRU.h"  // For GRUWeights definition

// 注意: GRU_bidirectional_complete.c 和 gtconvblock_forward_complete.c
// 已在主库 gtcrn_static 的 UTIL_SOURCES 中编译
// 这里提供必要的类型和函数声明

// BiGRNNWeights 结构体声明
// 实际定义在 GRU_bidirectional_complete.c 中
typedef struct {
    GRUWeights* fwd_g1;
    GRUWeights* fwd_g2;
    GRUWeights* bwd_g1;
    GRUWeights* bwd_g2;
} BiGRNNWeights;

// UniGRNNWeights 结构体声明
// 实际定义在 GRU_bidirectional_complete.c 中
typedef struct {
    GRUWeights* g1;
    GRUWeights* g2;
} UniGRNNWeights;

// 函数声明
BiGRNNWeights* bigrnn_weights_create(int input_size, int hidden_size);
void bigrnn_weights_free(BiGRNNWeights* weights);
UniGRNNWeights* unigrnn_weights_create(int input_size, int hidden_size);
void unigrnn_weights_free(UniGRNNWeights* weights);

void grnn_bidirectional_forward_complete(
    const float* input,
    float* output,
    const float* h_init_fwd_g1,
    const float* h_init_fwd_g2,
    const float* h_init_bwd_g1,
    const float* h_init_bwd_g2,
    const GRUWeights* weights_fwd_g1,
    const GRUWeights* weights_fwd_g2,
    const GRUWeights* weights_bwd_g1,
    const GRUWeights* weights_bwd_g2,
    int seq_len,
    float* temp
);

void grnn_unidirectional_forward_with_state(
    const float* input,
    float* output,
    const float* h_prev_g1,
    const float* h_prev_g2,
    float* h_next_g1,
    float* h_next_g2,
    const GRUWeights* weights_g1,
    const GRUWeights* weights_g2,
    int seq_len,
    float* temp
);

void gtconvblock_forward_complete(
    const Tensor* input,
    Tensor* output,
    GTConvBlock* block,
    int kernel_h,
    int dilation_h
);

// ============================================================================
// 示例1: 使用完整的双向分组GRU
// ============================================================================

void example_bidirectional_grnn() {
    printf("\n");
    printf("=================================================================\n");
    printf("示例1: 双向分组GRU (Bidirectional Grouped GRU)\n");
    printf("=================================================================\n\n");

    // GTCRN Intra-RNN 参数
    int seq_len = 97;      // 频率bins
    int input_size = 16;   // 通道数
    int hidden_size = 16;  // 输出大小（双向分组后与输入相同）

    printf("配置:\n");
    printf("  序列长度: %d (频率bins)\n", seq_len);
    printf("  输入大小: %d (通道数)\n", input_size);
    printf("  隐藏大小: %d\n", hidden_size);
    printf("  分组数: 2\n");
    printf("  方向: 双向\n\n");

    // 创建权重结构
    BiGRNNWeights* weights = bigrnn_weights_create(input_size, hidden_size);

    // 初始化权重（实际使用时应从文件加载）
    printf("初始化权重...\n");
    int input_size_per_group = input_size / 2;
    int hidden_size_per_group = hidden_size / 4;

    // 简单的随机初始化（仅用于演示）
    for (int i = 0; i < hidden_size_per_group * input_size_per_group; i++) {
        weights->fwd_g1->W_z[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        weights->fwd_g1->W_r[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        weights->fwd_g1->W_h[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        // ... 其他权重类似
    }

    // 分配输入/输出缓冲区
    float* input = (float*)malloc(seq_len * input_size * sizeof(float));
    float* output = (float*)malloc(seq_len * input_size * sizeof(float));
    float* temp = (float*)malloc(4 * hidden_size * sizeof(float));

    // 初始化输入数据
    printf("生成测试输入数据...\n");
    for (int i = 0; i < seq_len * input_size; i++) {
        input[i] = sinf(2.0f * M_PI * i / 100.0f) * 0.5f;
    }

    // 运行双向分组GRU
    printf("运行双向分组GRU...\n");
    grnn_bidirectional_forward_complete(
        input, output,
        NULL, NULL, NULL, NULL,  // 无初始隐藏状态
        weights->fwd_g1, weights->fwd_g2,
        weights->bwd_g1, weights->bwd_g2,
        seq_len, temp
    );

    // 显示结果
    printf("\n结果:\n");
    printf("  输入形状: (%d, %d)\n", seq_len, input_size);
    printf("  输出形状: (%d, %d)\n", seq_len, input_size);
    printf("\n  前5个时间步的输出 (仅显示前4个通道):\n");
    for (int t = 0; t < 5; t++) {
        printf("    t=%d: ", t);
        for (int c = 0; c < 4; c++) {
            printf("%.4f ", output[t * input_size + c]);
        }
        printf("...\n");
    }

    // 验证输出
    int nan_count = 0;
    for (int i = 0; i < seq_len * input_size; i++) {
        if (isnan(output[i]) || isinf(output[i])) {
            nan_count++;
        }
    }

    printf("\n验证:\n");
    if (nan_count == 0) {
        printf("  ✓ 输出无NaN/Inf值\n");
    } else {
        printf("  ✗ 发现 %d 个NaN/Inf值\n", nan_count);
    }

    // 清理
    free(input);
    free(output);
    free(temp);
    bigrnn_weights_free(weights);

    printf("\n✓ 示例1完成\n");
}

// ============================================================================
// 示例2: 使用完整的GTConvBlock
// ============================================================================

void example_gtconvblock() {
    printf("\n");
    printf("=================================================================\n");
    printf("示例2: 完整GTConvBlock前向传播\n");
    printf("=================================================================\n\n");

    // GTConvBlock 参数
    int B = 1;      // Batch size
    int C = 16;     // 通道数
    int T = 10;     // 时间帧数
    int F = 97;     // 频率bins

    printf("配置:\n");
    printf("  Batch size: %d\n", B);
    printf("  通道数: %d\n", C);
    printf("  时间帧: %d\n", T);
    printf("  频率bins: %d\n", F);
    printf("  卷积核: (3, 3)\n");
    printf("  膨胀率: (1, 1)\n\n");

    // 创建GTConvBlock
    printf("创建GTConvBlock...\n");
    GTConvBlock* block = gtconvblock_create(C, C, 3, 3, 1, 1, 0, 1, 1, 1, 0);

    // 分配输入/输出张量
    Tensor input = {
        .data = (float*)malloc(B * C * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C, .height = T, .width = F}
    };

    Tensor output = {
        .data = (float*)malloc(B * C * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C, .height = T, .width = F}
    };

    // 初始化输入数据
    printf("生成测试输入数据...\n");
    for (int i = 0; i < B * C * T * F; i++) {
        input.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    // 运行完整的GTConvBlock前向传播
    printf("运行GTConvBlock前向传播...\n");
    printf("  步骤:\n");
    printf("    1. Channel Split\n");
    printf("    2. SFE (子带特征提取)\n");
    printf("    3. Point Conv1 + BN + PReLU\n");
    printf("    4. Temporal Padding\n");
    printf("    5. Depth Conv + BN + PReLU\n");
    printf("    6. Temporal Unpadding\n");
    printf("    7. Point Conv2 + BN\n");
    printf("    8. TRA (时间循环注意力)\n");
    printf("    9. Channel Shuffle\n\n");

    gtconvblock_forward_complete(&input, &output, block, 3, 1);

    // 显示结果
    printf("结果:\n");
    printf("  输入形状: (%d, %d, %d, %d)\n", B, C, T, F);
    printf("  输出形状: (%d, %d, %d, %d)\n", B, C, T, F);

    // 计算统计信息
    float min_val = output.data[0];
    float max_val = output.data[0];
    double sum = 0.0;
    int total = B * C * T * F;

    for (int i = 0; i < total; i++) {
        float val = output.data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    float mean = sum / total;

    printf("\n  输出统计:\n");
    printf("    最小值: %.6f\n", min_val);
    printf("    最大值: %.6f\n", max_val);
    printf("    平均值: %.6f\n", mean);

    // 验证输出
    int nan_count = 0;
    for (int i = 0; i < total; i++) {
        if (isnan(output.data[i]) || isinf(output.data[i])) {
            nan_count++;
        }
    }

    printf("\n验证:\n");
    if (nan_count == 0) {
        printf("  ✓ 输出无NaN/Inf值\n");
    } else {
        printf("  ✗ 发现 %d 个NaN/Inf值\n", nan_count);
    }

    // 清理
    free(input.data);
    free(output.data);
    gtconvblock_free(block);

    printf("\n✓ 示例2完成\n");
}

// ============================================================================
// 示例3: 流式处理中使用状态缓存
// ============================================================================

void example_streaming_with_state() {
    printf("\n");
    printf("=================================================================\n");
    printf("示例3: 流式处理中的状态缓存\n");
    printf("=================================================================\n\n");

    // 参数
    int seq_len = 1;       // 流式处理：每次1帧
    int input_size = 16;
    int hidden_size = 16;
    int num_frames = 10;   // 处理10帧

    printf("配置:\n");
    printf("  每帧大小: %d\n", input_size);
    printf("  总帧数: %d\n", num_frames);
    printf("  使用状态缓存: 是\n\n");

    // 创建权重
    UniGRNNWeights* weights = unigrnn_weights_create(input_size, hidden_size);

    // 初始化隐藏状态缓存
    int hidden_size_per_group = hidden_size / 2;
    float* h_cache_g1 = (float*)calloc(hidden_size_per_group, sizeof(float));
    float* h_cache_g2 = (float*)calloc(hidden_size_per_group, sizeof(float));

    printf("逐帧处理:\n");

    // 逐帧处理
    for (int frame = 0; frame < num_frames; frame++) {
        // 准备当前帧输入
        float* input = (float*)malloc(seq_len * input_size * sizeof(float));
        float* output = (float*)malloc(seq_len * input_size * sizeof(float));
        float* temp = (float*)malloc(4 * hidden_size * sizeof(float));

        // 生成输入数据
        for (int i = 0; i < input_size; i++) {
            input[i] = sinf(2.0f * M_PI * (frame * input_size + i) / 100.0f);
        }

        // 使用状态缓存的单向GRU
        grnn_unidirectional_forward_with_state(
            input, output,
            h_cache_g1, h_cache_g2,  // 使用上一帧的隐藏状态
            h_cache_g1, h_cache_g2,  // 更新隐藏状态
            weights->g1, weights->g2,
            seq_len, temp
        );

        // 显示结果
        printf("  帧 %2d: 输入[0]=%.4f, 输出[0]=%.4f, 隐藏状态[0]=%.4f\n",
               frame, input[0], output[0], h_cache_g1[0]);

        free(input);
        free(output);
        free(temp);
    }

    printf("\n说明:\n");
    printf("  - 每帧使用上一帧的隐藏状态\n");
    printf("  - 隐藏状态在帧之间持久化\n");
    printf("  - 这是真正的流式处理（因果性）\n");

    // 清理
    free(h_cache_g1);
    free(h_cache_g2);
    unigrnn_weights_free(weights);

    printf("\n✓ 示例3完成\n");
}

// ============================================================================
// 主函数
// ============================================================================

int main() {
    printf("\n");
    printf("#################################################################\n");
    printf("# GTCRN 完整模块使用示例\n");
    printf("#################################################################\n");

    // 运行示例
    example_bidirectional_grnn();
    example_gtconvblock();
    example_streaming_with_state();

    printf("\n");
    printf("#################################################################\n");
    printf("# 所有示例完成！\n");
    printf("#################################################################\n\n");

    printf("下一步:\n");
    printf("  1. 查看 INTEGRATION_GUIDE.md 了解如何集成到主代码\n");
    printf("  2. 从PyTorch导出实际权重: python export_weights.py\n");
    printf("  3. 使用实际权重测试: ./denoise input.wav output.wav weights/\n\n");

    return 0;
}
