#include "nn_layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void print_separator() {
    printf("=================================================================\n");
}

void test_linear() {
    printf("\n");
    print_separator();
    printf("Test 1: nn.Linear - 全连接层\n");
    print_separator();
    printf("从 gtcrn1.py line 82: nn.Linear(channels*2, channels)\n");
    printf("用于 TRA (Temporal Recurrent Attention) 模块\n\n");

    int batch_size = 4;
    int in_features = 32;
    int out_features = 16;

    // 创建输入
    float* input = (float*)malloc(batch_size * in_features * sizeof(float));
    float* output = (float*)malloc(batch_size * out_features * sizeof(float));

    srand(42);
    for (int i = 0; i < batch_size * in_features; i++) {
        input[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // 创建权重和偏置
    float* weight = (float*)malloc(out_features * in_features * sizeof(float));
    float* bias = (float*)malloc(out_features * sizeof(float));

    for (int i = 0; i < out_features * in_features; i++) {
        weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < out_features; i++) {
        bias[i] = 0.0f;
    }

    // 创建 Linear 参数
    LinearParams* linear_params = linear_create(in_features, out_features, weight, bias, 1);

    printf("配置:\n");
    printf("  输入: [%d, %d]\n", batch_size, in_features);
    printf("  权重: [%d, %d]\n", out_features, in_features);
    printf("  输出: [%d, %d]\n\n", batch_size, out_features);

    // 前向传播
    clock_t start = clock();
    linear_forward(input, output, batch_size, linear_params);
    clock_t end = clock();

    printf("输入样本:\n  ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", input[i]);
    }
    printf("...\n");

    printf("输出样本:\n  ");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", output[i]);
    }
    printf("...\n\n");

    printf("时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // 清理
    free(input);
    free(output);
    free(weight);
    free(bias);
    linear_free(linear_params);
}

void test_unfold() {
    printf("\n");
    print_separator();
    printf("Test 2: nn.Unfold - 展开操作\n");
    print_separator();
    printf("从 gtcrn1.py line 69: nn.Unfold(kernel_size=(1,3), stride=(1,1), padding=(0,1))\n");
    printf("用于 SFE (Subband Feature Extraction) 模块\n\n");

    int batch = 1;
    int channels = 8;
    int height = 63;
    int width = 97;
    int kernel_size = 3;

    // 创建输入
    Tensor* input = tensor_create(batch, channels, height, width);

    srand(123);
    for (int i = 0; i < batch * channels * height * width; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    printf("配置:\n");
    printf("  输入: [%d, %d, %d, %d]\n", batch, channels, height, width);
    printf("  Kernel: (1, %d)\n", kernel_size);
    printf("  Stride: (1, 1)\n");
    printf("  Padding: (0, %d)\n\n", (kernel_size-1)/2);

    // 设置 Unfold 参数
    UnfoldParams unfold_params;
    unfold_params.kernel_h = 1;
    unfold_params.kernel_w = kernel_size;
    unfold_params.stride_h = 1;
    unfold_params.stride_w = 1;
    unfold_params.padding_h = 0;
    unfold_params.padding_w = (kernel_size - 1) / 2;
    unfold_params.dilation_h = 1;
    unfold_params.dilation_w = 1;

    // 创建输出 (B, C*kernel_size, T, F)
    int out_channels = channels * kernel_size;
    Tensor* output = tensor_create(batch, out_channels, height, width);

    printf("输出: [%d, %d, %d, %d]\n", batch, out_channels, height, width);
    printf("通道扩展: %d -> %d (×%d)\n\n", channels, out_channels, kernel_size);

    // 前向传播
    clock_t start = clock();
    unfold_reshape_4d(input, output, &unfold_params);
    clock_t end = clock();

    print_tensor_stats_v2("Input", input);
    print_tensor_stats_v2("Output", output);

    printf("\n时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\n说明:\n");
    printf("  Unfold 将每个位置的邻域展开为特征\n");
    printf("  例如: 位置 (t,f) 的 3 个邻域值 -> 3 个通道\n");
    printf("  这样可以捕获局部频率模式\n");

    // 清理
    tensor_free(input);
    tensor_free(output);
}

void test_prelu() {
    printf("\n");
    print_separator();
    printf("Test 3: nn.PReLU - Parametric ReLU\n");
    print_separator();
    printf("从 gtcrn1.py line 102, 119, 125: nn.PReLU()\n");
    printf("用于所有 ConvBlock 和 GTConvBlock\n\n");

    int batch = 1;
    int channels = 16;
    int height = 32;
    int width = 32;

    // 创建输入
    Tensor* input = tensor_create(batch, channels, height, width);

    srand(456);
    for (int i = 0; i < batch * channels * height * width; i++) {
        input->data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
    }

    printf("配置:\n");
    printf("  输入: [%d, %d, %d, %d]\n", batch, channels, height, width);
    printf("  参数: %d 个 (每通道一个)\n\n", channels);

    // 创建 PReLU 参数
    float* prelu_weights = (float*)malloc(channels * sizeof(float));
    for (int i = 0; i < channels; i++) {
        prelu_weights[i] = 0.25f;  // PyTorch 默认值
    }

    PReLUParams* prelu_params = prelu_create(channels, prelu_weights);

    print_tensor_stats_v2("Before PReLU", input);

    // 前向传播
    clock_t start = clock();
    prelu_forward_v2(input, prelu_params);
    clock_t end = clock();

    print_tensor_stats_v2("After PReLU", input);

    printf("\n时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\nPReLU 公式:\n");
    printf("  y = x           if x > 0\n");
    printf("  y = alpha * x   if x <= 0\n");
    printf("  (alpha = 0.25 for GTCRN)\n");

    // 清理
    free(prelu_weights);
    prelu_free(prelu_params);
    tensor_free(input);
}

void test_sigmoid() {
    printf("\n");
    print_separator();
    printf("Test 4: nn.Sigmoid - Sigmoid 激活\n");
    print_separator();
    printf("从 gtcrn1.py line 83: nn.Sigmoid()\n");
    printf("用于 TRA (Temporal Recurrent Attention) 的注意力权重\n\n");

    int batch = 1;
    int channels = 16;
    int height = 63;
    int width = 1;

    // 创建输入
    Tensor* input = tensor_create(batch, channels, height, width);

    srand(789);
    for (int i = 0; i < batch * channels * height * width; i++) {
        input->data[i] = (float)rand() / RAND_MAX * 4.0f - 2.0f;  // [-2, 2]
    }

    printf("配置:\n");
    printf("  输入: [%d, %d, %d, %d]\n", batch, channels, height, width);
    printf("  输出范围: (0, 1)\n\n");

    print_tensor_stats_v2("Before Sigmoid", input);

    // 前向传播
    clock_t start = clock();
    sigmoid_forward_tensor(input);
    clock_t end = clock();

    print_tensor_stats_v2("After Sigmoid", input);

    printf("\n时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\nSigmoid 公式:\n");
    printf("  y = 1 / (1 + exp(-x))\n");
    printf("  输出范围: (0, 1)\n");
    printf("  用于生成注意力权重\n");

    // 清理
    tensor_free(input);
}

void test_gtcrn_sfe_module() {
    printf("\n");
    print_separator();
    printf("Test 5: GTCRN SFE 模块完整流程\n");
    print_separator();
    printf("从 gtcrn1.py lines 64-74\n");
    printf("SFE (Subband Feature Extraction) 完整实现\n\n");

    int batch = 1;
    int channels = 8;  // 输入通道数 (in_channels//2)
    int height = 63;   // 时间帧
    int width = 97;    // 频率bins
    int kernel_size = 3;

    printf("SFE 模块:\n");
    printf("  输入: [%d, %d, %d, %d]\n", batch, channels, height, width);
    printf("  Unfold kernel_size=%d\n", kernel_size);
    printf("  输出: [%d, %d, %d, %d]\n\n", batch, channels*kernel_size, height, width);

    // 创建输入
    Tensor* input = tensor_create(batch, channels, height, width);

    srand(111);
    for (int i = 0; i < batch * channels * height * width; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    // 设置 Unfold 参数
    UnfoldParams unfold_params;
    unfold_params.kernel_h = 1;
    unfold_params.kernel_w = kernel_size;
    unfold_params.stride_h = 1;
    unfold_params.stride_w = 1;
    unfold_params.padding_h = 0;
    unfold_params.padding_w = (kernel_size - 1) / 2;
    unfold_params.dilation_h = 1;
    unfold_params.dilation_w = 1;

    // 创建输出
    int out_channels = channels * kernel_size;
    Tensor* output = tensor_create(batch, out_channels, height, width);

    // 执行 SFE
    clock_t start = clock();
    unfold_reshape_4d(input, output, &unfold_params);
    clock_t end = clock();

    print_tensor_stats_v2("SFE Input", input);
    print_tensor_stats_v2("SFE Output", output);

    printf("\n时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\nSFE 作用:\n");
    printf("  1. 提取子带特征（频率邻域）\n");
    printf("  2. 将局部频率模式展开为通道\n");
    printf("  3. 为后续卷积提供更丰富的特征\n");
    printf("  4. 在 GTConvBlock 中使用（line 115）\n");

    // 清理
    tensor_free(input);
    tensor_free(output);
}

void test_gtcrn_tra_attention() {
    printf("\n");
    print_separator();
    printf("Test 6: GTCRN TRA 注意力机制（部分）\n");
    print_separator();
    printf("从 gtcrn1.py lines 77-93\n");
    printf("TRA (Temporal Recurrent Attention) 的 Linear + Sigmoid 部分\n\n");

    int batch = 1;
    int channels = 8;
    int time_steps = 63;

    printf("TRA 流程:\n");
    printf("  1. 计算能量: zt = mean(x^2, dim=-1)\n");
    printf("  2. GRU: at = GRU(zt)  [跳过，需要 RNN 实现]\n");
    printf("  3. Linear: at = Linear(at)\n");
    printf("  4. Sigmoid: at = Sigmoid(at)\n");
    printf("  5. 应用注意力: x = x * at\n\n");

    // 模拟 GRU 输出（实际应该是 GRU 的输出）
    int gru_output_size = channels * 2;
    float* gru_output = (float*)malloc(batch * time_steps * gru_output_size * sizeof(float));

    srand(222);
    for (int i = 0; i < batch * time_steps * gru_output_size; i++) {
        gru_output[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    printf("模拟 GRU 输出: [%d, %d, %d]\n", batch, time_steps, gru_output_size);

    // Linear: channels*2 -> channels
    float* linear_output = (float*)malloc(batch * time_steps * channels * sizeof(float));

    float* weight = (float*)malloc(channels * gru_output_size * sizeof(float));
    float* bias = (float*)malloc(channels * sizeof(float));

    for (int i = 0; i < channels * gru_output_size; i++) {
        weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    for (int i = 0; i < channels; i++) {
        bias[i] = 0.0f;
    }

    LinearParams* linear_params = linear_create(gru_output_size, channels, weight, bias, 1);

    // 执行 Linear
    linear_forward(gru_output, linear_output, batch * time_steps, linear_params);

    printf("Linear 输出: [%d, %d, %d]\n", batch, time_steps, channels);

    // 执行 Sigmoid
    sigmoid_forward(linear_output, batch * time_steps * channels);

    printf("Sigmoid 输出 (注意力权重): [%d, %d, %d]\n", batch, time_steps, channels);

    // 打印一些注意力权重
    printf("\n注意力权重样本:\n  ");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", linear_output[i]);
    }
    printf("...\n");

    printf("\n说明:\n");
    printf("  注意力权重范围: (0, 1)\n");
    printf("  用于调制输入特征的重要性\n");
    printf("  在 GTConvBlock 中应用（line 149）\n");

    // 清理
    free(gru_output);
    free(linear_output);
    free(weight);
    free(bias);
    linear_free(linear_params);
}

int main() {
    printf("\n");
    printf("#################################################################\n");
    printf("#                                                               #\n");
    printf("#  神经网络基础层 C 实现                                         #\n");
    printf("#  nn.Linear, nn.Unfold, nn.PReLU, nn.Sigmoid                  #\n");
    printf("#                                                               #\n");
    printf("#################################################################\n");

    test_linear();
    test_unfold();
    test_prelu();
    test_sigmoid();
    test_gtcrn_sfe_module();
    test_gtcrn_tra_attention();

    printf("\n");
    print_separator();
    printf("总结\n");
    print_separator();
    printf("\n");
    printf("问题: nn.Linear, nn.Unfold, nn.PReLU 可以用 C 实现吗？\n");
    printf("\n");
    printf("答案: 是的！完全可以！\n");
    printf("\n");
    printf("已实现的层:\n");
    printf("  ✓ nn.Linear      - 全连接层（矩阵乘法）\n");
    printf("  ✓ nn.Unfold      - 展开操作（im2col）\n");
    printf("  ✓ nn.PReLU       - 参数化 ReLU\n");
    printf("  ✓ nn.Sigmoid     - Sigmoid 激活\n");
    printf("  ✓ nn.Tanh        - Tanh 激活（已在 conv2d.c）\n");
    printf("\n");
    printf("GTCRN 中的使用:\n");
    printf("  - SFE 模块: Unfold (line 69)\n");
    printf("  - TRA 模块: Linear (line 82) + Sigmoid (line 83)\n");
    printf("  - ConvBlock: PReLU (line 102)\n");
    printf("  - GTConvBlock: PReLU (line 119, 125)\n");
    printf("\n");
    printf("实现文件:\n");
    printf("  - nn_layers.h\n");
    printf("  - nn_layers.c\n");
    printf("  - test_nn_layers.c\n");
    printf("\n");
    printf("特点:\n");
    printf("  ✓ 纯 C99 实现\n");
    printf("  ✓ 无外部依赖\n");
    printf("  ✓ 高效实现\n");
    printf("  ✓ 易于集成\n");
    printf("\n");

    return 0;
}
