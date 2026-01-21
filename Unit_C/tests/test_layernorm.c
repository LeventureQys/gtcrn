#include "layernorm.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void print_separator() {
    printf("=================================================================\n");
}

void print_array(const char* name, const float* data, int size, int max_print) {
    printf("%s: ", name);
    int n = (size < max_print) ? size : max_print;
    for (int i = 0; i < n; i++) {
        printf("%.4f ", data[i]);
    }
    if (size > max_print) printf("...");
    printf("\n");
}

void test_parameter() {
    printf("\n");
    print_separator();
    printf("Test 1: nn.Parameter - 可学习参数\n");
    print_separator();
    printf("在 PyTorch 中，Parameter 是可学习的张量\n");
    printf("在 C 中，就是普通的 float 数组\n\n");

    // 创建参数
    int shape[] = {10, 20};
    Parameter* param = parameter_create(shape, 2);

    printf("创建参数:\n");
    print_parameter_info("  param", param);

    // 初始化参数
    srand(42);
    for (int i = 0; i < param->total_size; i++) {
        param->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    printf("\n参数值样本:\n");
    print_array("  ", param->data, param->total_size, 10);

    printf("\n说明:\n");
    printf("  - nn.Parameter 在 C 中就是 float 数组\n");
    printf("  - 需要手动管理内存\n");
    printf("  - 在模型加载时从文件读取\n");
    printf("  - 在推理时保持不变\n");

    parameter_free(param);
}

void test_layernorm_basic() {
    printf("\n");
    print_separator();
    printf("Test 2: nn.LayerNorm - 基础测试\n");
    print_separator();
    printf("LayerNorm 归一化指定的维度\n\n");

    int batch_size = 4;
    int num_features = 10;

    // 创建输入
    float* input = (float*)malloc(batch_size * num_features * sizeof(float));
    float* output = (float*)malloc(batch_size * num_features * sizeof(float));

    srand(123);
    for (int i = 0; i < batch_size * num_features; i++) {
        input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    printf("配置:\n");
    printf("  输入: [%d, %d]\n", batch_size, num_features);
    printf("  归一化维度: [%d]\n\n", num_features);

    // 创建 LayerNorm 参数
    int normalized_shape[] = {num_features};
    LayerNormParams* ln_params = layernorm_create(
        normalized_shape, 1, NULL, NULL, 1e-5f
    );

    print_layernorm_info("LayerNorm", ln_params);

    // 打印第一个样本
    printf("\n第一个样本:\n");
    print_array("  输入", input, num_features, 10);

    // 前向传播
    clock_t start = clock();
    layernorm_forward(input, output, batch_size, ln_params);
    clock_t end = clock();

    print_array("  输出", output, num_features, 10);

    // 验证归一化
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < num_features; i++) {
        sum += output[i];
        sum_sq += output[i] * output[i];
    }
    float mean = sum / num_features;
    float var = sum_sq / num_features - mean * mean;

    printf("\n归一化验证:\n");
    printf("  均值: %.10f (应接近 0)\n", mean);
    printf("  方差: %.10f (应接近 1)\n", var);

    printf("\n时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    free(input);
    free(output);
    layernorm_free(ln_params);
}

void test_layernorm_2d() {
    printf("\n");
    print_separator();
    printf("Test 3: nn.LayerNorm - 2D 归一化\n");
    print_separator();
    printf("归一化最后两个维度（GTCRN DPGRNN 使用方式）\n\n");

    int batch_size = 2;
    int dim1 = 97;   // width (F)
    int dim2 = 16;   // hidden_size (C)

    int num_features = dim1 * dim2;
    int total_size = batch_size * num_features;

    // 创建输入
    float* input = (float*)malloc(total_size * sizeof(float));
    float* output = (float*)malloc(total_size * sizeof(float));

    srand(456);
    for (int i = 0; i < total_size; i++) {
        input[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    printf("配置:\n");
    printf("  输入: [%d, %d, %d] (batch, width, hidden_size)\n", batch_size, dim1, dim2);
    printf("  归一化维度: [%d, %d]\n", dim1, dim2);
    printf("  特征总数: %d\n\n", num_features);

    // 创建 LayerNorm 参数
    int normalized_shape[] = {dim1, dim2};
    LayerNormParams* ln_params = layernorm_create(
        normalized_shape, 2, NULL, NULL, 1e-8f
    );

    print_layernorm_info("LayerNorm", ln_params);

    // 前向传播
    clock_t start = clock();
    layernorm_forward(input, output, batch_size, ln_params);
    clock_t end = clock();

    // 统计
    float min_val = output[0], max_val = output[0];
    double sum = 0.0;
    for (int i = 0; i < num_features; i++) {
        float val = output[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    printf("\n第一个样本统计:\n");
    printf("  最小值: %.6f\n", min_val);
    printf("  最大值: %.6f\n", max_val);
    printf("  均值: %.10f (应接近 0)\n", sum / num_features);

    printf("\n时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    free(input);
    free(output);
    layernorm_free(ln_params);
}

void test_gtcrn_dpgrnn_layernorm() {
    printf("\n");
    print_separator();
    printf("Test 4: GTCRN DPGRNN LayerNorm\n");
    print_separator();
    printf("从 gtcrn1.py lines 196, 200:\n");
    printf("  self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)\n");
    printf("  self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)\n\n");

    int batch = 1;
    int time_steps = 63;  // T
    int width = 97;       // F
    int hidden_size = 16; // C

    printf("DPGRNN 配置:\n");
    printf("  输入: [%d, %d, %d, %d] (B, T, F, C)\n", batch, time_steps, width, hidden_size);
    printf("  归一化: 最后两个维度 (F, C)\n");
    printf("  width=%d, hidden_size=%d\n\n", width, hidden_size);

    // 创建输入（模拟为 4D 张量）
    Tensor* input = tensor_create(batch, time_steps, width, hidden_size);

    srand(789);
    for (int i = 0; i < batch * time_steps * width * hidden_size; i++) {
        input->data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }

    // 创建 LayerNorm 参数
    int normalized_shape[] = {width, hidden_size};
    LayerNormParams* intra_ln = layernorm_create(
        normalized_shape, 2, NULL, NULL, 1e-8f
    );

    printf("Intra LayerNorm:\n");
    print_layernorm_info("  ", intra_ln);

    // 打印归一化前的统计
    int sample_size = width * hidden_size;
    float min_before = input->data[0], max_before = input->data[0];
    double sum_before = 0.0;

    for (int i = 0; i < sample_size; i++) {
        float val = input->data[i];
        if (val < min_before) min_before = val;
        if (val > max_before) max_before = val;
        sum_before += val;
    }

    printf("\n归一化前（第一个样本）:\n");
    printf("  最小值: %.6f\n", min_before);
    printf("  最大值: %.6f\n", max_before);
    printf("  均值: %.6f\n", sum_before / sample_size);

    // 执行 LayerNorm
    clock_t start = clock();
    layernorm_forward_4d(input, intra_ln);
    clock_t end = clock();

    // 打印归一化后的统计
    float min_after = input->data[0], max_after = input->data[0];
    double sum_after = 0.0, sum_sq = 0.0;

    for (int i = 0; i < sample_size; i++) {
        float val = input->data[i];
        if (val < min_after) min_after = val;
        if (val > max_after) max_after = val;
        sum_after += val;
        sum_sq += val * val;
    }

    float mean_after = sum_after / sample_size;
    float var_after = sum_sq / sample_size - mean_after * mean_after;

    printf("\n归一化后（第一个样本）:\n");
    printf("  最小值: %.6f\n", min_after);
    printf("  最大值: %.6f\n", max_after);
    printf("  均值: %.10f (应接近 0)\n", mean_after);
    printf("  方差: %.10f (应接近 1)\n", var_after);

    printf("\n时间: %.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\n说明:\n");
    printf("  - DPGRNN 使用 LayerNorm 稳定训练\n");
    printf("  - 归一化 (F, C) 维度\n");
    printf("  - intra_ln 用于 intra RNN 输出\n");
    printf("  - inter_ln 用于 inter RNN 输出\n");
    printf("  - 配合残差连接使用\n");

    layernorm_free(intra_ln);
    tensor_free(input);
}

void test_layernorm_vs_batchnorm() {
    printf("\n");
    print_separator();
    printf("Test 5: LayerNorm vs BatchNorm 对比\n");
    print_separator();
    printf("理解两者的区别\n\n");

    printf("BatchNorm2d:\n");
    printf("  - 归一化: 对每个通道，在 batch 和空间维度上归一化\n");
    printf("  - 统计: 跨 batch 计算均值和方差\n");
    printf("  - 输入: (B, C, H, W)\n");
    printf("  - 归一化维度: (B, H, W) 对每个 C\n");
    printf("  - 参数: gamma[C], beta[C]\n");
    printf("  - 用途: CNN，依赖 batch 统计\n\n");

    printf("LayerNorm:\n");
    printf("  - 归一化: 对每个样本，在特征维度上归一化\n");
    printf("  - 统计: 每个样本独立计算均值和方差\n");
    printf("  - 输入: (B, ..., normalized_dims)\n");
    printf("  - 归一化维度: normalized_dims\n");
    printf("  - 参数: gamma[normalized_dims], beta[normalized_dims]\n");
    printf("  - 用途: RNN/Transformer，不依赖 batch\n\n");

    printf("GTCRN 中的使用:\n");
    printf("  - BatchNorm2d: 用于 CNN 层（ConvBlock, GTConvBlock）\n");
    printf("  - LayerNorm: 用于 RNN 层（DPGRNN）\n\n");

    printf("为什么 DPGRNN 使用 LayerNorm?\n");
    printf("  1. RNN 处理序列，batch 大小可能很小\n");
    printf("  2. 不依赖 batch 统计，更稳定\n");
    printf("  3. 每个样本独立归一化\n");
    printf("  4. 适合变长序列\n");
}

void test_layernorm_with_learnable_params() {
    printf("\n");
    print_separator();
    printf("Test 6: LayerNorm 可学习参数\n");
    print_separator();
    printf("gamma 和 beta 是可学习的参数\n\n");

    int batch_size = 2;
    int num_features = 10;

    // 创建输入
    float* input = (float*)malloc(batch_size * num_features * sizeof(float));
    float* output = (float*)malloc(batch_size * num_features * sizeof(float));

    for (int i = 0; i < batch_size * num_features; i++) {
        input[i] = (float)i / 10.0f;
    }

    // 创建自定义的 gamma 和 beta
    float* gamma = (float*)malloc(num_features * sizeof(float));
    float* beta = (float*)malloc(num_features * sizeof(float));

    for (int i = 0; i < num_features; i++) {
        gamma[i] = 2.0f;   // 缩放因子
        beta[i] = 0.5f;    // 偏移
    }

    printf("配置:\n");
    printf("  输入: [%d, %d]\n", batch_size, num_features);
    printf("  gamma: 全部为 2.0\n");
    printf("  beta: 全部为 0.5\n\n");

    // 创建 LayerNorm
    int normalized_shape[] = {num_features};
    LayerNormParams* ln_params = layernorm_create(
        normalized_shape, 1, gamma, beta, 1e-5f
    );

    // 前向传播
    layernorm_forward(input, output, batch_size, ln_params);

    printf("第一个样本:\n");
    print_array("  输入", input, num_features, 10);
    print_array("  输出", output, num_features, 10);

    printf("\n说明:\n");
    printf("  - gamma 控制缩放\n");
    printf("  - beta 控制偏移\n");
    printf("  - 在训练时学习这些参数\n");
    printf("  - 在推理时使用训练好的值\n");

    free(input);
    free(output);
    free(gamma);
    free(beta);
    layernorm_free(ln_params);
}

int main() {
    printf("\n");
    printf("#################################################################\n");
    printf("#                                                               #\n");
    printf("#  nn.Parameter 和 nn.LayerNorm C 实现                          #\n");
    printf("#                                                               #\n");
    printf("#################################################################\n");

    test_parameter();
    test_layernorm_basic();
    test_layernorm_2d();
    test_gtcrn_dpgrnn_layernorm();
    test_layernorm_vs_batchnorm();
    test_layernorm_with_learnable_params();

    printf("\n");
    print_separator();
    printf("总结\n");
    print_separator();
    printf("\n");
    printf("问题: nn.Parameter 和 nn.LayerNorm 可以用 C 实现吗？\n");
    printf("\n");
    printf("答案: 是的！完全可以！\n");
    printf("\n");
    printf("nn.Parameter:\n");
    printf("  - 在 C 中就是普通的 float 数组\n");
    printf("  - 需要手动管理内存\n");
    printf("  - 从模型文件加载\n");
    printf("  - 推理时保持不变\n");
    printf("\n");
    printf("nn.LayerNorm:\n");
    printf("  - 归一化指定维度\n");
    printf("  - 每个样本独立计算统计量\n");
    printf("  - 不依赖 batch 大小\n");
    printf("  - 适合 RNN/Transformer\n");
    printf("\n");
    printf("GTCRN 中的使用:\n");
    printf("  - DPGRNN intra_ln: LayerNorm((width, hidden_size))\n");
    printf("  - DPGRNN inter_ln: LayerNorm((width, hidden_size))\n");
    printf("  - 输入: (B, T, F, C)\n");
    printf("  - 归一化: (F, C)\n");
    printf("\n");
    printf("公式:\n");
    printf("  mean = mean(x, dim=normalized_dims)\n");
    printf("  var = var(x, dim=normalized_dims)\n");
    printf("  y = gamma * (x - mean) / sqrt(var + eps) + beta\n");
    printf("\n");
    printf("实现文件:\n");
    printf("  - layernorm.h\n");
    printf("  - layernorm.c\n");
    printf("  - test_layernorm.c\n");
    printf("\n");

    return 0;
}
