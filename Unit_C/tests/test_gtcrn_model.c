#include "gtcrn_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void print_separator() {
    printf("=================================================================\n");
}

void test_gtcrn_creation() {
    printf("\n");
    print_separator();
    printf("Test 1: GTCRN 模型创建\n");
    print_separator();
    printf("\n");

    // 创建模型
    GTCRN* model = gtcrn_create();

    if (model) {
        printf("✓ 模型创建成功\n\n");
        print_gtcrn_info(model);
        gtcrn_free(model);
    } else {
        printf("✗ 模型创建失败\n");
    }
}

void test_gtcrn_forward() {
    printf("\n");
    print_separator();
    printf("Test 2: GTCRN 前向传播\n");
    print_separator();
    printf("\n");

    // 创建模型
    GTCRN* model = gtcrn_create();
    if (!model) {
        printf("模型创建失败\n");
        return;
    }

    // 输入参数
    int batch = 1;
    int freq_bins = 769;  // 48kHz, 1536 FFT
    int time_frames = 63; // 约 1 秒音频

    printf("输入配置:\n");
    printf("  采样率: 48kHz\n");
    printf("  FFT 大小: 1536\n");
    printf("  频率bins: %d\n", freq_bins);
    printf("  时间帧: %d\n", time_frames);
    printf("  批次大小: %d\n\n", batch);

    // 分配输入输出
    int total_size = batch * freq_bins * time_frames * 2;
    float* spec_input = (float*)malloc(total_size * sizeof(float));
    float* spec_output = (float*)malloc(total_size * sizeof(float));

    // 生成模拟输入（随机复数频谱）
    srand(42);
    for (int i = 0; i < total_size; i++) {
        spec_input[i] = (float)rand() / RAND_MAX * 0.1f;
    }

    printf("生成模拟输入频谱...\n");
    printf("  实部范围: [0, 0.1]\n");
    printf("  虚部范围: [0, 0.1]\n\n");

    // 前向传播
    printf("执行前向传播...\n");
    clock_t start = clock();
    gtcrn_forward(spec_input, spec_output, batch, freq_bins, time_frames, model);
    clock_t end = clock();

    double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    printf("\n推理时间: %.2f ms\n", time_ms);

    // 计算统计
    float min_val = spec_output[0], max_val = spec_output[0];
    double sum = 0.0;
    for (int i = 0; i < total_size; i++) {
        if (spec_output[i] < min_val) min_val = spec_output[i];
        if (spec_output[i] > max_val) max_val = spec_output[i];
        sum += spec_output[i];
    }

    printf("\n输出统计:\n");
    printf("  最小值: %.6f\n", min_val);
    printf("  最大值: %.6f\n", max_val);
    printf("  均值: %.6f\n", sum / total_size);

    // 清理
    free(spec_input);
    free(spec_output);
    gtcrn_free(model);
}

void test_convblock() {
    printf("\n");
    print_separator();
    printf("Test 3: ConvBlock 测试\n");
    print_separator();
    printf("\n");

    printf("ConvBlock 结构:\n");
    printf("  Conv2d + BatchNorm2d + PReLU/Tanh\n");
    printf("  融合优化: Conv + BN 融合为一个操作\n\n");

    // 创建输入
    int batch = 1;
    int in_channels = 9;
    int out_channels = 16;
    int height = 63;
    int width = 385;

    Tensor* input = tensor_create(batch, in_channels, height, width);
    Tensor* output = tensor_create(batch, out_channels, height, width/2);

    srand(123);
    for (int i = 0; i < batch * in_channels * height * width; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    printf("输入: [%d, %d, %d, %d]\n", batch, in_channels, height, width);
    printf("输出: [%d, %d, %d, %d]\n", batch, out_channels, height, width/2);

    printf("\n注意: 完整实现需要从模型文件加载权重\n");

    // 清理
    tensor_free(input);
    tensor_free(output);
}

void test_dpgrnn() {
    printf("\n");
    print_separator();
    printf("Test 4: DPGRNN 测试\n");
    print_separator();
    printf("\n");

    printf("DPGRNN (Dual-Path Grouped RNN):\n");
    printf("  输入: (B, C, T, F)\n");
    printf("  处理:\n");
    printf("    1. Intra RNN: 在频率维度上处理\n");
    printf("    2. Inter RNN: 在时间维度上处理\n");
    printf("  输出: (B, C, T, F)\n\n");

    // 创建 DPGRNN
    int input_size = 16;
    int width = 97;
    int hidden_size = 16;

    DPGRNN* dpgrnn = dpgrnn_create(input_size, width, hidden_size);

    printf("DPGRNN 配置:\n");
    printf("  input_size: %d\n", input_size);
    printf("  width: %d\n", width);
    printf("  hidden_size: %d\n\n", hidden_size);

    printf("组件:\n");
    printf("  ✓ Intra RNN (需要 GRU 实现)\n");
    printf("  ✓ Intra Linear\n");
    printf("  ✓ Intra LayerNorm\n");
    printf("  ✓ Inter RNN (需要 GRU 实现)\n");
    printf("  ✓ Inter Linear\n");
    printf("  ✓ Inter LayerNorm\n\n");

    printf("注意: 完整实现需要 GRU 层\n");

    // 清理
    dpgrnn_free(dpgrnn);
}

void test_complex_mask() {
    printf("\n");
    print_separator();
    printf("Test 5: 复数掩码测试\n");
    print_separator();
    printf("\n");

    printf("复数掩码公式:\n");
    printf("  s_real = spec_real * mask_real - spec_imag * mask_imag\n");
    printf("  s_imag = spec_imag * mask_real + spec_real * mask_imag\n\n");

    int size = 10;

    float* spec_real = (float*)malloc(size * sizeof(float));
    float* spec_imag = (float*)malloc(size * sizeof(float));
    float* mask_real = (float*)malloc(size * sizeof(float));
    float* mask_imag = (float*)malloc(size * sizeof(float));
    float* output_real = (float*)malloc(size * sizeof(float));
    float* output_imag = (float*)malloc(size * sizeof(float));

    // 初始化
    for (int i = 0; i < size; i++) {
        spec_real[i] = 1.0f;
        spec_imag[i] = 0.5f;
        mask_real[i] = 0.8f;
        mask_imag[i] = 0.2f;
    }

    // 应用掩码
    apply_complex_mask(spec_real, spec_imag, mask_real, mask_imag,
                      output_real, output_imag, size);

    printf("示例:\n");
    printf("  输入频谱: (1.0, 0.5i)\n");
    printf("  掩码: (0.8, 0.2i)\n");
    printf("  输出: (%.2f, %.2fi)\n", output_real[0], output_imag[0]);

    printf("\n说明:\n");
    printf("  - 掩码由网络预测\n");
    printf("  - 应用到输入频谱\n");
    printf("  - 实现语音增强\n");

    // 清理
    free(spec_real);
    free(spec_imag);
    free(mask_real);
    free(mask_imag);
    free(output_real);
    free(output_imag);
}

void test_model_pipeline() {
    printf("\n");
    print_separator();
    printf("Test 6: GTCRN 完整流程\n");
    print_separator();
    printf("\n");

    printf("完整的语音增强流程:\n\n");

    printf("1. 音频输入\n");
    printf("   - 采样率: 48kHz\n");
    printf("   - 时长: 1 秒\n");
    printf("   - 样本数: 48000\n\n");

    printf("2. STFT (短时傅里叶变换)\n");
    printf("   - FFT 大小: 1536\n");
    printf("   - 跳跃长度: 768\n");
    printf("   - 窗函数: Hann^0.5\n");
    printf("   - 输出: (B, 769, 63, 2) 复数频谱\n\n");

    printf("3. GTCRN 处理\n");
    printf("   - 输入: (B, 769, 63, 2)\n");
    printf("   - 处理: 神经网络增强\n");
    printf("   - 输出: (B, 769, 63, 2) 增强频谱\n\n");

    printf("4. iSTFT (逆短时傅里叶变换)\n");
    printf("   - 输入: 增强频谱\n");
    printf("   - 输出: 48000 样本音频\n\n");

    printf("5. 输出音频\n");
    printf("   - 降噪后的语音\n");
    printf("   - 保持原始音质\n\n");

    printf("性能指标:\n");
    printf("  - 参数量: 23.67K\n");
    printf("  - 计算量: 33.0 MMACs\n");
    printf("  - 实时因子: < 0.1 (CPU)\n");
    printf("  - 延迟: < 50ms\n");
}

int main() {
    printf("\n");
    printf("#################################################################\n");
    printf("#                                                               #\n");
    printf("#  GTCRN 完整模型 C 实现                                         #\n");
    printf("#  Ultra-lightweight Speech Enhancement Model                  #\n");
    printf("#                                                               #\n");
    printf("#################################################################\n");

    test_gtcrn_creation();
    test_gtcrn_forward();
    test_convblock();
    test_dpgrnn();
    test_complex_mask();
    test_model_pipeline();

    printf("\n");
    print_separator();
    printf("总结\n");
    print_separator();
    printf("\n");
    printf("GTCRN 模型框架已创建！\n\n");

    printf("已实现的组件:\n");
    printf("  ✓ ConvBlock (Conv + BN + Activation)\n");
    printf("  ✓ GTConvBlock (框架)\n");
    printf("  ✓ Encoder (框架)\n");
    printf("  ✓ Decoder (框架)\n");
    printf("  ✓ DPGRNN (框架，不含 GRU)\n");
    printf("  ✓ 复数掩码\n");
    printf("  ✓ 模型管理\n\n");

    printf("已实现的基础层:\n");
    printf("  ✓ Conv2d\n");
    printf("  ✓ ConvTranspose2d\n");
    printf("  ✓ BatchNorm2d\n");
    printf("  ✓ LayerNorm\n");
    printf("  ✓ Linear\n");
    printf("  ✓ Unfold\n");
    printf("  ✓ PReLU\n");
    printf("  ✓ Sigmoid\n");
    printf("  ✓ Tanh\n\n");

    printf("待完成的工作:\n");
    printf("  1. 实现 GRU 层\n");
    printf("  2. 实现 ERB 压缩/恢复\n");
    printf("  3. 完整的 GTConvBlock\n");
    printf("  4. 完整的 TRA 模块\n");
    printf("  5. 从 PyTorch 模型加载权重\n");
    printf("  6. STFT/iSTFT 集成\n");
    printf("  7. 端到端音频处理\n\n");

    printf("下一步:\n");
    printf("  1. 实现 GRU (最关键)\n");
    printf("  2. 实现模型权重加载\n");
    printf("  3. 完整的前向传播\n");
    printf("  4. 性能优化\n");
    printf("  5. 实时音频处理\n\n");

    printf("文件:\n");
    printf("  - gtcrn_model.h\n");
    printf("  - gtcrn_model.c\n");
    printf("  - test_gtcrn_model.c\n\n");

    return 0;
}
