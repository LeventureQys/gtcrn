#include "gtcrn_modules.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void print_separator() {
    printf("=================================================================\n");
}

void test_erb() {
    printf("\n");
    print_separator();
    printf("Test 1: ERB (Equivalent Rectangular Bandwidth)\n");
    print_separator();
    printf("\n");

    printf("ERB 模块用于频率压缩和恢复\n");
    printf("从 gtcrn1.py lines 11-61\n\n");

    // 创建 ERB 参数
    int erb_subband_1 = 195;  // 低频保持
    int erb_subband_2 = 190;  // ERB 压缩
    int nfft = 1536;
    int high_lim = 24000;
    int fs = 48000;

    ERBParams* erb = erb_create(erb_subband_1, erb_subband_2, nfft, high_lim, fs);

    printf("\n配置:\n");
    printf("  输入频率bins: %d (48kHz, 1536 FFT)\n", nfft/2+1);
    printf("  低频保持: %d bins\n", erb_subband_1);
    printf("  ERB 压缩: %d bins\n", erb_subband_2);
    printf("  输出频率bins: %d\n\n", erb_subband_1 + erb_subband_2);

    // 测试压缩
    int batch = 1;
    int channels = 3;
    int time_steps = 63;
    int freq_bins = 769;

    Tensor* input = tensor_create(batch, channels, time_steps, freq_bins);
    Tensor* compressed = tensor_create(batch, channels, time_steps, 385);
    Tensor* decompressed = tensor_create(batch, channels, time_steps, freq_bins);

    // 填充输入
    srand(42);
    for (int i = 0; i < batch * channels * time_steps * freq_bins; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    printf("测试压缩和恢复:\n");
    printf("  输入: [%d, %d, %d, %d]\n", batch, channels, time_steps, freq_bins);

    // 压缩
    clock_t start = clock();
    erb_compress(input, compressed, erb);
    clock_t end = clock();
    printf("  压缩: [%d, %d, %d, %d] (%.2f ms)\n",
           batch, channels, time_steps, 385,
           (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // 恢复
    start = clock();
    erb_decompress(compressed, decompressed, erb);
    end = clock();
    printf("  恢复: [%d, %d, %d, %d] (%.2f ms)\n",
           batch, channels, time_steps, freq_bins,
           (double)(end - start) / CLOCKS_PER_SEC * 1000);

    // 计算误差
    double mse = 0.0;
    for (int i = 0; i < batch * channels * time_steps * freq_bins; i++) {
        float diff = input->data[i] - decompressed->data[i];
        mse += diff * diff;
    }
    mse /= (batch * channels * time_steps * freq_bins);

    printf("\n重建误差 (MSE): %.6e\n", mse);
    printf("说明: ERB 是有损压缩，会有一定误差\n");

    // 清理
    erb_free(erb);
    tensor_free(input);
    tensor_free(compressed);
    tensor_free(decompressed);
}

void test_sfe() {
    printf("\n");
    print_separator();
    printf("Test 2: SFE (Subband Feature Extraction)\n");
    print_separator();
    printf("\n");

    printf("SFE 模块使用 Unfold 提取子带特征\n");
    printf("从 gtcrn1.py lines 64-74\n\n");

    // 创建 SFE 参数
    int kernel_size = 3;
    int stride = 1;
    SFEParams* sfe = sfe_create(kernel_size, stride);

    printf("配置:\n");
    printf("  kernel_size: %d\n", kernel_size);
    printf("  stride: %d\n\n", stride);

    // 测试
    int batch = 1;
    int channels = 3;
    int time_steps = 63;
    int freq_bins = 385;

    Tensor* input = tensor_create(batch, channels, time_steps, freq_bins);
    Tensor* output = tensor_create(batch, channels * kernel_size, time_steps, freq_bins);

    // 填充输入
    srand(123);
    for (int i = 0; i < batch * channels * time_steps * freq_bins; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    printf("测试:\n");
    printf("  输入: [%d, %d, %d, %d]\n", batch, channels, time_steps, freq_bins);

    // 前向传播
    clock_t start = clock();
    sfe_forward(input, output, sfe);
    clock_t end = clock();

    printf("  输出: [%d, %d, %d, %d] (%.2f ms)\n",
           batch, channels * kernel_size, time_steps, freq_bins,
           (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\n说明:\n");
    printf("  - 在频率维度上提取邻域特征\n");
    printf("  - 每个位置提取 %d 个邻域值\n", kernel_size);
    printf("  - 通道数扩展: %d -> %d\n", channels, channels * kernel_size);

    // 清理
    sfe_free(sfe);
    tensor_free(input);
    tensor_free(output);
}

void test_gru() {
    printf("\n");
    print_separator();
    printf("Test 3: GRU (Gated Recurrent Unit)\n");
    print_separator();
    printf("\n");

    printf("GRU 是循环神经网络的一种\n");
    printf("用于 TRA 模块生成注意力权重\n\n");

    // 创建 GRU 参数
    int input_size = 8;
    int hidden_size = 16;
    int num_layers = 1;
    int bidirectional = 0;

    GRUParams* gru = gru_create(input_size, hidden_size, num_layers, bidirectional);

    printf("配置:\n");
    printf("  input_size: %d\n", input_size);
    printf("  hidden_size: %d\n", hidden_size);
    printf("  num_layers: %d\n", num_layers);
    printf("  bidirectional: %d\n\n", bidirectional);

    // 测试
    int batch = 2;
    int seq_len = 63;

    float* input = (float*)malloc(batch * seq_len * input_size * sizeof(float));
    float* output = (float*)malloc(batch * seq_len * hidden_size * sizeof(float));

    // 填充输入
    srand(456);
    for (int i = 0; i < batch * seq_len * input_size; i++) {
        input[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    printf("测试:\n");
    printf("  输入: [%d, %d, %d]\n", batch, seq_len, input_size);

    // 前向传播
    clock_t start = clock();
    gru_forward(input, output, NULL, batch, seq_len, gru);
    clock_t end = clock();

    printf("  输出: [%d, %d, %d] (%.2f ms)\n",
           batch, seq_len, hidden_size,
           (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\n注意:\n");
    printf("  当前是简化版本（直接复制）\n");
    printf("  完整实现需要:\n");
    printf("    1. Reset gate\n");
    printf("    2. Update gate\n");
    printf("    3. New gate\n");
    printf("    4. Hidden state update\n");
    printf("  需要从模型文件加载权重\n");

    // 清理
    gru_free(gru);
    free(input);
    free(output);
}

void test_tra() {
    printf("\n");
    print_separator();
    printf("Test 4: TRA (Temporal Recurrent Attention)\n");
    print_separator();
    printf("\n");

    printf("TRA 模块生成时间注意力权重\n");
    printf("从 gtcrn1.py lines 77-93\n\n");

    // 创建 TRA 参数
    int channels = 8;
    TRAParams* tra = tra_create(channels);

    printf("配置:\n");
    printf("  channels: %d\n\n", channels);

    // 测试
    int batch = 1;
    int time_steps = 63;
    int freq_bins = 97;

    Tensor* input = tensor_create(batch, channels, time_steps, freq_bins);
    Tensor* output = tensor_create(batch, channels, time_steps, freq_bins);

    // 填充输入
    srand(789);
    for (int i = 0; i < batch * channels * time_steps * freq_bins; i++) {
        input->data[i] = (float)rand() / RAND_MAX;
    }

    printf("测试:\n");
    printf("  输入: [%d, %d, %d, %d]\n", batch, channels, time_steps, freq_bins);

    // 前向传播
    clock_t start = clock();
    tra_forward(input, output, tra);
    clock_t end = clock();

    printf("  输出: [%d, %d, %d, %d] (%.2f ms)\n",
           batch, channels, time_steps, freq_bins,
           (double)(end - start) / CLOCKS_PER_SEC * 1000);

    printf("\n流程:\n");
    printf("  1. 计算能量: zt = mean(x^2, dim=-1)\n");
    printf("  2. GRU: at = GRU(zt)\n");
    printf("  3. Linear: at = Linear(at)\n");
    printf("  4. Sigmoid: at = Sigmoid(at)\n");
    printf("  5. 应用注意力: output = input * at\n");

    printf("\n注意:\n");
    printf("  当前 GRU 使用简化版本\n");
    printf("  完整实现需要训练好的 GRU 权重\n");

    // 清理
    tra_free(tra);
    tensor_free(input);
    tensor_free(output);
}

void test_hz_erb_conversion() {
    printf("\n");
    print_separator();
    printf("Test 5: Hz <-> ERB 转换\n");
    print_separator();
    printf("\n");

    printf("ERB (Equivalent Rectangular Bandwidth) 尺度\n");
    printf("模拟人耳的频率感知特性\n\n");

    printf("测试频率转换:\n");
    float test_freqs[] = {100, 500, 1000, 2000, 4000, 8000, 16000};
    int num_freqs = sizeof(test_freqs) / sizeof(float);

    printf("  Hz -> ERB -> Hz\n");
    for (int i = 0; i < num_freqs; i++) {
        float hz = test_freqs[i];
        float erb = hz2erb(hz);
        float hz_back = erb2hz(erb);

        printf("  %.0f Hz -> %.2f ERB -> %.0f Hz\n", hz, erb, hz_back);
    }

    printf("\n说明:\n");
    printf("  - ERB 尺度在低频更密集\n");
    printf("  - 符合人耳的频率分辨率\n");
    printf("  - 用于频率压缩，减少计算量\n");
}

void test_complete_pipeline() {
    printf("\n");
    print_separator();
    printf("Test 6: 完整流程测试\n");
    print_separator();
    printf("\n");

    printf("模拟 GTCRN 的完整数据流:\n\n");

    // 1. 输入
    printf("1. 输入频谱: (B, 3, T, 769)\n");
    printf("   - 3 通道: [magnitude, real, imag]\n\n");

    // 2. ERB 压缩
    printf("2. ERB 压缩: (B, 3, T, 769) -> (B, 3, T, 385)\n");
    ERBParams* erb = erb_create(195, 190, 1536, 24000, 48000);
    printf("\n");

    // 3. SFE
    printf("3. SFE: (B, 3, T, 385) -> (B, 9, T, 385)\n");
    SFEParams* sfe = sfe_create(3, 1);
    printf("\n");

    // 4. Encoder (省略)
    printf("4. Encoder: (B, 9, T, 385) -> (B, 16, T, 97)\n");
    printf("   - 5 层卷积\n\n");

    // 5. DPGRNN (省略)
    printf("5. DPGRNN: (B, 16, T, 97) -> (B, 16, T, 97)\n");
    printf("   - 2 层双路径 RNN\n\n");

    // 6. Decoder (省略)
    printf("6. Decoder: (B, 16, T, 97) -> (B, 2, T, 385)\n");
    printf("   - 5 层反卷积\n\n");

    // 7. ERB 恢复
    printf("7. ERB 恢复: (B, 2, T, 385) -> (B, 2, T, 769)\n\n");

    // 8. 复数掩码
    printf("8. 复数掩码: 应用到输入频谱\n\n");

    printf("输出: (B, 769, T, 2) 增强后的复数频谱\n");

    // 清理
    erb_free(erb);
    sfe_free(sfe);
}

int main() {
    printf("\n");
    printf("#################################################################\n");
    printf("#                                                               #\n");
    printf("#  GTCRN 模块 C 实现                                             #\n");
    printf("#  ERB, SFE, TRA                                                #\n");
    printf("#                                                               #\n");
    printf("#################################################################\n");

    test_erb();
    test_sfe();
    test_gru();
    test_tra();
    test_hz_erb_conversion();
    test_complete_pipeline();

    printf("\n");
    print_separator();
    printf("总结\n");
    print_separator();
    printf("\n");
    printf("已实现的 GTCRN 模块:\n\n");

    printf("✓ ERB (Equivalent Rectangular Bandwidth)\n");
    printf("  - 频率压缩: 769 bins -> 385 bins\n");
    printf("  - 频率恢复: 385 bins -> 769 bins\n");
    printf("  - 基于人耳感知的频率尺度\n\n");

    printf("✓ SFE (Subband Feature Extraction)\n");
    printf("  - 使用 Unfold 提取子带特征\n");
    printf("  - 在频率维度上提取邻域\n");
    printf("  - 通道扩展: C -> C*kernel_size\n\n");

    printf("✓ GRU (Gated Recurrent Unit)\n");
    printf("  - 循环神经网络\n");
    printf("  - 用于 TRA 模块\n");
    printf("  - 当前: 简化版本\n");
    printf("  - 需要: 完整实现 + 权重加载\n\n");

    printf("✓ TRA (Temporal Recurrent Attention)\n");
    printf("  - 生成时间注意力权重\n");
    printf("  - 流程: 能量 -> GRU -> Linear -> Sigmoid\n");
    printf("  - 应用注意力到输入\n\n");

    printf("完整的 GTCRN 组件:\n");
    printf("  ✓ Conv2d, ConvTranspose2d\n");
    printf("  ✓ BatchNorm2d, LayerNorm\n");
    printf("  ✓ Linear, Unfold\n");
    printf("  ✓ PReLU, Sigmoid, Tanh\n");
    printf("  ✓ ERB, SFE, TRA\n");
    printf("  ⏳ GRU (需要完整实现)\n\n");

    printf("下一步:\n");
    printf("  1. 完整实现 GRU\n");
    printf("  2. 从 PyTorch 模型加载权重\n");
    printf("  3. 集成所有模块\n");
    printf("  4. 端到端测试\n\n");

    printf("文件:\n");
    printf("  - gtcrn_modules.h\n");
    printf("  - gtcrn_modules.c\n");
    printf("  - test_gtcrn_modules.c\n\n");

    return 0;
}
