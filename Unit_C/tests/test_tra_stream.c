/**
 * test_tra_stream.c - 测试TRA流式处理
 *
 * 演示如何使用 tra_forward_stream() 进行实时流式处理
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gtcrn_modules.h"

// 辅助函数：生成测试数据
void generate_test_data(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

// 辅助函数：计算两个数组的最大差异
float compute_max_diff(const float* a, const float* b, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int main() {
    printf("=================================================================\n");
    printf("TRA 流式处理测试\n");
    printf("=================================================================\n\n");

    // 测试参数
    int batch = 1;
    int channels = 8;  // 16的一半（GTConvBlock会分成两半）
    int freq_bins = 97;
    int total_frames = 10;  // 模拟10帧

    printf("测试配置:\n");
    printf("  Batch: %d\n", batch);
    printf("  Channels: %d\n", channels);
    printf("  Frequency bins: %d\n", freq_bins);
    printf("  Total frames: %d\n\n", total_frames);

    // 创建TRA模块
    printf("1. 创建TRA模块...\n");
    TRAParams* tra = tra_create(channels);
    if (!tra) {
        printf("错误: 无法创建TRA模块\n");
        return 1;
    }
    printf("   ✓ TRA模块创建成功\n\n");

    // 分配缓冲区
    int frame_size = batch * channels * 1 * freq_bins;  // T=1 (单帧)
    int batch_size = batch * channels * total_frames * freq_bins;  // T=total_frames (批处理)

    float* input_frame = (float*)malloc(frame_size * sizeof(float));
    float* output_frame = (float*)malloc(frame_size * sizeof(float));
    float* input_batch = (float*)malloc(batch_size * sizeof(float));
    float* output_batch = (float*)malloc(batch_size * sizeof(float));
    float* output_stream = (float*)malloc(batch_size * sizeof(float));

    // TRA缓存: (1, B, channels*2)
    int cache_size = 1 * batch * channels * 2;
    float* tra_cache = (float*)calloc(cache_size, sizeof(float));

    // 生成测试数据
    printf("2. 生成测试数据...\n");
    generate_test_data(input_batch, batch_size);
    printf("   ✓ 生成 %d 个样本\n\n", batch_size);

    // 测试1: 批处理模式
    printf("3. 测试批处理模式 (T=%d)...\n", total_frames);
    Tensor input_tensor_batch = {
        .data = input_batch,
        .shape = {
            .batch = batch,
            .channels = channels,
            .height = total_frames,
            .width = freq_bins
        }
    };
    Tensor output_tensor_batch = {
        .data = output_batch,
        .shape = {
            .batch = batch,
            .channels = channels,
            .height = total_frames,
            .width = freq_bins
        }
    };

    tra_forward(&input_tensor_batch, &output_tensor_batch, tra);
    printf("   ✓ 批处理完成\n\n");

    // 测试2: 流式处理模式
    printf("4. 测试流式处理模式 (逐帧处理)...\n");

    // 重置缓存
    memset(tra_cache, 0, cache_size * sizeof(float));

    for (int frame = 0; frame < total_frames; frame++) {
        // 提取当前帧
        for (int i = 0; i < frame_size; i++) {
            int batch_idx = frame * frame_size + i;
            input_frame[i] = input_batch[batch_idx];
        }

        // 创建单帧tensor
        Tensor input_tensor_frame = {
            .data = input_frame,
            .shape = {
                .batch = batch,
                .channels = channels,
                .height = 1,  // T=1
                .width = freq_bins
            }
        };
        Tensor output_tensor_frame = {
            .data = output_frame,
            .shape = {
                .batch = batch,
                .channels = channels,
                .height = 1,
                .width = freq_bins
            }
        };

        // 流式处理
        tra_forward_stream(&input_tensor_frame, &output_tensor_frame, tra_cache, tra);

        // 保存输出
        for (int i = 0; i < frame_size; i++) {
            int batch_idx = frame * frame_size + i;
            output_stream[batch_idx] = output_frame[i];
        }

        printf("   Frame %d/%d 处理完成\n", frame + 1, total_frames);
    }
    printf("   ✓ 流式处理完成\n\n");

    // 测试3: 比较结果
    printf("5. 比较批处理和流式处理结果...\n");
    float max_diff = compute_max_diff(output_batch, output_stream, batch_size);
    printf("   最大差异: %.2e\n", max_diff);

    if (max_diff < 1e-5f) {
        printf("   ✓ 测试通过! 流式处理与批处理结果一致\n\n");
    } else {
        printf("   ✗ 测试失败! 差异过大\n\n");
    }

    // 测试4: 验证状态缓存
    printf("6. 验证状态缓存...\n");
    int non_zero_count = 0;
    for (int i = 0; i < cache_size; i++) {
        if (fabsf(tra_cache[i]) > 1e-8f) {
            non_zero_count++;
        }
    }
    printf("   缓存大小: %d\n", cache_size);
    printf("   非零元素: %d (%.1f%%)\n", non_zero_count,
           100.0f * non_zero_count / cache_size);

    if (non_zero_count > 0) {
        printf("   ✓ 状态缓存正常工作\n\n");
    } else {
        printf("   ⚠ 警告: 状态缓存全为零（可能权重未加载）\n\n");
    }

    // 清理
    printf("7. 清理资源...\n");
    tra_free(tra);
    free(input_frame);
    free(output_frame);
    free(input_batch);
    free(output_batch);
    free(output_stream);
    free(tra_cache);
    printf("   ✓ 清理完成\n\n");

    printf("=================================================================\n");
    printf("测试完成\n");
    printf("=================================================================\n");

    return 0;
}
