/**
 * stream_conv.c - Streaming Convolution Implementation
 *
 * 实现支持状态缓存的流式卷积操作
 */

#include "stream_conv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * 辅助函数实现
 * ============================================================================ */

void stream_conv_cache_init(
    float* cache,
    int batch,
    int channels,
    int cache_size,
    int freq_bins
) {
    int total_size = batch * channels * cache_size * freq_bins;
    memset(cache, 0, total_size * sizeof(float));
}

void stream_conv_cache_update(
    float* cache,
    const float* input,
    int batch,
    int channels,
    int cache_size,
    int input_time,
    int freq_bins
) {
    /*
     * 更新缓存策略:
     * 1. 将缓存向前移动1帧（丢弃最旧的帧）
     * 2. 将新输入添加到缓存末尾
     *
     * cache: (B, C, cache_size, F)
     * input: (B, C, input_time, F)
     *
     * 操作: cache = concat(cache[:,:,1:], input)[:,:,-cache_size:]
     */

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            // 移动缓存: 将 [1:cache_size] 移到 [0:cache_size-1]
            if (cache_size > 1) {
                memmove(
                    cache + (b * channels + c) * cache_size * freq_bins,
                    cache + (b * channels + c) * cache_size * freq_bins + freq_bins,
                    (cache_size - 1) * freq_bins * sizeof(float)
                );
            }

            // 添加新帧到缓存末尾
            // 如果input_time >= cache_size，只取最后cache_size帧
            int copy_frames = (input_time < cache_size) ? input_time : cache_size;
            int src_offset = (input_time > cache_size) ? (input_time - cache_size) : 0;

            for (int t = 0; t < copy_frames; t++) {
                int cache_idx = (b * channels + c) * cache_size * freq_bins +
                               (cache_size - copy_frames + t) * freq_bins;
                int input_idx = (b * channels + c) * input_time * freq_bins +
                               (src_offset + t) * freq_bins;

                memcpy(
                    cache + cache_idx,
                    input + input_idx,
                    freq_bins * sizeof(float)
                );
            }
        }
    }
}

/* ============================================================================
 * StreamConv2d 实现
 * ============================================================================ */

void stream_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,
    const Conv2dParams* conv_params
) {
    /*
     * 流式Conv2d前向传播
     *
     * 步骤:
     * 1. 拼接缓存和输入: inp = concat([cache, input], dim=2)
     * 2. 执行卷积: output = Conv2d(inp)
     * 3. 更新缓存: cache = inp[:,:,1:]
     */

    int batch = input->shape.batch;
    int in_channels = input->shape.channels;
    int input_time = input->shape.height;
    int freq_bins = input->shape.width;

    int kernel_h = conv_params->kernel_h;
    int dilation_h = conv_params->dilation_h;
    int cache_size = stream_conv_cache_size(kernel_h, dilation_h);

    // 验证参数
    if (conv_params->padding_h != 0) {
        printf("错误: StreamConv2d要求padding_h=0（因果性要求）\n");
        return;
    }

    // 1. 拼接缓存和输入
    int concat_time = cache_size + input_time;
    Tensor concat_input = {
        .data = (float*)malloc(batch * in_channels * concat_time * freq_bins * sizeof(float)),
        .shape = {
            .batch = batch,
            .channels = in_channels,
            .height = concat_time,
            .width = freq_bins
        }
    };

    // 复制缓存
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_channels; c++) {
            for (int t = 0; t < cache_size; t++) {
                int cache_idx = (b * in_channels + c) * cache_size * freq_bins + t * freq_bins;
                int concat_idx = (b * in_channels + c) * concat_time * freq_bins + t * freq_bins;
                memcpy(
                    concat_input.data + concat_idx,
                    conv_cache + cache_idx,
                    freq_bins * sizeof(float)
                );
            }
        }
    }

    // 复制输入
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_channels; c++) {
            for (int t = 0; t < input_time; t++) {
                int input_idx = (b * in_channels + c) * input_time * freq_bins + t * freq_bins;
                int concat_idx = (b * in_channels + c) * concat_time * freq_bins +
                                (cache_size + t) * freq_bins;
                memcpy(
                    concat_input.data + concat_idx,
                    input->data + input_idx,
                    freq_bins * sizeof(float)
                );
            }
        }
    }

    // 2. 执行卷积
    conv2d_forward(&concat_input, output, conv_params);

    // 3. 更新缓存: cache = concat_input[:,:,1:]
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_channels; c++) {
            for (int t = 0; t < cache_size; t++) {
                int concat_idx = (b * in_channels + c) * concat_time * freq_bins +
                                (t + 1) * freq_bins;  // 从第1帧开始（跳过第0帧）
                int cache_idx = (b * in_channels + c) * cache_size * freq_bins + t * freq_bins;
                memcpy(
                    conv_cache + cache_idx,
                    concat_input.data + concat_idx,
                    freq_bins * sizeof(float)
                );
            }
        }
    }

    // 清理
    free(concat_input.data);
}

/* ============================================================================
 * StreamConvTranspose2d 实现
 * ============================================================================ */

void stream_conv_transpose2d_forward(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,
    const Conv2dParams* conv_params
) {
    /*
     * 流式ConvTranspose2d前向传播
     *
     * 使用Conv2d实现ConvTranspose2d（权重时间反转）
     *
     * 步骤:
     * 1. 拼接缓存和输入: inp = concat([cache, input], dim=2)
     * 2. 更新缓存: cache = inp[:,:,1:]
     * 3. 频率维度处理（上采样和padding）
     * 4. 执行卷积（使用Conv2d）
     */

    int batch = input->shape.batch;
    int in_channels = input->shape.channels;
    int input_time = input->shape.height;
    int freq_bins = input->shape.width;

    int kernel_h = conv_params->kernel_h;
    int kernel_w = conv_params->kernel_w;
    int stride_w = conv_params->stride_w;
    int padding_w = conv_params->padding_w;
    int dilation_h = conv_params->dilation_h;
    int dilation_w = conv_params->dilation_w;

    int cache_size = stream_conv_cache_size(kernel_h, dilation_h);

    // 验证参数
    if (conv_params->stride_h != 1) {
        printf("错误: StreamConvTranspose2d要求stride_h=1\n");
        return;
    }
    if (conv_params->padding_h != 0) {
        printf("错误: StreamConvTranspose2d要求padding_h=0\n");
        return;
    }

    // 1. 拼接缓存和输入
    int concat_time = cache_size + input_time;
    float* concat_data = (float*)malloc(batch * in_channels * concat_time * freq_bins * sizeof(float));

    // 复制缓存和输入（与StreamConv2d相同）
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_channels; c++) {
            // 复制缓存
            for (int t = 0; t < cache_size; t++) {
                int cache_idx = (b * in_channels + c) * cache_size * freq_bins + t * freq_bins;
                int concat_idx = (b * in_channels + c) * concat_time * freq_bins + t * freq_bins;
                memcpy(concat_data + concat_idx, conv_cache + cache_idx, freq_bins * sizeof(float));
            }
            // 复制输入
            for (int t = 0; t < input_time; t++) {
                int input_idx = (b * in_channels + c) * input_time * freq_bins + t * freq_bins;
                int concat_idx = (b * in_channels + c) * concat_time * freq_bins +
                                (cache_size + t) * freq_bins;
                memcpy(concat_data + concat_idx, input->data + input_idx, freq_bins * sizeof(float));
            }
        }
    }

    // 2. 更新缓存
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_channels; c++) {
            for (int t = 0; t < cache_size; t++) {
                int concat_idx = (b * in_channels + c) * concat_time * freq_bins + (t + 1) * freq_bins;
                int cache_idx = (b * in_channels + c) * cache_size * freq_bins + t * freq_bins;
                memcpy(conv_cache + cache_idx, concat_data + concat_idx, freq_bins * sizeof(float));
            }
        }
    }

    // 3. 频率维度处理
    int freq_bins_upsampled = freq_bins;
    float* processed_data = concat_data;

    if (stride_w > 1) {
        // 频率维度上采样
        freq_bins_upsampled = freq_bins * stride_w;
        processed_data = (float*)calloc(batch * in_channels * concat_time * freq_bins_upsampled, sizeof(float));

        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < in_channels; c++) {
                for (int t = 0; t < concat_time; t++) {
                    for (int f = 0; f < freq_bins; f++) {
                        int src_idx = (b * in_channels + c) * concat_time * freq_bins + t * freq_bins + f;
                        int dst_idx = (b * in_channels + c) * concat_time * freq_bins_upsampled +
                                     t * freq_bins_upsampled + f * stride_w;
                        processed_data[dst_idx] = concat_data[src_idx];
                    }
                }
            }
        }
    }

    // 频率维度padding
    int pad_left = (kernel_w - 1) * dilation_w - padding_w;
    int pad_right = (kernel_w - 1) * dilation_w - padding_w;
    int freq_bins_padded = freq_bins_upsampled + pad_left + pad_right;

    float* padded_data = (float*)calloc(batch * in_channels * concat_time * freq_bins_padded, sizeof(float));

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < in_channels; c++) {
            for (int t = 0; t < concat_time; t++) {
                int src_idx = (b * in_channels + c) * concat_time * freq_bins_upsampled + t * freq_bins_upsampled;
                int dst_idx = (b * in_channels + c) * concat_time * freq_bins_padded +
                             t * freq_bins_padded + pad_left;
                memcpy(padded_data + dst_idx, processed_data + src_idx,
                       freq_bins_upsampled * sizeof(float));
            }
        }
    }

    // 4. 执行卷积
    Tensor padded_input = {
        .data = padded_data,
        .shape = {
            .batch = batch,
            .channels = in_channels,
            .height = concat_time,
            .width = freq_bins_padded
        }
    };

    conv2d_forward(&padded_input, output, conv_params);

    // 清理
    if (stride_w > 1 && processed_data != concat_data) {
        free(processed_data);
    }
    free(padded_data);
    free(concat_data);
}
