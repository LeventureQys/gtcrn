/**
 * gtcrn_streaming_impl.c - GTCRN流式处理实现
 *
 * 实现所有模块的流式前向传播函数，支持实时音频处理
 */

#include "gtcrn_model.h"
#include "stream_conv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ============================================================================
 * DPGRNN 流式实现
 * ============================================================================ */

void dpgrnn_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* inter_cache,
    DPGRNN* dpgrnn
) {
    /*
     * DPGRNN 流式前向传播
     *
     * Input: (B, C, T, F) - 通常 T=1 用于实时处理
     * Output: (B, C, T, F)
     * inter_cache: (1, B*F, hidden_size) - Inter-RNN的隐藏状态
     *
     * 流程:
     * 1. Intra RNN: 双向处理频率维度（无需状态缓存）
     * 2. Inter RNN: 单向处理时间维度（使用状态缓存）
     */

    int B = input->shape.batch;
    int C = input->shape.channels;
    int T = input->shape.height;
    int F = input->shape.width;

    // 分配工作缓冲区
    float* x_btfc = (float*)malloc(B * T * F * C * sizeof(float));
    float* intra_out = (float*)malloc(B * T * F * C * sizeof(float));
    float* intra_x = (float*)malloc(B * T * F * C * sizeof(float));
    float* intra_residual = (float*)malloc(B * T * F * C * sizeof(float));
    float* inter_in = (float*)malloc(B * F * T * C * sizeof(float));
    float* inter_out = (float*)malloc(B * F * T * C * sizeof(float));
    float* inter_x = (float*)malloc(B * F * T * C * sizeof(float));
    float* temp = (float*)malloc(4 * dpgrnn->hidden_size * sizeof(float));

    // ========================================================================
    // Intra RNN (双向，无需状态缓存)
    // ========================================================================

    // 1. Permute: (B,C,T,F) -> (B,T,F,C)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                for (int c = 0; c < C; c++) {
                    int in_idx = b * (C * T * F) + c * (T * F) + t * F + f;
                    int out_idx = b * (T * F * C) + t * (F * C) + f * C + c;
                    x_btfc[out_idx] = input->data[in_idx];
                }
            }
        }
    }

    // 保存输入用于残差连接
    memcpy(intra_residual, x_btfc, B * T * F * C * sizeof(float));

    // 2. Process each (B*T) sample with bidirectional GRNN
    for (int bt = 0; bt < B * T; bt++) {
        const float* input_bt = x_btfc + bt * F * C;
        float* output_bt = intra_out + bt * F * C;

        // 使用完整的双向分组GRU
        grnn_bidirectional_forward_complete(
            input_bt,
            output_bt,
            NULL, NULL, NULL, NULL,  // 无初始隐藏状态
            dpgrnn->intra_gru_g1_fwd,
            dpgrnn->intra_gru_g2_fwd,
            dpgrnn->intra_gru_g1_bwd,
            dpgrnn->intra_gru_g2_bwd,
            F,  // 序列长度 = 频率bins
            temp
        );
    }

    // 3. Linear layer
    if (dpgrnn->intra_fc) {
        linear_forward(intra_out, intra_x, B * T * F, dpgrnn->intra_fc);
    } else {
        memcpy(intra_x, intra_out, B * T * F * C * sizeof(float));
    }

    // 4. LayerNorm
    if (dpgrnn->intra_ln) {
        // 应用LayerNorm: (B,T,F,C) with normalized_shape=(F,C)
        int norm_size = F * C;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* sample = intra_x + (b * T + t) * norm_size;

                // 计算均值
                float mean = 0.0f;
                for (int i = 0; i < norm_size; i++) {
                    mean += sample[i];
                }
                mean /= norm_size;

                // 计算方差
                float var = 0.0f;
                for (int i = 0; i < norm_size; i++) {
                    float diff = sample[i] - mean;
                    var += diff * diff;
                }
                var /= norm_size;

                // 归一化
                float std = sqrtf(var + dpgrnn->intra_ln->eps);
                for (int i = 0; i < norm_size; i++) {
                    sample[i] = (sample[i] - mean) / std;

                    // 应用可学习参数
                    if (dpgrnn->intra_ln->gamma) {
                        sample[i] *= dpgrnn->intra_ln->gamma[i];
                    }
                    if (dpgrnn->intra_ln->beta) {
                        sample[i] += dpgrnn->intra_ln->beta[i];
                    }
                }
            }
        }
    }

    // 5. Residual connection
    for (int i = 0; i < B * T * F * C; i++) {
        intra_x[i] += intra_residual[i];
    }

    // ========================================================================
    // Inter RNN (单向，使用状态缓存)
    // ========================================================================

    // 1. Permute: (B,T,F,C) -> (B,F,T,C)
    for (int b = 0; b < B; b++) {
        for (int f = 0; f < F; f++) {
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < C; c++) {
                    int in_idx = b * (T * F * C) + t * (F * C) + f * C + c;
                    int out_idx = b * (F * T * C) + f * (T * C) + t * C + c;
                    inter_in[out_idx] = intra_x[in_idx];
                }
            }
        }
    }

    // 2. Process each (B*F) sample with unidirectional GRNN (使用缓存)
    for (int bf = 0; bf < B * F; bf++) {
        const float* input_bf = inter_in + bf * T * C;
        float* output_bf = inter_out + bf * T * C;

        // 使用缓存的隐藏状态
        float* h_init = inter_cache ? (inter_cache + bf * C) : NULL;

        // Unidirectional GRNN across time dimension (causal)
        grnn_forward(
            input_bf,
            output_bf,
            h_init,  // 使用缓存的隐藏状态
            dpgrnn->inter_gru_g1,
            dpgrnn->inter_gru_g2,
            T,     // Sequence length = time steps
            0,     // Unidirectional (causal)
            temp
        );

        // 更新缓存为最后一个时间步的隐藏状态
        if (inter_cache && T > 0) {
            int last_t = T - 1;
            for (int c = 0; c < C; c++) {
                inter_cache[bf * C + c] = output_bf[last_t * C + c];
            }
        }
    }

    // 3. Linear layer
    if (dpgrnn->inter_fc) {
        linear_forward(inter_out, inter_x, B * F * T, dpgrnn->inter_fc);
    } else {
        memcpy(inter_x, inter_out, B * F * T * C * sizeof(float));
    }

    // 4. Permute: (B,F,T,C) -> (B,T,F,C)
    float* inter_x_btfc = (float*)malloc(B * T * F * C * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                for (int c = 0; c < C; c++) {
                    int in_idx = b * (F * T * C) + f * (T * C) + t * C + c;
                    int out_idx = b * (T * F * C) + t * (F * C) + f * C + c;
                    inter_x_btfc[out_idx] = inter_x[in_idx];
                }
            }
        }
    }

    // 5. LayerNorm
    if (dpgrnn->inter_ln) {
        int norm_size = F * C;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* sample = inter_x_btfc + (b * T + t) * norm_size;

                float mean = 0.0f;
                for (int i = 0; i < norm_size; i++) {
                    mean += sample[i];
                }
                mean /= norm_size;

                float var = 0.0f;
                for (int i = 0; i < norm_size; i++) {
                    float diff = sample[i] - mean;
                    var += diff * diff;
                }
                var /= norm_size;

                float std = sqrtf(var + dpgrnn->inter_ln->eps);
                for (int i = 0; i < norm_size; i++) {
                    sample[i] = (sample[i] - mean) / std;

                    if (dpgrnn->inter_ln->gamma) {
                        sample[i] *= dpgrnn->inter_ln->gamma[i];
                    }
                    if (dpgrnn->inter_ln->beta) {
                        sample[i] += dpgrnn->inter_ln->beta[i];
                    }
                }
            }
        }
    }

    // 6. Residual connection (加到Intra-RNN的输出上)
    for (int i = 0; i < B * T * F * C; i++) {
        inter_x_btfc[i] += intra_x[i];
    }

    // ========================================================================
    // Final permute back: (B,T,F,C) -> (B,C,T,F)
    // ========================================================================
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < T; t++) {
                for (int f = 0; f < F; f++) {
                    int in_idx = b * (T * F * C) + t * (F * C) + f * C + c;
                    int out_idx = b * (C * T * F) + c * (T * F) + t * F + f;
                    output->data[out_idx] = inter_x_btfc[in_idx];
                }
            }
        }
    }

    // 释放缓冲区
    free(x_btfc);
    free(intra_out);
    free(intra_x);
    free(intra_residual);
    free(inter_in);
    free(inter_out);
    free(inter_x);
    free(inter_x_btfc);
    free(temp);
}

/* ============================================================================
 * GTConvBlock 流式实现
 * ============================================================================ */

void gtconvblock_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,
    float* tra_cache,
    GTConvBlock* block,
    int kernel_h,
    int dilation_h
) {
    /*
     * GTConvBlock 流式前向传播
     *
     * 流程:
     * 1. Channel split: x -> x1, x2
     * 2. SFE: x1 -> x1_sfe
     * 3. Point Conv1 + PReLU: x1_sfe -> h1
     * 4. Depth Conv (流式): h1 -> h1 (使用conv_cache)
     * 5. PReLU: h1 -> h1
     * 6. Point Conv2: h1 -> h1
     * 7. TRA (流式): h1 -> h1 (使用tra_cache)
     * 8. Channel shuffle: (h1, x2) -> output
     */

    int B = input->shape.batch;
    int C = input->shape.channels;
    int T = input->shape.height;
    int F = input->shape.width;
    int C_half = C / 2;

    // 1. Channel split
    Tensor x1 = {
        .data = (float*)malloc(B * C_half * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
    };
    Tensor x2 = {
        .data = (float*)malloc(B * C_half * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
    };

    for (int i = 0; i < B * C_half * T * F; i++) {
        x1.data[i] = input->data[i];
        x2.data[i] = input->data[B * C_half * T * F + i];
    }

    // 2. SFE
    Tensor x1_sfe = {
        .data = (float*)malloc(B * C_half * 3 * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C_half * 3, .height = T, .width = F}
    };
    if (block->sfe) {
        sfe_forward(&x1, &x1_sfe, block->sfe);
    } else {
        memcpy(x1_sfe.data, x1.data, B * C_half * T * F * sizeof(float));
    }

    // 3. Point Conv1 + BN + PReLU
    Tensor h1 = {
        .data = (float*)malloc(B * 16 * T * F * sizeof(float)),  // hidden_channels=16
        .shape = {.batch = B, .channels = 16, .height = T, .width = F}
    };
    fused_conv_bn_forward(&x1_sfe, &h1, &block->point_conv1);
    if (block->point_prelu1) {
        prelu_forward_v2(&h1, block->point_prelu1);
    }

    // 4. Depth Conv (流式，使用缓存)
    Tensor h1_conv = {
        .data = (float*)malloc(B * 16 * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = F}
    };

    // 使用流式卷积
    Conv2dParams depth_conv_params;
    // 从block->depth_conv中提取参数
    // 注意: 这里需要根据实际的FusedConvBN结构来提取参数
    stream_conv2d_forward(&h1, &h1_conv, conv_cache, &depth_conv_params);

    // 5. BN + PReLU
    // 注意: BN已经在流式卷积中处理
    if (block->depth_prelu) {
        prelu_forward_v2(&h1_conv, block->depth_prelu);
    }

    // 6. Point Conv2 + BN
    Tensor h1_final = {
        .data = (float*)malloc(B * C_half * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
    };
    fused_conv_bn_forward(&h1_conv, &h1_final, &block->point_conv2);

    // 7. TRA (流式，使用缓存)
    if (block->tra && block->use_tra) {
        Tensor h1_tra = {
            .data = (float*)malloc(B * C_half * T * F * sizeof(float)),
            .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
        };
        tra_forward_stream(&h1_final, &h1_tra, tra_cache, block->tra);
        free(h1_final.data);
        h1_final = h1_tra;
    }

    // 8. Channel shuffle
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C_half; c++) {
            for (int t = 0; t < T; t++) {
                for (int f = 0; f < F; f++) {
                    // h1_final 的通道放在偶数位置
                    int out_idx1 = b * (C * T * F) + (2 * c) * (T * F) + t * F + f;
                    int in_idx1 = b * (C_half * T * F) + c * (T * F) + t * F + f;
                    output->data[out_idx1] = h1_final.data[in_idx1];

                    // x2 的通道放在奇数位置
                    int out_idx2 = b * (C * T * F) + (2 * c + 1) * (T * F) + t * F + f;
                    int in_idx2 = b * (C_half * T * F) + c * (T * F) + t * F + f;
                    output->data[out_idx2] = x2.data[in_idx2];
                }
            }
        }
    }

    // 清理
    free(x1.data);
    free(x2.data);
    free(x1_sfe.data);
    free(h1.data);
    free(h1_conv.data);
    free(h1_final.data);
}

/* ============================================================================
 * 完整GTCRN流式实现
 * ============================================================================ */

void gtcrn_forward_stream(
    const float* spec_input,
    float* spec_output,
    float* conv_cache,
    float* tra_cache,
    float* inter_cache,
    int batch,
    int freq_bins,
    GTCRN* model
) {
    /*
     * GTCRN 完整流式前向传播
     *
     * Input: spec_input (B, F, 1, 2) - 单帧复数频谱
     * Output: spec_output (B, F, 1, 2) - 增强后的单帧复数频谱
     *
     * 注意: 这是一个简化的实现框架
     * 完整实现需要:
     * 1. 正确管理所有缓存的布局和索引
     * 2. 实现Encoder和Decoder的流式版本
     * 3. 集成所有流式组件
     */

    printf("GTCRN流式前向传播 (单帧处理)\n");
    printf("输入: [%d, %d, 1, 2]\n", batch, freq_bins);

    // TODO: 实现完整的流式处理流程
    // 1. 输入预处理
    // 2. ERB压缩
    // 3. SFE
    // 4. Encoder (流式)
    // 5. DPGRNN (流式)
    // 6. Decoder (流式)
    // 7. ERB恢复
    // 8. 复数掩码

    printf("警告: gtcrn_forward_stream 尚未完全实现\n");
    printf("      请使用 gtcrn_streaming.h 中的高级接口\n");

    // 临时: 复制输入到输出
    memcpy(spec_output, spec_input, batch * freq_bins * 1 * 2 * sizeof(float));
}
