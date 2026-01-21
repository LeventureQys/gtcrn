#include "gtcrn_modules.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ============================================================================
// 辅助函数
// ============================================================================

float hz2erb(float freq_hz) {
    // ERB scale: erb_f = 21.4 * log10(0.00437 * freq_hz + 1)
    return 21.4f * log10f(0.00437f * freq_hz + 1.0f);
}

float erb2hz(float erb_f) {
    // Inverse: freq_hz = (10^(erb_f/21.4) - 1) / 0.00437
    return (powf(10.0f, erb_f / 21.4f) - 1.0f) / 0.00437f;
}

void generate_erb_filters(
    float* erb_filters,
    int erb_subband_1,
    int erb_subband_2,
    int nfft,
    int high_lim,
    int fs
) {
    int nfreqs = nfft / 2 + 1;
    int filter_width = nfreqs - erb_subband_1;

    // 初始化为 0
    memset(erb_filters, 0, erb_subband_2 * filter_width * sizeof(float));

    // 计算 ERB 频率点
    float low_lim = (float)erb_subband_1 / nfft * fs;
    float erb_low = hz2erb(low_lim);
    float erb_high = hz2erb((float)high_lim);

    // 生成 ERB 点
    float* erb_points = (float*)malloc(erb_subband_2 * sizeof(float));
    for (int i = 0; i < erb_subband_2; i++) {
        erb_points[i] = erb_low + (erb_high - erb_low) * i / (erb_subband_2 - 1);
    }

    // 转换为频率 bins
    int* bins = (int*)malloc(erb_subband_2 * sizeof(int));
    for (int i = 0; i < erb_subband_2; i++) {
        float freq = erb2hz(erb_points[i]);
        bins[i] = (int)roundf(freq / fs * nfft);
    }

    // 生成三角滤波器
    // 第一个滤波器
    for (int j = bins[0]; j < bins[1] && j < nfreqs; j++) {
        int idx = j - erb_subband_1;
        if (idx >= 0 && idx < filter_width) {
            erb_filters[0 * filter_width + idx] =
                (bins[1] - j + 1e-12f) / (bins[1] - bins[0] + 1e-12f);
        }
    }

    // 中间滤波器
    for (int i = 0; i < erb_subband_2 - 2; i++) {
        // 上升沿
        for (int j = bins[i]; j < bins[i+1] && j < nfreqs; j++) {
            int idx = j - erb_subband_1;
            if (idx >= 0 && idx < filter_width) {
                erb_filters[(i+1) * filter_width + idx] =
                    (j - bins[i] + 1e-12f) / (bins[i+1] - bins[i] + 1e-12f);
            }
        }
        // 下降沿
        for (int j = bins[i+1]; j < bins[i+2] && j < nfreqs; j++) {
            int idx = j - erb_subband_1;
            if (idx >= 0 && idx < filter_width) {
                erb_filters[(i+1) * filter_width + idx] =
                    (bins[i+2] - j + 1e-12f) / (bins[i+2] - bins[i+1] + 1e-12f);
            }
        }
    }

    // 最后一个滤波器
    for (int j = bins[erb_subband_2-2]; j <= bins[erb_subband_2-1] && j < nfreqs; j++) {
        int idx = j - erb_subband_1;
        if (idx >= 0 && idx < filter_width) {
            erb_filters[(erb_subband_2-1) * filter_width + idx] =
                1.0f - erb_filters[(erb_subband_2-2) * filter_width + idx];
        }
    }

    free(erb_points);
    free(bins);
}


// ============================================================================
// ERB 实现
// ============================================================================

ERBParams* erb_create(
    int erb_subband_1,
    int erb_subband_2,
    int nfft,
    int high_lim,
    int fs
) {
    ERBParams* params = (ERBParams*)malloc(sizeof(ERBParams));
    if (!params) return NULL;

    params->erb_subband_1 = erb_subband_1;
    params->erb_subband_2 = erb_subband_2;
    params->nfft = nfft;
    params->nfreqs = nfft / 2 + 1;

    int filter_width = params->nfreqs - erb_subband_1;

    // 分配滤波器
    params->erb_filters = (float*)malloc(erb_subband_2 * filter_width * sizeof(float));
    params->ierb_filters = (float*)malloc(erb_subband_2 * filter_width * sizeof(float));

    if (!params->erb_filters || !params->ierb_filters) {
        erb_free(params);
        return NULL;
    }

    // 生成 ERB 滤波器
    generate_erb_filters(params->erb_filters, erb_subband_1, erb_subband_2,
                        nfft, high_lim, fs);

    // 转置用于恢复
    for (int i = 0; i < erb_subband_2; i++) {
        for (int j = 0; j < filter_width; j++) {
            params->ierb_filters[j * erb_subband_2 + i] =
                params->erb_filters[i * filter_width + j];
        }
    }

    // 创建 Linear 参数
    params->erb_fc = linear_create(filter_width, erb_subband_2,
                                   params->erb_filters, NULL, 0);
    params->ierb_fc = linear_create(erb_subband_2, filter_width,
                                    params->ierb_filters, NULL, 0);

    printf("ERB 模块创建成功\n");
    printf("  低频保持: %d bins\n", erb_subband_1);
    printf("  ERB 压缩: %d bins\n", erb_subband_2);
    printf("  总输出: %d bins\n", erb_subband_1 + erb_subband_2);

    return params;
}

void erb_compress(
    const Tensor* input,
    Tensor* output,
    const ERBParams* params
) {
    // Input: (B, C, T, F) where F=769
    // Output: (B, C, T, F_erb) where F_erb=385

    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int time_steps = input->shape.height;
    int freq_bins = input->shape.width;

    int erb_subband_1 = params->erb_subband_1;
    int erb_subband_2 = params->erb_subband_2;

    // 对每个 (B, C, T) 样本
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time_steps; t++) {
                // 低频部分：直接复制
                for (int f = 0; f < erb_subband_1; f++) {
                    int in_idx = ((b * channels + c) * time_steps + t) * freq_bins + f;
                    int out_idx = ((b * channels + c) * time_steps + t) * (erb_subband_1 + erb_subband_2) + f;
                    output->data[out_idx] = input->data[in_idx];
                }

                // 高频部分：ERB 压缩
                float* high_freq_in = &input->data[((b * channels + c) * time_steps + t) * freq_bins + erb_subband_1];
                float* high_freq_out = &output->data[((b * channels + c) * time_steps + t) * (erb_subband_1 + erb_subband_2) + erb_subband_1];

                // 矩阵乘法: (1, nfreqs-erb_subband_1) @ (nfreqs-erb_subband_1, erb_subband_2)
                linear_forward(high_freq_in, high_freq_out, 1, params->erb_fc);
            }
        }
    }
}

void erb_decompress(
    const Tensor* input,
    Tensor* output,
    const ERBParams* params
) {
    // Input: (B, C, T, F_erb) where F_erb=385
    // Output: (B, C, T, F) where F=769

    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int time_steps = input->shape.height;
    int freq_bins_erb = input->shape.width;

    int erb_subband_1 = params->erb_subband_1;
    int erb_subband_2 = params->erb_subband_2;
    int freq_bins = params->nfreqs;

    // 对每个 (B, C, T) 样本
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time_steps; t++) {
                // 低频部分：直接复制
                for (int f = 0; f < erb_subband_1; f++) {
                    int in_idx = ((b * channels + c) * time_steps + t) * freq_bins_erb + f;
                    int out_idx = ((b * channels + c) * time_steps + t) * freq_bins + f;
                    output->data[out_idx] = input->data[in_idx];
                }

                // 高频部分：ERB 恢复
                float* high_freq_in = &input->data[((b * channels + c) * time_steps + t) * freq_bins_erb + erb_subband_1];
                float* high_freq_out = &output->data[((b * channels + c) * time_steps + t) * freq_bins + erb_subband_1];

                // 矩阵乘法: (1, erb_subband_2) @ (erb_subband_2, nfreqs-erb_subband_1)
                linear_forward(high_freq_in, high_freq_out, 1, params->ierb_fc);
            }
        }
    }
}

void erb_free(ERBParams* params) {
    if (params) {
        if (params->erb_filters) free(params->erb_filters);
        if (params->ierb_filters) free(params->ierb_filters);
        if (params->erb_fc) linear_free(params->erb_fc);
        if (params->ierb_fc) linear_free(params->ierb_fc);
        free(params);
    }
}


// ============================================================================
// SFE 实现
// ============================================================================

SFEParams* sfe_create(int kernel_size, int stride) {
    SFEParams* params = (SFEParams*)malloc(sizeof(SFEParams));
    if (!params) return NULL;

    params->kernel_size = kernel_size;

    // 设置 Unfold 参数
    params->unfold_params.kernel_h = 1;
    params->unfold_params.kernel_w = kernel_size;
    params->unfold_params.stride_h = 1;
    params->unfold_params.stride_w = stride;
    params->unfold_params.padding_h = 0;
    params->unfold_params.padding_w = (kernel_size - 1) / 2;
    params->unfold_params.dilation_h = 1;
    params->unfold_params.dilation_w = 1;

    printf("SFE 模块创建成功\n");
    printf("  kernel_size: %d\n", kernel_size);
    printf("  stride: %d\n", stride);

    return params;
}

void sfe_forward(
    const Tensor* input,
    Tensor* output,
    const SFEParams* params
) {
    // Input: (B, C, T, F)
    // Output: (B, C*kernel_size, T, F)

    unfold_reshape_4d(input, output, &params->unfold_params);
}

void sfe_free(SFEParams* params) {
    if (params) {
        free(params);
    }
}


// ============================================================================
// GRU 实现（简化版本）
// ============================================================================

GRUParams* gru_create(
    int input_size,
    int hidden_size,
    int num_layers,
    int bidirectional
) {
    GRUParams* params = (GRUParams*)malloc(sizeof(GRUParams));
    if (!params) return NULL;

    params->input_size = input_size;
    params->hidden_size = hidden_size;
    params->num_layers = num_layers;
    params->bidirectional = bidirectional;

    // 分配权重（简化版本，实际需要更复杂的结构）
    int weight_ih_size = hidden_size * input_size * 3;  // 3 gates
    int weight_hh_size = hidden_size * hidden_size * 3;

    params->weight_ih = (float*)malloc(weight_ih_size * sizeof(float));
    params->weight_hh = (float*)malloc(weight_hh_size * sizeof(float));
    params->bias_ih = (float*)malloc(hidden_size * 3 * sizeof(float));
    params->bias_hh = (float*)malloc(hidden_size * 3 * sizeof(float));

    if (!params->weight_ih || !params->weight_hh ||
        !params->bias_ih || !params->bias_hh) {
        gru_free(params);
        return NULL;
    }

    // 初始化为 0（实际应该从模型加载）
    memset(params->weight_ih, 0, weight_ih_size * sizeof(float));
    memset(params->weight_hh, 0, weight_hh_size * sizeof(float));
    memset(params->bias_ih, 0, hidden_size * 3 * sizeof(float));
    memset(params->bias_hh, 0, hidden_size * 3 * sizeof(float));

    printf("GRU 模块创建成功（简化版本）\n");
    printf("  input_size: %d\n", input_size);
    printf("  hidden_size: %d\n", hidden_size);
    printf("  注意: 需要从模型文件加载权重\n");

    return params;
}

// 注意: gru_forward 函数已在 GRU.c 中实现
// gtcrn_modules.c 使用 GRU.c 中的完整 GRU 实现

void gru_free(GRUParams* params) {
    if (params) {
        if (params->weight_ih) free(params->weight_ih);
        if (params->weight_hh) free(params->weight_hh);
        if (params->bias_ih) free(params->bias_ih);
        if (params->bias_hh) free(params->bias_hh);
        free(params);
    }
}


// ============================================================================
// TRA 实现
// ============================================================================

void compute_energy(
    const Tensor* input,
    float* energy,
    int batch,
    int channels,
    int time_steps,
    int freq_bins
) {
    // 计算能量: zt = mean(x^2, dim=-1)
    // Input: (B, C, T, F)
    // Output: (B, C, T)

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time_steps; t++) {
                double sum = 0.0;
                for (int f = 0; f < freq_bins; f++) {
                    int idx = ((b * channels + c) * time_steps + t) * freq_bins + f;
                    float val = input->data[idx];
                    sum += val * val;
                }
                energy[(b * channels + c) * time_steps + t] = (float)(sum / freq_bins);
            }
        }
    }
}

TRAParams* tra_create(int channels) {
    TRAParams* params = (TRAParams*)malloc(sizeof(TRAParams));
    if (!params) return NULL;

    params->channels = channels;

    // 创建 GRU: (channels, channels*2) - 使用GRU.c中的完整实现
    params->att_gru_weights = gru_weights_create(channels, channels * 2);
    if (!params->att_gru_weights) {
        fprintf(stderr, "Error: tra_create - failed to create GRU weights\n");
        free(params);
        return NULL;
    }

    // 创建 Linear: (channels*2, channels)
    // 注意: 权重需要从模型加载
    float* weight = (float*)calloc(channels * channels * 2, sizeof(float));
    if (!weight) {
        fprintf(stderr, "Error: tra_create - failed to allocate linear weight\n");
        gru_weights_free(params->att_gru_weights);
        free(params);
        return NULL;
    }
    params->att_fc = linear_create(channels * 2, channels, weight, NULL, 0);
    free(weight);

    if (!params->att_fc) {
        fprintf(stderr, "Error: tra_create - failed to create linear layer\n");
        gru_weights_free(params->att_gru_weights);
        free(params);
        return NULL;
    }

    printf("TRA 模块创建成功（使用完整GRU实现）\n");
    printf("  channels: %d\n", channels);
    printf("  GRU: input_size=%d, hidden_size=%d\n", channels, channels * 2);
    printf("  注意: 需要从模型文件加载 GRU 和 Linear 权重\n");

    return params;
}

void tra_forward(
    const Tensor* input,
    Tensor* output,
    TRAParams* params
) {
    // 参数验证
    if (!input || !output || !params) {
        fprintf(stderr, "Error: tra_forward - NULL parameter\n");
        return;
    }
    if (!params->att_gru_weights) {
        fprintf(stderr, "Error: tra_forward - att_gru_weights is NULL\n");
        return;
    }

    // Input: (B, C, T, F)
    // Output: (B, C, T, F) - 应用注意力权重

    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int time_steps = input->shape.height;
    int freq_bins = input->shape.width;

    // 1. 计算能量: zt = mean(x^2, dim=-1)  # (B,C,T)
    float* energy = (float*)malloc(batch * channels * time_steps * sizeof(float));
    compute_energy(input, energy, batch, channels, time_steps, freq_bins);

    // 2. GRU: at = GRU(zt)  # (B,C,T) -> (B,C*2,T)
    // 需要转置: (B,C,T) -> (B,T,C)
    float* gru_input = (float*)malloc(batch * time_steps * channels * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time_steps; t++) {
                gru_input[(b * time_steps + t) * channels + c] =
                    energy[(b * channels + c) * time_steps + t];
            }
        }
    }

    float* gru_output = (float*)malloc(batch * time_steps * channels * 2 * sizeof(float));
    float* temp = (float*)malloc(4 * channels * 2 * sizeof(float));

    // 使用GRU.c中的完整GRU实现
    for (int b = 0; b < batch; b++) {
        gru_forward(
            gru_input + b * time_steps * channels,
            gru_output + b * time_steps * channels * 2,
            NULL,  // 无初始隐藏状态
            params->att_gru_weights,
            time_steps,
            temp
        );
    }

    free(temp);

    // 3. Linear: at = Linear(at)  # (B,T,C*2) -> (B,T,C)
    float* linear_output = (float*)malloc(batch * time_steps * channels * sizeof(float));
    linear_forward(gru_output, linear_output, batch * time_steps, params->att_fc);

    // 4. Sigmoid: at = Sigmoid(at)
    sigmoid_forward(linear_output, batch * time_steps * channels);

    // 5. 转置回来: (B,T,C) -> (B,C,T)
    float* attention = (float*)malloc(batch * channels * time_steps * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < time_steps; t++) {
            for (int c = 0; c < channels; c++) {
                attention[(b * channels + c) * time_steps + t] =
                    linear_output[(b * time_steps + t) * channels + c];
            }
        }
    }

    // 6. 应用注意力: output = input * attention[..., None]
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time_steps; t++) {
                float att_weight = attention[(b * channels + c) * time_steps + t];
                for (int f = 0; f < freq_bins; f++) {
                    int idx = ((b * channels + c) * time_steps + t) * freq_bins + f;
                    output->data[idx] = input->data[idx] * att_weight;
                }
            }
        }
    }

    // 清理
    free(energy);
    free(gru_input);
    free(gru_output);
    free(linear_output);
    free(attention);
}

void tra_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* h_cache,
    TRAParams* params
) {
    /*
     * 流式TRA前向传播，支持状态缓存
     *
     * Input: (B, C, T, F) - 通常 T=1 用于实时处理
     * Output: (B, C, T, F) - 应用注意力权重后
     * h_cache: (1, B, channels*2) - GRU隐藏状态，会被更新
     *
     * 与批处理版本的区别：
     * 1. 接受并更新 h_cache，保持GRU状态在帧之间传递
     * 2. 优化用于 T=1 的情况（单帧处理）
     */

    // 参数验证
    if (!input || !output || !params) {
        fprintf(stderr, "Error: tra_forward_stream - NULL parameter\n");
        return;
    }
    if (!params->att_gru_weights) {
        fprintf(stderr, "Error: tra_forward_stream - att_gru_weights is NULL\n");
        return;
    }

    int batch = input->shape.batch;
    int channels = input->shape.channels;
    int time_steps = input->shape.height;
    int freq_bins = input->shape.width;

    // 1. 计算能量: zt = mean(x^2, dim=-1)  # (B,C,T)
    float* energy = (float*)malloc(batch * channels * time_steps * sizeof(float));
    compute_energy(input, energy, batch, channels, time_steps, freq_bins);

    // 2. GRU: at = GRU(zt)  # (B,C,T) -> (B,C*2,T)
    // 转置: (B,C,T) -> (B,T,C)
    float* gru_input = (float*)malloc(batch * time_steps * channels * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time_steps; t++) {
                gru_input[(b * time_steps + t) * channels + c] =
                    energy[(b * channels + c) * time_steps + t];
            }
        }
    }

    float* gru_output = (float*)malloc(batch * time_steps * channels * 2 * sizeof(float));
    float* temp = (float*)malloc(4 * channels * 2 * sizeof(float));

    // 使用GRU.c中的完整GRU实现，传入h_cache
    for (int b = 0; b < batch; b++) {
        // h_cache布局: (1, B, channels*2) -> 提取第b个batch的隐藏状态
        float* h_init = h_cache ? (h_cache + b * channels * 2) : NULL;

        gru_forward(
            gru_input + b * time_steps * channels,
            gru_output + b * time_steps * channels * 2,
            h_init,  // 使用缓存的隐藏状态
            params->att_gru_weights,
            time_steps,
            temp
        );

        // 更新h_cache为最后一个时间步的隐藏状态
        if (h_cache && time_steps > 0) {
            // 最后一个时间步的输出就是新的隐藏状态
            int last_t = time_steps - 1;
            for (int h = 0; h < channels * 2; h++) {
                h_cache[b * channels * 2 + h] =
                    gru_output[(b * time_steps + last_t) * channels * 2 + h];
            }
        }
    }

    free(temp);

    // 3. Linear: at = Linear(at)  # (B,T,C*2) -> (B,T,C)
    float* linear_output = (float*)malloc(batch * time_steps * channels * sizeof(float));
    linear_forward(gru_output, linear_output, batch * time_steps, params->att_fc);

    // 4. Sigmoid: at = Sigmoid(at)
    sigmoid_forward(linear_output, batch * time_steps * channels);

    // 5. 转置回来: (B,T,C) -> (B,C,T)
    float* attention = (float*)malloc(batch * channels * time_steps * sizeof(float));
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < time_steps; t++) {
            for (int c = 0; c < channels; c++) {
                attention[(b * channels + c) * time_steps + t] =
                    linear_output[(b * time_steps + t) * channels + c];
            }
        }
    }

    // 6. 应用注意力: output = input * attention[..., None]
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time_steps; t++) {
                float att_weight = attention[(b * channels + c) * time_steps + t];
                for (int f = 0; f < freq_bins; f++) {
                    int idx = ((b * channels + c) * time_steps + t) * freq_bins + f;
                    output->data[idx] = input->data[idx] * att_weight;
                }
            }
        }
    }

    // 清理
    free(energy);
    free(gru_input);
    free(gru_output);
    free(linear_output);
    free(attention);
}

void tra_free(TRAParams* params) {
    if (params) {
        if (params->att_gru_weights) gru_weights_free(params->att_gru_weights);
        if (params->att_fc) linear_free(params->att_fc);
        free(params);
    }
}
