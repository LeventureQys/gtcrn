/// <file>gtcrn_forward.c</file>
/// <summary>GTCRN完整前向传播实现</summary>
/// <author>江月希 李文轩</author>

#include "gtcrn_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* 调试标志 - 设为1启用调试输出 */
#ifndef GTCRN_DEBUG
#define GTCRN_DEBUG 1
#endif

/* 调试辅助函数 */
static double debug_tensor_sum(const gtcrn_float* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

#if GTCRN_DEBUG
#define DEBUG_PRINT_SUM(name, data, size) \
    printf("  [C] %s: sum=%.6f\n", name, debug_tensor_sum(data, size))
#else
#define DEBUG_PRINT_SUM(name, data, size)
#endif

// 层参数辅助结构体

typedef struct {
    const gtcrn_float* conv_weight;
    const gtcrn_float* conv_bias;
    const gtcrn_float* bn_gamma;
    const gtcrn_float* bn_beta;
    const gtcrn_float* bn_mean;
    const gtcrn_float* bn_var;
    const gtcrn_float* prelu_alpha;
    int in_ch, out_ch, kh, kw, sh, sw, ph, pw, groups;
    int is_deconv;
    int use_tanh;  /* 最后一层使用tanh激活 */
} conv_block_params_t;

typedef struct {
    /* Point conv 1 */
    const gtcrn_float* pc1_weight;
    const gtcrn_float* pc1_bias;
    const gtcrn_float* bn1_gamma;
    const gtcrn_float* bn1_beta;
    const gtcrn_float* bn1_mean;
    const gtcrn_float* bn1_var;
    const gtcrn_float* prelu1;

    /* Depth conv */
    const gtcrn_float* dc_weight;
    const gtcrn_float* dc_bias;
    const gtcrn_float* bn2_gamma;
    const gtcrn_float* bn2_beta;
    const gtcrn_float* bn2_mean;
    const gtcrn_float* bn2_var;
    const gtcrn_float* prelu2;

    /* Point conv 2 */
    const gtcrn_float* pc2_weight;
    const gtcrn_float* pc2_bias;
    const gtcrn_float* bn3_gamma;
    const gtcrn_float* bn3_beta;
    const gtcrn_float* bn3_mean;
    const gtcrn_float* bn3_var;

    /* TRA */
    const gtcrn_float* tra_gru_ih;
    const gtcrn_float* tra_gru_hh;
    const gtcrn_float* tra_gru_bih;
    const gtcrn_float* tra_gru_bhh;
    const gtcrn_float* tra_fc_weight;
    const gtcrn_float* tra_fc_bias;

    int dilation;
    int is_deconv;
} gtconv_block_params_t;

// ConvBlock前向传播 (Conv2d + BN + PReLU/Tanh)

static void conv_block_forward(const conv_block_params_t* p,
                               const gtcrn_float* input,
                               gtcrn_float* output,
                               gtcrn_float* workspace,
                               int batch, int in_h, int in_w,
                               int out_h, int out_w) {
    gtcrn_conv2d_t conv = {
        .weight = p->conv_weight,
        .bias = p->conv_bias,
        .in_channels = p->in_ch,
        .out_channels = p->out_ch,
        .kernel_h = p->kh,
        .kernel_w = p->kw,
        .stride_h = p->sh,
        .stride_w = p->sw,
        .padding_h = p->ph,
        .padding_w = p->pw,
        .dilation_h = 1,
        .dilation_w = 1,
        .groups = p->groups
    };

    if (p->is_deconv) {
        gtcrn_conv_transpose2d_forward(&conv, input, workspace, batch, in_h, in_w, out_h, out_w);
    } else {
        gtcrn_conv2d_forward(&conv, input, workspace, batch, in_h, in_w, out_h, out_w);
    }

    /* 批归一化 */
    gtcrn_batchnorm2d_t bn = {
        .gamma = p->bn_gamma,
        .beta = p->bn_beta,
        .running_mean = p->bn_mean,
        .running_var = p->bn_var,
        .num_features = p->out_ch,
        .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn, workspace, batch, out_h, out_w);

    /* 激活函数 */
    int total = batch * p->out_ch * out_h * out_w;
    if (p->use_tanh) {
        for (int i = 0; i < total; i++) {
            output[i] = tanhf(workspace[i]);
        }
    } else {
        gtcrn_prelu_t prelu = {
            .alpha = p->prelu_alpha,
            .num_parameters = 1  /* GTCRN中PReLU使用共享alpha */
        };
        memcpy(output, workspace, total * sizeof(gtcrn_float));
        gtcrn_prelu_forward(&prelu, output, batch, p->out_ch, out_h * out_w);
    }
}

// SFE前向传播 (子带特征提取)

static void sfe_forward(const gtcrn_float* input,
                        gtcrn_float* output,
                        int batch, int channels, int time, int freq) {
    int out_ch = channels * 3;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < freq; f++) {
                    int f_left = f - 1;
                    int f_right = f + 1;

                    gtcrn_float v_left = (f_left >= 0) ?
                        input[GTCRN_IDX4(b, c, t, f_left, channels, time, freq)] : 0.0f;
                    gtcrn_float v_center = input[GTCRN_IDX4(b, c, t, f, channels, time, freq)];
                    gtcrn_float v_right = (f_right < freq) ?
                        input[GTCRN_IDX4(b, c, t, f_right, channels, time, freq)] : 0.0f;

                    output[GTCRN_IDX4(b, c * 3 + 0, t, f, out_ch, time, freq)] = v_left;
                    output[GTCRN_IDX4(b, c * 3 + 1, t, f, out_ch, time, freq)] = v_center;
                    output[GTCRN_IDX4(b, c * 3 + 2, t, f, out_ch, time, freq)] = v_right;
                }
            }
        }
    }
}

// TRA前向传播 (时序循环注意力)

static void tra_forward(const gtcrn_float* tra_gru_ih,
                        const gtcrn_float* tra_gru_hh,
                        const gtcrn_float* tra_gru_bih,
                        const gtcrn_float* tra_gru_bhh,
                        const gtcrn_float* tra_fc_weight,
                        const gtcrn_float* tra_fc_bias,
                        gtcrn_float* x,  /* In-place: (B, C, T, F) */
                        int batch, int channels, int time, int freq,
                        gtcrn_float* workspace) {
    /* 计算zt = mean(x^2, dim=-1): (B, C, T) */
    gtcrn_float* zt = workspace;
    int gru_hidden = channels * 2;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                gtcrn_float sum = 0.0f;
                for (int f = 0; f < freq; f++) {
                    gtcrn_float val = x[GTCRN_IDX4(b, c, t, f, channels, time, freq)];
                    sum += val * val;
                }
                zt[(b * channels + c) * time + t] = sum / freq;
            }
        }
    }

    /* GRU前向传播: 输入(B, T, C), 输出(B, T, 2C) */
    gtcrn_float* gru_out = zt + batch * channels * time;
    gtcrn_float* h_prev = gru_out + batch * time * gru_hidden;
    gtcrn_float* h_curr = h_prev + gru_hidden;
    gtcrn_float* gru_workspace = h_curr + gru_hidden;

    gtcrn_vec_zero(h_prev, gru_hidden);

    for (int t = 0; t < time; t++) {
        /* 准备当前时间步输入: (B*C,) -> 转置为(B, C)
         * x_t不能与gru_out重叠,放在gates缓冲区之后
         */
        gtcrn_float* x_t = gru_workspace + 3 * gru_hidden * 2;
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                x_t[b * channels + c] = zt[(b * channels + c) * time + t];
            }
        }

        /* GRU cell */
        for (int b = 0; b < batch; b++) {
            gtcrn_float* gates_ih = gru_workspace;
            gtcrn_float* gates_hh = gru_workspace + 3 * gru_hidden;

            /* W_ih @ x + b_ih */
            for (int i = 0; i < 3 * gru_hidden; i++) {
                gtcrn_float sum = tra_gru_bih ? tra_gru_bih[i] : 0.0f;
                for (int j = 0; j < channels; j++) {
                    sum += tra_gru_ih[i * channels + j] * x_t[b * channels + j];
                }
                gates_ih[i] = sum;
            }

            /* W_hh @ h + b_hh */
            for (int i = 0; i < 3 * gru_hidden; i++) {
                gtcrn_float sum = tra_gru_bhh ? tra_gru_bhh[i] : 0.0f;
                for (int j = 0; j < gru_hidden; j++) {
                    sum += tra_gru_hh[i * gru_hidden + j] * h_prev[j];
                }
                gates_hh[i] = sum;
            }

            /* Compute gates and new hidden state */
            for (int i = 0; i < gru_hidden; i++) {
                gtcrn_float r = gtcrn_sigmoid(gates_ih[i] + gates_hh[i]);
                gtcrn_float z = gtcrn_sigmoid(gates_ih[gru_hidden + i] + gates_hh[gru_hidden + i]);
                gtcrn_float n = gtcrn_tanh(gates_ih[2 * gru_hidden + i] + r * gates_hh[2 * gru_hidden + i]);
                h_curr[i] = (1.0f - z) * n + z * h_prev[i];
            }

            /* Store output */
            for (int i = 0; i < gru_hidden; i++) {
                gru_out[(b * time + t) * gru_hidden + i] = h_curr[i];
            }

            /* Swap h_prev and h_curr */
            gtcrn_float* tmp = h_prev;
            h_prev = h_curr;
            h_curr = tmp;
        }
    }

    /* 全连接层: (B, T, 2C) -> (B, T, C) */
    gtcrn_float* at = workspace;
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < time; t++) {
            for (int c = 0; c < channels; c++) {
                gtcrn_float sum = tra_fc_bias[c];
                for (int j = 0; j < gru_hidden; j++) {
                    sum += tra_fc_weight[c * gru_hidden + j] * gru_out[(b * time + t) * gru_hidden + j];
                }
                /* 存储为(B, C, T)布局以匹配PyTorch转置后的结果 */
                at[(b * channels + c) * time + t] = gtcrn_sigmoid(sum);
            }
        }
    }

    /* 应用注意力: x = x * at[..., None] */
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                gtcrn_float alpha = at[(b * channels + c) * time + t];
                for (int f = 0; f < freq; f++) {
                    x[GTCRN_IDX4(b, c, t, f, channels, time, freq)] *= alpha;
                }
            }
        }
    }
}

// 通道混洗
static void channel_shuffle(const gtcrn_float* x1, const gtcrn_float* x2,
                            gtcrn_float* output,
                            int batch, int half_ch, int time, int freq) {
    int out_ch = half_ch * 2;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < half_ch; c++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < freq; f++) {
                    output[GTCRN_IDX4(b, c * 2, t, f, out_ch, time, freq)] =
                        x1[GTCRN_IDX4(b, c, t, f, half_ch, time, freq)];
                    output[GTCRN_IDX4(b, c * 2 + 1, t, f, out_ch, time, freq)] =
                        x2[GTCRN_IDX4(b, c, t, f, half_ch, time, freq)];
                }
            }
        }
    }
}

// GTConvBlock前向传播
static void gtconv_block_forward(const gtconv_block_params_t* p,
                                 gtcrn_float* x,  /* In-place: (B, 16, T, F) */
                                 int batch, int time, int freq,
                                 gtcrn_float* workspace) {
    int channels = 16;
    int half_ch = channels / 2;
    int half_size = half_ch * time * freq;     /* 8*T*F = 2640 for T=10, F=33 */
    int full_size = channels * time * freq;    /* 16*T*F = 5280 */
    int sfe_size = 24 * time * freq;           /* 24*T*F = 7920 */

    /* Calculate padded time for this dilation */
    int pad_t = (3 - 1) * p->dilation;  /* (kT - 1) * dT */
    int padded_time = time + pad_t;
    int padded_size = channels * padded_time * freq;  /* 16 * (T+pad) * F */

    /* 工作空间布局(使用期间不能重叠):
     * - x1: [0, half_size) - 输入的前半部分
     * - x2: [half_size, 2*half_size) - 输入的后半部分,混洗前保持不变
     * - x1_sfe: [2*half_size, 2*half_size + sfe_size) - SFE输出
     * - h1: [2*half_size + sfe_size, 2*half_size + sfe_size + full_size) - 卷积输出
     * - h1_padded: [2*half_size + sfe_size + full_size, ... + padded_size) - 深度卷积的填充输入
     * - h1_out: h1_padded之后
     * - tra_workspace: h1_out之后
     */

    gtcrn_float* x1 = workspace;                            /* [0, 2640) */
    gtcrn_float* x2 = x1 + half_size;                       /* [2640, 5280) */
    gtcrn_float* x1_sfe = x2 + half_size;                   /* [5280, 13200) */
    gtcrn_float* h1 = x1_sfe + sfe_size;                    /* [13200, 18480) */
    gtcrn_float* h1_padded = h1 + full_size;                /* [18480, 18480 + padded_size) */
    gtcrn_float* h1_out = h1_padded + padded_size;          /* After h1_padded */
    gtcrn_float* tra_workspace = h1_out + half_size;        /* After h1_out */

    /* 通道分割 */
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < half_ch; c++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < freq; f++) {
                    x1[GTCRN_IDX4(b, c, t, f, half_ch, time, freq)] =
                        x[GTCRN_IDX4(b, c, t, f, channels, time, freq)];
                    x2[GTCRN_IDX4(b, c, t, f, half_ch, time, freq)] =
                        x[GTCRN_IDX4(b, c + half_ch, t, f, channels, time, freq)];
                }
            }
        }
    }

    /* x1的SFE: (B, 8, T, F) -> (B, 24, T, F) */
    sfe_forward(x1, x1_sfe, batch, half_ch, time, freq);

    /* 点卷积1: (B, 24, T, F) -> (B, 16, T, F) */
    gtcrn_conv2d_t pc1 = {
        .weight = p->pc1_weight, .bias = p->pc1_bias,
        .in_channels = 24, .out_channels = 16,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1
    };

    if (p->is_deconv) {
        gtcrn_conv_transpose2d_forward(&pc1, x1_sfe, h1, batch, time, freq, time, freq);
    } else {
        gtcrn_conv2d_forward(&pc1, x1_sfe, h1, batch, time, freq, time, freq);
    }

    /* BN1 + PReLU1 */
    gtcrn_batchnorm2d_t bn1 = {
        .gamma = p->bn1_gamma, .beta = p->bn1_beta,
        .running_mean = p->bn1_mean, .running_var = p->bn1_var,
        .num_features = 16, .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn1, h1, batch, time, freq);

    gtcrn_prelu_t prelu1 = { .alpha = p->prelu1, .num_parameters = 1 };
    gtcrn_prelu_forward(&prelu1, h1, batch, 16, time * freq);

    /* 深度卷积处理 - 编码器和解码器都需要手动因果填充
       关键区别是卷积类型:
       - 编码器: Conv2d, padding=(0,1)
       - 解码器: ConvTranspose2d, padding=(2*dilation,1)裁剪输出 */

    /* 手动因果填充: F.pad(h1, [0, 0, pad_size, 0])
       h1_padded缓冲区在函数开始时已分配 */
    gtcrn_vec_zero(h1_padded, batch * 16 * padded_time * freq);
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < 16; c++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < freq; f++) {
                    h1_padded[GTCRN_IDX4(b, c, t + pad_t, f, 16, padded_time, freq)] =
                        h1[GTCRN_IDX4(b, c, t, f, 16, time, freq)];
                }
            }
        }
    }

    gtcrn_float* dc_out = h1;  /* Can reuse h1 buffer */

    if (p->is_deconv) {
        /* 解码器: ConvTranspose2d, padding=(2*dilation, 1)
           输入: (B, 16, T+pad_t, F), 输出: (B, 16, T, F)
           ConvTranspose2d的padding参数会裁剪输出 */
        gtcrn_conv2d_t dc = {
            .weight = p->dc_weight, .bias = p->dc_bias,
            .in_channels = 16, .out_channels = 16,
            .kernel_h = 3, .kernel_w = 3,
            .stride_h = 1, .stride_w = 1,
            .padding_h = 2 * p->dilation, .padding_w = 1,  /* padding=(2*dilation, 1) */
            .dilation_h = p->dilation, .dilation_w = 1,
            .groups = 16  /* 深度可分离卷积 */
        };

        gtcrn_conv_transpose2d_forward(&dc, h1_padded, dc_out, batch, padded_time, freq, time, freq);
    } else {
        /* 编码器: Conv2d, padding=(0,1) */
        gtcrn_conv2d_t dc = {
            .weight = p->dc_weight, .bias = p->dc_bias,
            .in_channels = 16, .out_channels = 16,
            .kernel_h = 3, .kernel_w = 3,
            .stride_h = 1, .stride_w = 1,
            .padding_h = 0, .padding_w = 1,
            .dilation_h = p->dilation, .dilation_w = 1,
            .groups = 16  /* 深度可分离卷积 */
        };

        gtcrn_conv2d_forward(&dc, h1_padded, dc_out, batch, padded_time, freq, time, freq);
    }

    /* BN2 + PReLU2 */
    gtcrn_batchnorm2d_t bn2 = {
        .gamma = p->bn2_gamma, .beta = p->bn2_beta,
        .running_mean = p->bn2_mean, .running_var = p->bn2_var,
        .num_features = 16, .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn2, dc_out, batch, time, freq);

    gtcrn_prelu_t prelu2 = { .alpha = p->prelu2, .num_parameters = 1 };
    gtcrn_prelu_forward(&prelu2, dc_out, batch, 16, time * freq);

    /* 点卷积2: (B, 16, T, F) -> (B, 8, T, F) */
    gtcrn_conv2d_t pc2 = {
        .weight = p->pc2_weight, .bias = p->pc2_bias,
        .in_channels = 16, .out_channels = 8,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1
    };

    if (p->is_deconv) {
        gtcrn_conv_transpose2d_forward(&pc2, dc_out, h1_out, batch, time, freq, time, freq);
    } else {
        gtcrn_conv2d_forward(&pc2, dc_out, h1_out, batch, time, freq, time, freq);
    }

    /* BN3 (无激活函数) */
    gtcrn_batchnorm2d_t bn3 = {
        .gamma = p->bn3_gamma, .beta = p->bn3_beta,
        .running_mean = p->bn3_mean, .running_var = p->bn3_var,
        .num_features = 8, .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn3, h1_out, batch, time, freq);

    /* TRA注意力 */
    tra_forward(p->tra_gru_ih, p->tra_gru_hh, p->tra_gru_bih, p->tra_gru_bhh,
                p->tra_fc_weight, p->tra_fc_bias,
                h1_out, batch, 8, time, freq, tra_workspace);

    /* 通道混洗并输出 */
    channel_shuffle(h1_out, x2, x, batch, 8, time, freq);
}

/*
 * DPGRNN前向传播 (双路径分组RNN)
 */

/// <summary>分组GRU前向传播(处理一半通道)</summary>
/// <param name="weight_ih">输入-隐藏层权重 (3*hidden, input)</param>
/// <param name="weight_hh">隐藏-隐藏层权重 (3*hidden, hidden)</param>
/// <param name="bias_ih">输入-隐藏层偏置 (3*hidden)</param>
/// <param name="bias_hh">隐藏-隐藏层偏置 (3*hidden)</param>
/// <param name="input">输入序列 (seq_len, input_size)</param>
/// <param name="output">输出序列 (seq_len, hidden_size)</param>
/// <param name="seq_len">序列长度</param>
/// <param name="input_size">输入特征维度</param>
/// <param name="hidden_size">隐藏层维度</param>
/// <param name="reverse">是否反向处理</param>
/// <param name="workspace">临时工作空间</param>
static void grnn_half_forward(const gtcrn_float* weight_ih,
                              const gtcrn_float* weight_hh,
                              const gtcrn_float* bias_ih,
                              const gtcrn_float* bias_hh,
                              const gtcrn_float* input,
                              gtcrn_float* output,
                              int seq_len, int input_size, int hidden_size,
                              int reverse,
                              gtcrn_float* workspace) {
    gtcrn_float* h = workspace;
    gtcrn_float* gates_ih = h + hidden_size;
    gtcrn_float* gates_hh = gates_ih + 3 * hidden_size;

    /* 初始化隐藏状态为零 */
    gtcrn_vec_zero(h, hidden_size);

    for (int t_idx = 0; t_idx < seq_len; t_idx++) {
        int t = reverse ? (seq_len - 1 - t_idx) : t_idx;
        const gtcrn_float* x_t = input + t * input_size;
        gtcrn_float* y_t = output + t * hidden_size;

        /* Compute W_ih @ x + b_ih */
        for (int i = 0; i < 3 * hidden_size; i++) {
            gtcrn_float sum = bias_ih ? bias_ih[i] : 0.0f;
            for (int j = 0; j < input_size; j++) {
                sum += weight_ih[i * input_size + j] * x_t[j];
            }
            gates_ih[i] = sum;
        }

        /* Compute W_hh @ h + b_hh */
        for (int i = 0; i < 3 * hidden_size; i++) {
            gtcrn_float sum = bias_hh ? bias_hh[i] : 0.0f;
            for (int j = 0; j < hidden_size; j++) {
                sum += weight_hh[i * hidden_size + j] * h[j];
            }
            gates_hh[i] = sum;
        }

        /* GRU单元: r, z, n门控 */
        for (int i = 0; i < hidden_size; i++) {
            gtcrn_float r = gtcrn_sigmoid(gates_ih[i] + gates_hh[i]);
            gtcrn_float z = gtcrn_sigmoid(gates_ih[hidden_size + i] + gates_hh[hidden_size + i]);
            gtcrn_float n = gtcrn_tanh(gates_ih[2 * hidden_size + i] + r * gates_hh[2 * hidden_size + i]);
            h[i] = (1.0f - z) * n + z * h[i];
        }

        /* 复制隐藏状态到输出 */
        memcpy(y_t, h, hidden_size * sizeof(gtcrn_float));
    }
}

/// <summary>双向分组RNN (帧内处理)</summary>
/// <remarks>输入: (batch*time, freq, channels), 输出: (batch*time, freq, channels)。使用2个分组GRU (rnn1, rnn2), 每个处理一半通道的双向信息。</remarks>
static void intra_grnn_forward(const gtcrn_float* rnn1_ih, const gtcrn_float* rnn1_hh,
                               const gtcrn_float* rnn1_bih, const gtcrn_float* rnn1_bhh,
                               const gtcrn_float* rnn1_ih_rev, const gtcrn_float* rnn1_hh_rev,
                               const gtcrn_float* rnn1_bih_rev, const gtcrn_float* rnn1_bhh_rev,
                               const gtcrn_float* rnn2_ih, const gtcrn_float* rnn2_hh,
                               const gtcrn_float* rnn2_bih, const gtcrn_float* rnn2_bhh,
                               const gtcrn_float* rnn2_ih_rev, const gtcrn_float* rnn2_hh_rev,
                               const gtcrn_float* rnn2_bih_rev, const gtcrn_float* rnn2_bhh_rev,
                               const gtcrn_float* fc_weight, const gtcrn_float* fc_bias,
                               const gtcrn_float* ln_gamma, const gtcrn_float* ln_beta,
                               const gtcrn_float* input, gtcrn_float* output,
                               int batch_time, int freq, int channels,
                               gtcrn_float* workspace) {
    /* GRNN: 将通道分成两半,每半用双向GRU处理 */
    int half_ch = channels / 2;  /* 8 */
    int hidden_half = half_ch / 2;  /* 4 - 每个方向的GRU隐藏层大小 */

    gtcrn_float* temp_in1 = workspace;
    gtcrn_float* temp_in2 = temp_in1 + freq * half_ch;
    gtcrn_float* temp_out1_fwd = temp_in2 + freq * half_ch;
    gtcrn_float* temp_out1_bwd = temp_out1_fwd + freq * hidden_half;
    gtcrn_float* temp_out2_fwd = temp_out1_bwd + freq * hidden_half;
    gtcrn_float* temp_out2_bwd = temp_out2_fwd + freq * hidden_half;
    gtcrn_float* rnn_work = temp_out2_bwd + freq * hidden_half;
    gtcrn_float* fc_out = rnn_work + 256;  /* Enough for GRU workspace */

    for (int bt = 0; bt < batch_time; bt++) {
        const gtcrn_float* x = input + bt * freq * channels;

        /* 通道分割: 前半部分给rnn1,后半部分给rnn2 */
        for (int f = 0; f < freq; f++) {
            for (int c = 0; c < half_ch; c++) {
                temp_in1[f * half_ch + c] = x[f * channels + c];
                temp_in2[f * half_ch + c] = x[f * channels + half_ch + c];
            }
        }

        /* RNN1: 双向GRU(input=8, hidden=4) */
        grnn_half_forward(rnn1_ih, rnn1_hh, rnn1_bih, rnn1_bhh,
                          temp_in1, temp_out1_fwd,
                          freq, half_ch, hidden_half, 0, rnn_work);
        grnn_half_forward(rnn1_ih_rev, rnn1_hh_rev, rnn1_bih_rev, rnn1_bhh_rev,
                          temp_in1, temp_out1_bwd,
                          freq, half_ch, hidden_half, 1, rnn_work);

        /* RNN2: 双向GRU(input=8, hidden=4) */
        grnn_half_forward(rnn2_ih, rnn2_hh, rnn2_bih, rnn2_bhh,
                          temp_in2, temp_out2_fwd,
                          freq, half_ch, hidden_half, 0, rnn_work);
        grnn_half_forward(rnn2_ih_rev, rnn2_hh_rev, rnn2_bih_rev, rnn2_bhh_rev,
                          temp_in2, temp_out2_bwd,
                          freq, half_ch, hidden_half, 1, rnn_work);

        /* 拼接输出: [rnn1_fwd, rnn1_bwd, rnn2_fwd, rnn2_bwd] = (freq, 16) */
        gtcrn_float* concat = fc_out + freq * channels;
        for (int f = 0; f < freq; f++) {
            for (int c = 0; c < hidden_half; c++) {
                concat[f * channels + c] = temp_out1_fwd[f * hidden_half + c];
                concat[f * channels + hidden_half + c] = temp_out1_bwd[f * hidden_half + c];
                concat[f * channels + 2 * hidden_half + c] = temp_out2_fwd[f * hidden_half + c];
                concat[f * channels + 3 * hidden_half + c] = temp_out2_bwd[f * hidden_half + c];
            }
        }

        /* 全连接层: (freq, 16) -> (freq, 16) */
        for (int f = 0; f < freq; f++) {
            for (int c = 0; c < channels; c++) {
                gtcrn_float sum = fc_bias[c];
                for (int k = 0; k < channels; k++) {
                    sum += fc_weight[c * channels + k] * concat[f * channels + k];
                }
                fc_out[f * channels + c] = sum;
            }
        }

        /* 对(freq, channels)维度进行层归一化 */
        gtcrn_float mean = 0.0f, var = 0.0f;
        int total = freq * channels;
        for (int i = 0; i < total; i++) {
            mean += fc_out[i];
        }
        mean /= total;
        for (int i = 0; i < total; i++) {
            gtcrn_float diff = fc_out[i] - mean;
            var += diff * diff;
        }
        var /= total;
        gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);

        /* 应用层归一化(可学习参数)并添加残差 */
        gtcrn_float* y = output + bt * freq * channels;
        for (int f = 0; f < freq; f++) {
            for (int c = 0; c < channels; c++) {
                int idx = f * channels + c;
                gtcrn_float normalized = (fc_out[idx] - mean) * inv_std;
                y[idx] = x[idx] + normalized * ln_gamma[idx] + ln_beta[idx];
            }
        }
    }
}

/// <summary>单向分组RNN (帧间处理)</summary>
/// <remarks>对每个频率bin在时间维度上独立处理</remarks>
static void inter_grnn_forward(const gtcrn_float* rnn1_ih, const gtcrn_float* rnn1_hh,
                               const gtcrn_float* rnn1_bih, const gtcrn_float* rnn1_bhh,
                               const gtcrn_float* rnn2_ih, const gtcrn_float* rnn2_hh,
                               const gtcrn_float* rnn2_bih, const gtcrn_float* rnn2_bhh,
                               const gtcrn_float* fc_weight, const gtcrn_float* fc_bias,
                               const gtcrn_float* ln_gamma, const gtcrn_float* ln_beta,
                               const gtcrn_float* input, gtcrn_float* output,
                               int batch, int time, int freq, int channels,
                               gtcrn_float* workspace) {
    int half_ch = channels / 2;  /* 8 */
    int hidden = half_ch;  /* 8 - GRU hidden size */

    gtcrn_float* h1 = workspace;
    gtcrn_float* h2 = h1 + hidden;
    gtcrn_float* temp_in1 = h2 + hidden;
    gtcrn_float* temp_in2 = temp_in1 + time * half_ch;
    gtcrn_float* temp_out1 = temp_in2 + time * half_ch;
    gtcrn_float* temp_out2 = temp_out1 + time * hidden;
    gtcrn_float* rnn_work = temp_out2 + time * hidden;
    gtcrn_float* fc_out_all = rnn_work + 256;  /* Store all FC outputs for LN: (freq, time, channels) */

    for (int b = 0; b < batch; b++) {
        /* 处理所有频率bin并收集FC输出 */
        for (int f = 0; f < freq; f++) {
            /* 提取该频率bin的时间序列 */
            for (int t = 0; t < time; t++) {
                const gtcrn_float* x_t = input + (b * time + t) * freq * channels + f * channels;
                for (int c = 0; c < half_ch; c++) {
                    temp_in1[t * half_ch + c] = x_t[c];
                    temp_in2[t * half_ch + c] = x_t[half_ch + c];
                }
            }

            /* RNN1: 单向GRU(input=8, hidden=8) */
            grnn_half_forward(rnn1_ih, rnn1_hh, rnn1_bih, rnn1_bhh,
                              temp_in1, temp_out1,
                              time, half_ch, hidden, 0, rnn_work);

            /* RNN2: 单向GRU(input=8, hidden=8) */
            grnn_half_forward(rnn2_ih, rnn2_hh, rnn2_bih, rnn2_bhh,
                              temp_in2, temp_out2,
                              time, half_ch, hidden, 0, rnn_work);

            /* 对每个时间步应用FC层并存储用于LayerNorm */
            for (int t = 0; t < time; t++) {
                /* 拼接: [rnn1_out, rnn2_out] = (16,) */
                gtcrn_float concat[16];
                for (int c = 0; c < hidden; c++) {
                    concat[c] = temp_out1[t * hidden + c];
                    concat[hidden + c] = temp_out2[t * hidden + c];
                }

                /* 全连接层 */
                gtcrn_float* fc_out = fc_out_all + (t * freq + f) * channels;
                for (int c = 0; c < channels; c++) {
                    gtcrn_float sum = fc_bias[c];
                    for (int k = 0; k < channels; k++) {
                        sum += fc_weight[c * channels + k] * concat[k];
                    }
                    fc_out[c] = sum;
                }
            }
        }

        /* 对每个时间步的(freq, channels)维度进行层归一化,然后添加残差 */
        int ln_size = freq * channels;
        for (int t = 0; t < time; t++) {
            gtcrn_float* fc_slice = fc_out_all + t * freq * channels;

            /* Compute mean and variance */
            gtcrn_float mean = 0.0f, var = 0.0f;
            for (int i = 0; i < ln_size; i++) {
                mean += fc_slice[i];
            }
            mean /= ln_size;
            for (int i = 0; i < ln_size; i++) {
                gtcrn_float diff = fc_slice[i] - mean;
                var += diff * diff;
            }
            var /= ln_size;
            gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);

            /* 应用gamma、beta并添加残差 */
            for (int f = 0; f < freq; f++) {
                const gtcrn_float* x_t = input + (b * time + t) * freq * channels + f * channels;
                gtcrn_float* y_t = output + (b * time + t) * freq * channels + f * channels;
                for (int c = 0; c < channels; c++) {
                    int idx = f * channels + c;
                    gtcrn_float normalized = (fc_slice[idx] - mean) * inv_std;
                    y_t[c] = x_t[c] + normalized * ln_gamma[idx] + ln_beta[idx];
                }
            }
        }
    }
}

/// <summary>完整DPGRNN前向传播</summary>
static void dpgrnn_forward(const gtcrn_weights_t* w,
                           gtcrn_float* x,  /* In-place: (B, 16, T, 33) */
                           int batch, int time, int freq,
                           gtcrn_float* workspace,
                           int dpgrnn_idx) {
    int channels = 16;

    /* 根据DPGRNN索引选择权重 */
    const gtcrn_float* intra_rnn1_ih, *intra_rnn1_hh, *intra_rnn1_bih, *intra_rnn1_bhh;
    const gtcrn_float* intra_rnn1_ih_rev, *intra_rnn1_hh_rev, *intra_rnn1_bih_rev, *intra_rnn1_bhh_rev;
    const gtcrn_float* intra_rnn2_ih, *intra_rnn2_hh, *intra_rnn2_bih, *intra_rnn2_bhh;
    const gtcrn_float* intra_rnn2_ih_rev, *intra_rnn2_hh_rev, *intra_rnn2_bih_rev, *intra_rnn2_bhh_rev;
    const gtcrn_float* intra_fc_w, *intra_fc_b, *intra_ln_g, *intra_ln_b;
    const gtcrn_float* inter_rnn1_ih, *inter_rnn1_hh, *inter_rnn1_bih, *inter_rnn1_bhh;
    const gtcrn_float* inter_rnn2_ih, *inter_rnn2_hh, *inter_rnn2_bih, *inter_rnn2_bhh;
    const gtcrn_float* inter_fc_w, *inter_fc_b, *inter_ln_g, *inter_ln_b;

    if (dpgrnn_idx == 0) {
        intra_rnn1_ih = w->dp1_intra_rnn1_ih; intra_rnn1_hh = w->dp1_intra_rnn1_hh;
        intra_rnn1_bih = w->dp1_intra_rnn1_bih; intra_rnn1_bhh = w->dp1_intra_rnn1_bhh;
        intra_rnn1_ih_rev = w->dp1_intra_rnn1_ih_rev; intra_rnn1_hh_rev = w->dp1_intra_rnn1_hh_rev;
        intra_rnn1_bih_rev = w->dp1_intra_rnn1_bih_rev; intra_rnn1_bhh_rev = w->dp1_intra_rnn1_bhh_rev;
        intra_rnn2_ih = w->dp1_intra_rnn2_ih; intra_rnn2_hh = w->dp1_intra_rnn2_hh;
        intra_rnn2_bih = w->dp1_intra_rnn2_bih; intra_rnn2_bhh = w->dp1_intra_rnn2_bhh;
        intra_rnn2_ih_rev = w->dp1_intra_rnn2_ih_rev; intra_rnn2_hh_rev = w->dp1_intra_rnn2_hh_rev;
        intra_rnn2_bih_rev = w->dp1_intra_rnn2_bih_rev; intra_rnn2_bhh_rev = w->dp1_intra_rnn2_bhh_rev;
        intra_fc_w = w->dp1_intra_fc_weight; intra_fc_b = w->dp1_intra_fc_bias;
        intra_ln_g = w->dp1_intra_ln_gamma; intra_ln_b = w->dp1_intra_ln_beta;
        inter_rnn1_ih = w->dp1_inter_rnn1_ih; inter_rnn1_hh = w->dp1_inter_rnn1_hh;
        inter_rnn1_bih = w->dp1_inter_rnn1_bih; inter_rnn1_bhh = w->dp1_inter_rnn1_bhh;
        inter_rnn2_ih = w->dp1_inter_rnn2_ih; inter_rnn2_hh = w->dp1_inter_rnn2_hh;
        inter_rnn2_bih = w->dp1_inter_rnn2_bih; inter_rnn2_bhh = w->dp1_inter_rnn2_bhh;
        inter_fc_w = w->dp1_inter_fc_weight; inter_fc_b = w->dp1_inter_fc_bias;
        inter_ln_g = w->dp1_inter_ln_gamma; inter_ln_b = w->dp1_inter_ln_beta;
    } else {
        intra_rnn1_ih = w->dp2_intra_rnn1_ih; intra_rnn1_hh = w->dp2_intra_rnn1_hh;
        intra_rnn1_bih = w->dp2_intra_rnn1_bih; intra_rnn1_bhh = w->dp2_intra_rnn1_bhh;
        intra_rnn1_ih_rev = w->dp2_intra_rnn1_ih_rev; intra_rnn1_hh_rev = w->dp2_intra_rnn1_hh_rev;
        intra_rnn1_bih_rev = w->dp2_intra_rnn1_bih_rev; intra_rnn1_bhh_rev = w->dp2_intra_rnn1_bhh_rev;
        intra_rnn2_ih = w->dp2_intra_rnn2_ih; intra_rnn2_hh = w->dp2_intra_rnn2_hh;
        intra_rnn2_bih = w->dp2_intra_rnn2_bih; intra_rnn2_bhh = w->dp2_intra_rnn2_bhh;
        intra_rnn2_ih_rev = w->dp2_intra_rnn2_ih_rev; intra_rnn2_hh_rev = w->dp2_intra_rnn2_hh_rev;
        intra_rnn2_bih_rev = w->dp2_intra_rnn2_bih_rev; intra_rnn2_bhh_rev = w->dp2_intra_rnn2_bhh_rev;
        intra_fc_w = w->dp2_intra_fc_weight; intra_fc_b = w->dp2_intra_fc_bias;
        intra_ln_g = w->dp2_intra_ln_gamma; intra_ln_b = w->dp2_intra_ln_beta;
        inter_rnn1_ih = w->dp2_inter_rnn1_ih; inter_rnn1_hh = w->dp2_inter_rnn1_hh;
        inter_rnn1_bih = w->dp2_inter_rnn1_bih; inter_rnn1_bhh = w->dp2_inter_rnn1_bhh;
        inter_rnn2_ih = w->dp2_inter_rnn2_ih; inter_rnn2_hh = w->dp2_inter_rnn2_hh;
        inter_rnn2_bih = w->dp2_inter_rnn2_bih; inter_rnn2_bhh = w->dp2_inter_rnn2_bhh;
        inter_fc_w = w->dp2_inter_fc_weight; inter_fc_b = w->dp2_inter_fc_bias;
        inter_ln_g = w->dp2_inter_ln_gamma; inter_ln_b = w->dp2_inter_ln_beta;
    }

    /* 分配临时缓冲区 */
    gtcrn_float* temp = workspace;
    gtcrn_float* intra_out = temp + batch * time * freq * channels;
    gtcrn_float* work = intra_out + batch * time * freq * channels;

    /* 重塑: (B, C, T, F) -> (B*T, F, C) 用于帧内处理 */
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < time; t++) {
            for (int f = 0; f < freq; f++) {
                for (int c = 0; c < channels; c++) {
                    temp[(b * time + t) * freq * channels + f * channels + c] =
                        x[GTCRN_IDX4(b, c, t, f, channels, time, freq)];
                }
            }
        }
    }

    /* 帧内双向RNN */
    intra_grnn_forward(intra_rnn1_ih, intra_rnn1_hh, intra_rnn1_bih, intra_rnn1_bhh,
                       intra_rnn1_ih_rev, intra_rnn1_hh_rev, intra_rnn1_bih_rev, intra_rnn1_bhh_rev,
                       intra_rnn2_ih, intra_rnn2_hh, intra_rnn2_bih, intra_rnn2_bhh,
                       intra_rnn2_ih_rev, intra_rnn2_hh_rev, intra_rnn2_bih_rev, intra_rnn2_bhh_rev,
                       intra_fc_w, intra_fc_b, intra_ln_g, intra_ln_b,
                       temp, intra_out, batch * time, freq, channels, work);

    /* 帧间单向RNN */
    inter_grnn_forward(inter_rnn1_ih, inter_rnn1_hh, inter_rnn1_bih, inter_rnn1_bhh,
                       inter_rnn2_ih, inter_rnn2_hh, inter_rnn2_bih, inter_rnn2_bhh,
                       inter_fc_w, inter_fc_b, inter_ln_g, inter_ln_b,
                       intra_out, temp, batch, time, freq, channels, work);

    /* 重塑回: (B*T, F, C) -> (B, C, T, F) */
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < time; t++) {
            for (int f = 0; f < freq; f++) {
                for (int c = 0; c < channels; c++) {
                    x[GTCRN_IDX4(b, c, t, f, channels, time, freq)] =
                        temp[(b * time + t) * freq * channels + f * channels + c];
                }
            }
        }
    }
}

/*
 * ERB函数
 */

static void erb_bm(const gtcrn_weights_t* w,
                   const gtcrn_float* input,
                   gtcrn_float* output,
                   int batch, int channels, int time) {
    int erb_sub1 = GTCRN_ERB_SUBBAND_1;
    int erb_sub2 = GTCRN_ERB_SUBBAND_2;
    int freq_in = GTCRN_FREQ_BINS;
    int freq_out = GTCRN_ERB_TOTAL;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                /* 低频: 直接复制 */
                for (int f = 0; f < erb_sub1; f++) {
                    output[GTCRN_IDX4(b, c, t, f, channels, time, freq_out)] =
                        input[GTCRN_IDX4(b, c, t, f, channels, time, freq_in)];
                }
                /* 高频: ERB压缩 */
                for (int fo = 0; fo < erb_sub2; fo++) {
                    gtcrn_float sum = 0.0f;
                    for (int fi = 0; fi < freq_in - erb_sub1; fi++) {
                        sum += w->erb_fc_weight[fo * (freq_in - erb_sub1) + fi] *
                               input[GTCRN_IDX4(b, c, t, erb_sub1 + fi, channels, time, freq_in)];
                    }
                    output[GTCRN_IDX4(b, c, t, erb_sub1 + fo, channels, time, freq_out)] = sum;
                }
            }
        }
    }
}

static void erb_bs(const gtcrn_weights_t* w,
                   const gtcrn_float* input,
                   gtcrn_float* output,
                   int batch, int channels, int time) {
    int erb_sub1 = GTCRN_ERB_SUBBAND_1;
    int erb_sub2 = GTCRN_ERB_SUBBAND_2;
    int freq_in = GTCRN_ERB_TOTAL;
    int freq_out = GTCRN_FREQ_BINS;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                /* 低频: 直接复制 */
                for (int f = 0; f < erb_sub1; f++) {
                    output[GTCRN_IDX4(b, c, t, f, channels, time, freq_out)] =
                        input[GTCRN_IDX4(b, c, t, f, channels, time, freq_in)];
                }
                /* 高频: ERB扩展 */
                for (int fo = 0; fo < freq_out - erb_sub1; fo++) {
                    gtcrn_float sum = 0.0f;
                    for (int fi = 0; fi < erb_sub2; fi++) {
                        sum += w->ierb_fc_weight[fo * erb_sub2 + fi] *
                               input[GTCRN_IDX4(b, c, t, erb_sub1 + fi, channels, time, freq_in)];
                    }
                    output[GTCRN_IDX4(b, c, t, erb_sub1 + fo, channels, time, freq_out)] = sum;
                }
            }
        }
    }
}

/*
 * 完整前向传播 (导出函数)
 */

/* 使用外部工作空间的前向传播 - 避免与STFT缓冲区重叠 */
void gtcrn_forward_complete_with_workspace(gtcrn_t* model,
                                            const gtcrn_float* spec_real,
                                            const gtcrn_float* spec_imag,
                                            gtcrn_float* out_real,
                                            gtcrn_float* out_imag,
                                            int n_frames,
                                            gtcrn_float* workspace) {
    gtcrn_weights_t* w = model->weights;
    int batch = 1;
    int time = n_frames;
    int freq_in = GTCRN_FREQ_BINS;      /* 257 */
    int freq_erb = GTCRN_ERB_TOTAL;     /* 129 */
    int freq_down = GTCRN_DPGRNN_WIDTH; /* 33 */

    /* 工作空间布局 - 使用提供的工作空间指针 */
    size_t buf_size = 16 * time * freq_in;
    gtcrn_float* buf1 = workspace;
    gtcrn_float* buf2 = buf1 + buf_size;
    gtcrn_float* buf3 = buf2 + buf_size;
    gtcrn_float* scratch = buf3 + buf_size;

    /* 步骤1: 创建特征张量 (B, 3, T, 257) = [mag, real, imag] */
    for (int t = 0; t < time; t++) {
        for (int f = 0; f < freq_in; f++) {
            gtcrn_float r = spec_real[t * freq_in + f];
            gtcrn_float i = spec_imag[t * freq_in + f];
            gtcrn_float mag = sqrtf(r * r + i * i + GTCRN_EPS);
            buf1[GTCRN_IDX4(0, 0, t, f, 3, time, freq_in)] = mag;
            buf1[GTCRN_IDX4(0, 1, t, f, 3, time, freq_in)] = r;
            buf1[GTCRN_IDX4(0, 2, t, f, 3, time, freq_in)] = i;
        }
    }

    /* 步骤2: ERB压缩 (B, 3, T, 257) -> (B, 3, T, 129) */
    erb_bm(w, buf1, buf2, batch, 3, time);

    /* 步骤3: SFE (B, 3, T, 129) -> (B, 9, T, 129) */
    sfe_forward(buf2, buf1, batch, 3, time, freq_erb);
    DEBUG_PRINT_SUM("SFE", buf1, 9 * time * freq_erb);

    /* 步骤4: 编码器Conv0 (B, 9, T, 129) -> (B, 16, T, 65) */
    conv_block_params_t en_conv0 = {
        .conv_weight = w->en_conv0_weight, .conv_bias = w->en_conv0_bias,
        .bn_gamma = w->en_bn0_gamma, .bn_beta = w->en_bn0_beta,
        .bn_mean = w->en_bn0_mean, .bn_var = w->en_bn0_var,
        .prelu_alpha = w->en_prelu0,
        .in_ch = 9, .out_ch = 16, .kh = 1, .kw = 5,
        .sh = 1, .sw = 2, .ph = 0, .pw = 2, .groups = 1,
        .is_deconv = 0, .use_tanh = 0
    };
    int freq_65 = 65;
    conv_block_forward(&en_conv0, buf1, buf2, scratch, batch, time, freq_erb, time, freq_65);
    DEBUG_PRINT_SUM("EnConv0", buf2, 16 * time * freq_65);

    /* 保存编码器输出0用于跳跃连接 */
    gtcrn_float* en_out0 = buf3;
    memcpy(en_out0, buf2, 16 * time * freq_65 * sizeof(gtcrn_float));

    /* 步骤5: 编码器Conv1 (B, 16, T, 65) -> (B, 16, T, 33) */
    conv_block_params_t en_conv1 = {
        .conv_weight = w->en_conv1_weight, .conv_bias = w->en_conv1_bias,
        .bn_gamma = w->en_bn1_gamma, .bn_beta = w->en_bn1_beta,
        .bn_mean = w->en_bn1_mean, .bn_var = w->en_bn1_var,
        .prelu_alpha = w->en_prelu1,
        .in_ch = 16, .out_ch = 16, .kh = 1, .kw = 5,
        .sh = 1, .sw = 2, .ph = 0, .pw = 2, .groups = 2,
        .is_deconv = 0, .use_tanh = 0
    };
    conv_block_forward(&en_conv1, buf2, buf1, scratch, batch, time, freq_65, time, freq_down);
    DEBUG_PRINT_SUM("EnConv1", buf1, 16 * time * freq_down);

    /* 保存编码器输出1 */
    gtcrn_float* en_out1 = en_out0 + 16 * time * freq_65;
    memcpy(en_out1, buf1, 16 * time * freq_down * sizeof(gtcrn_float));

    /* 步骤6-8: 编码器GTConvBlocks (膨胀率 1, 2, 5) */
    /* GTConvBlock 2 (dilation=1) */
    gtconv_block_params_t en_gt2 = {
        .pc1_weight = w->en_gt2_pc1_weight, .pc1_bias = w->en_gt2_pc1_bias,
        .bn1_gamma = w->en_gt2_bn1_gamma, .bn1_beta = w->en_gt2_bn1_beta,
        .bn1_mean = w->en_gt2_bn1_mean, .bn1_var = w->en_gt2_bn1_var,
        .prelu1 = w->en_gt2_prelu1,
        .dc_weight = w->en_gt2_dc_weight, .dc_bias = w->en_gt2_dc_bias,
        .bn2_gamma = w->en_gt2_bn2_gamma, .bn2_beta = w->en_gt2_bn2_beta,
        .bn2_mean = w->en_gt2_bn2_mean, .bn2_var = w->en_gt2_bn2_var,
        .prelu2 = w->en_gt2_prelu2,
        .pc2_weight = w->en_gt2_pc2_weight, .pc2_bias = w->en_gt2_pc2_bias,
        .bn3_gamma = w->en_gt2_bn3_gamma, .bn3_beta = w->en_gt2_bn3_beta,
        .bn3_mean = w->en_gt2_bn3_mean, .bn3_var = w->en_gt2_bn3_var,
        .tra_gru_ih = w->en_gt2_tra_gru_ih, .tra_gru_hh = w->en_gt2_tra_gru_hh,
        .tra_gru_bih = w->en_gt2_tra_gru_bih, .tra_gru_bhh = w->en_gt2_tra_gru_bhh,
        .tra_fc_weight = w->en_gt2_tra_fc_weight, .tra_fc_bias = w->en_gt2_tra_fc_bias,
        .dilation = 1, .is_deconv = 0
    };
    gtconv_block_forward(&en_gt2, buf1, batch, time, freq_down, scratch);
    DEBUG_PRINT_SUM("EnGT2", buf1, 16 * time * freq_down);

    /* 保存编码器输出2 */
    gtcrn_float* en_out2 = en_out1 + 16 * time * freq_down;
    memcpy(en_out2, buf1, 16 * time * freq_down * sizeof(gtcrn_float));

    /* GTConvBlock 3 (dilation=2) */
    gtconv_block_params_t en_gt3 = {
        .pc1_weight = w->en_gt3_pc1_weight, .pc1_bias = w->en_gt3_pc1_bias,
        .bn1_gamma = w->en_gt3_bn1_gamma, .bn1_beta = w->en_gt3_bn1_beta,
        .bn1_mean = w->en_gt3_bn1_mean, .bn1_var = w->en_gt3_bn1_var,
        .prelu1 = w->en_gt3_prelu1,
        .dc_weight = w->en_gt3_dc_weight, .dc_bias = w->en_gt3_dc_bias,
        .bn2_gamma = w->en_gt3_bn2_gamma, .bn2_beta = w->en_gt3_bn2_beta,
        .bn2_mean = w->en_gt3_bn2_mean, .bn2_var = w->en_gt3_bn2_var,
        .prelu2 = w->en_gt3_prelu2,
        .pc2_weight = w->en_gt3_pc2_weight, .pc2_bias = w->en_gt3_pc2_bias,
        .bn3_gamma = w->en_gt3_bn3_gamma, .bn3_beta = w->en_gt3_bn3_beta,
        .bn3_mean = w->en_gt3_bn3_mean, .bn3_var = w->en_gt3_bn3_var,
        .tra_gru_ih = w->en_gt3_tra_gru_ih, .tra_gru_hh = w->en_gt3_tra_gru_hh,
        .tra_gru_bih = w->en_gt3_tra_gru_bih, .tra_gru_bhh = w->en_gt3_tra_gru_bhh,
        .tra_fc_weight = w->en_gt3_tra_fc_weight, .tra_fc_bias = w->en_gt3_tra_fc_bias,
        .dilation = 2, .is_deconv = 0
    };
    gtconv_block_forward(&en_gt3, buf1, batch, time, freq_down, scratch);
    DEBUG_PRINT_SUM("EnGT3", buf1, 16 * time * freq_down);

    /* 保存编码器输出3 */
    gtcrn_float* en_out3 = en_out2 + 16 * time * freq_down;
    memcpy(en_out3, buf1, 16 * time * freq_down * sizeof(gtcrn_float));

    /* GTConvBlock 4 (dilation=5) */
    gtconv_block_params_t en_gt4 = {
        .pc1_weight = w->en_gt4_pc1_weight, .pc1_bias = w->en_gt4_pc1_bias,
        .bn1_gamma = w->en_gt4_bn1_gamma, .bn1_beta = w->en_gt4_bn1_beta,
        .bn1_mean = w->en_gt4_bn1_mean, .bn1_var = w->en_gt4_bn1_var,
        .prelu1 = w->en_gt4_prelu1,
        .dc_weight = w->en_gt4_dc_weight, .dc_bias = w->en_gt4_dc_bias,
        .bn2_gamma = w->en_gt4_bn2_gamma, .bn2_beta = w->en_gt4_bn2_beta,
        .bn2_mean = w->en_gt4_bn2_mean, .bn2_var = w->en_gt4_bn2_var,
        .prelu2 = w->en_gt4_prelu2,
        .pc2_weight = w->en_gt4_pc2_weight, .pc2_bias = w->en_gt4_pc2_bias,
        .bn3_gamma = w->en_gt4_bn3_gamma, .bn3_beta = w->en_gt4_bn3_beta,
        .bn3_mean = w->en_gt4_bn3_mean, .bn3_var = w->en_gt4_bn3_var,
        .tra_gru_ih = w->en_gt4_tra_gru_ih, .tra_gru_hh = w->en_gt4_tra_gru_hh,
        .tra_gru_bih = w->en_gt4_tra_gru_bih, .tra_gru_bhh = w->en_gt4_tra_gru_bhh,
        .tra_fc_weight = w->en_gt4_tra_fc_weight, .tra_fc_bias = w->en_gt4_tra_fc_bias,
        .dilation = 5, .is_deconv = 0
    };
    gtconv_block_forward(&en_gt4, buf1, batch, time, freq_down, scratch);
    DEBUG_PRINT_SUM("EnGT4", buf1, 16 * time * freq_down);

    /* 保存编码器输出4 (最后一个GTConvBlock之后) */
    gtcrn_float* en_out4 = en_out3 + 16 * time * freq_down;
    memcpy(en_out4, buf1, 16 * time * freq_down * sizeof(gtcrn_float));

    /* 步骤9-10: DPGRNN 1 & 2 */
    dpgrnn_forward(w, buf1, batch, time, freq_down, scratch, 0);
    DEBUG_PRINT_SUM("DPGRNN1", buf1, 16 * time * freq_down);
    dpgrnn_forward(w, buf1, batch, time, freq_down, scratch, 1);
    DEBUG_PRINT_SUM("DPGRNN2", buf1, 16 * time * freq_down);

    /* =========== 解码器 =========== */

    /* 解码器GTConvBlock 0 (dilation=5) 与en_out4的跳跃连接 */
    /* 添加跳跃连接 */
    for (int i = 0; i < 16 * time * freq_down; i++) {
        buf1[i] += en_out4[i];
    }

    gtconv_block_params_t de_gt0 = {
        .pc1_weight = w->de_gt0_pc1_weight, .pc1_bias = w->de_gt0_pc1_bias,
        .bn1_gamma = w->de_gt0_bn1_gamma, .bn1_beta = w->de_gt0_bn1_beta,
        .bn1_mean = w->de_gt0_bn1_mean, .bn1_var = w->de_gt0_bn1_var,
        .prelu1 = w->de_gt0_prelu1,
        .dc_weight = w->de_gt0_dc_weight, .dc_bias = w->de_gt0_dc_bias,
        .bn2_gamma = w->de_gt0_bn2_gamma, .bn2_beta = w->de_gt0_bn2_beta,
        .bn2_mean = w->de_gt0_bn2_mean, .bn2_var = w->de_gt0_bn2_var,
        .prelu2 = w->de_gt0_prelu2,
        .pc2_weight = w->de_gt0_pc2_weight, .pc2_bias = w->de_gt0_pc2_bias,
        .bn3_gamma = w->de_gt0_bn3_gamma, .bn3_beta = w->de_gt0_bn3_beta,
        .bn3_mean = w->de_gt0_bn3_mean, .bn3_var = w->de_gt0_bn3_var,
        .tra_gru_ih = w->de_gt0_tra_gru_ih, .tra_gru_hh = w->de_gt0_tra_gru_hh,
        .tra_gru_bih = w->de_gt0_tra_gru_bih, .tra_gru_bhh = w->de_gt0_tra_gru_bhh,
        .tra_fc_weight = w->de_gt0_tra_fc_weight, .tra_fc_bias = w->de_gt0_tra_fc_bias,
        .dilation = 5, .is_deconv = 1
    };
    gtconv_block_forward(&de_gt0, buf1, batch, time, freq_down, scratch);

    /* 解码器GTConvBlock 1 (dilation=2) 与en_out3的跳跃连接 */
    for (int i = 0; i < 16 * time * freq_down; i++) {
        buf1[i] += en_out3[i];
    }

    gtconv_block_params_t de_gt1 = {
        .pc1_weight = w->de_gt1_pc1_weight, .pc1_bias = w->de_gt1_pc1_bias,
        .bn1_gamma = w->de_gt1_bn1_gamma, .bn1_beta = w->de_gt1_bn1_beta,
        .bn1_mean = w->de_gt1_bn1_mean, .bn1_var = w->de_gt1_bn1_var,
        .prelu1 = w->de_gt1_prelu1,
        .dc_weight = w->de_gt1_dc_weight, .dc_bias = w->de_gt1_dc_bias,
        .bn2_gamma = w->de_gt1_bn2_gamma, .bn2_beta = w->de_gt1_bn2_beta,
        .bn2_mean = w->de_gt1_bn2_mean, .bn2_var = w->de_gt1_bn2_var,
        .prelu2 = w->de_gt1_prelu2,
        .pc2_weight = w->de_gt1_pc2_weight, .pc2_bias = w->de_gt1_pc2_bias,
        .bn3_gamma = w->de_gt1_bn3_gamma, .bn3_beta = w->de_gt1_bn3_beta,
        .bn3_mean = w->de_gt1_bn3_mean, .bn3_var = w->de_gt1_bn3_var,
        .tra_gru_ih = w->de_gt1_tra_gru_ih, .tra_gru_hh = w->de_gt1_tra_gru_hh,
        .tra_gru_bih = w->de_gt1_tra_gru_bih, .tra_gru_bhh = w->de_gt1_tra_gru_bhh,
        .tra_fc_weight = w->de_gt1_tra_fc_weight, .tra_fc_bias = w->de_gt1_tra_fc_bias,
        .dilation = 2, .is_deconv = 1
    };
    gtconv_block_forward(&de_gt1, buf1, batch, time, freq_down, scratch);

    /* 解码器GTConvBlock 2 (dilation=1) 与en_out2的跳跃连接 */
    for (int i = 0; i < 16 * time * freq_down; i++) {
        buf1[i] += en_out2[i];
    }

    gtconv_block_params_t de_gt2 = {
        .pc1_weight = w->de_gt2_pc1_weight, .pc1_bias = w->de_gt2_pc1_bias,
        .bn1_gamma = w->de_gt2_bn1_gamma, .bn1_beta = w->de_gt2_bn1_beta,
        .bn1_mean = w->de_gt2_bn1_mean, .bn1_var = w->de_gt2_bn1_var,
        .prelu1 = w->de_gt2_prelu1,
        .dc_weight = w->de_gt2_dc_weight, .dc_bias = w->de_gt2_dc_bias,
        .bn2_gamma = w->de_gt2_bn2_gamma, .bn2_beta = w->de_gt2_bn2_beta,
        .bn2_mean = w->de_gt2_bn2_mean, .bn2_var = w->de_gt2_bn2_var,
        .prelu2 = w->de_gt2_prelu2,
        .pc2_weight = w->de_gt2_pc2_weight, .pc2_bias = w->de_gt2_pc2_bias,
        .bn3_gamma = w->de_gt2_bn3_gamma, .bn3_beta = w->de_gt2_bn3_beta,
        .bn3_mean = w->de_gt2_bn3_mean, .bn3_var = w->de_gt2_bn3_var,
        .tra_gru_ih = w->de_gt2_tra_gru_ih, .tra_gru_hh = w->de_gt2_tra_gru_hh,
        .tra_gru_bih = w->de_gt2_tra_gru_bih, .tra_gru_bhh = w->de_gt2_tra_gru_bhh,
        .tra_fc_weight = w->de_gt2_tra_fc_weight, .tra_fc_bias = w->de_gt2_tra_fc_bias,
        .dilation = 1, .is_deconv = 1
    };
    gtconv_block_forward(&de_gt2, buf1, batch, time, freq_down, scratch);

    /* 解码器ConvBlock 3: (16, T, 33) -> (16, T, 65) 与en_out1的跳跃连接 */
    for (int i = 0; i < 16 * time * freq_down; i++) {
        buf1[i] += en_out1[i];
    }

    conv_block_params_t de_conv3 = {
        .conv_weight = w->de_conv3_weight, .conv_bias = w->de_conv3_bias,
        .bn_gamma = w->de_bn3_gamma, .bn_beta = w->de_bn3_beta,
        .bn_mean = w->de_bn3_mean, .bn_var = w->de_bn3_var,
        .prelu_alpha = w->de_prelu3,
        .in_ch = 16, .out_ch = 16, .kh = 1, .kw = 5,
        .sh = 1, .sw = 2, .ph = 0, .pw = 2, .groups = 2,
        .is_deconv = 1, .use_tanh = 0
    };
    conv_block_forward(&de_conv3, buf1, buf2, scratch, batch, time, freq_down, time, freq_65);

    /* 解码器ConvBlock 4: (16, T, 65) -> (2, T, 129) 与en_out0的跳跃连接, Tanh激活 */
    for (int i = 0; i < 16 * time * freq_65; i++) {
        buf2[i] += en_out0[i];
    }

    conv_block_params_t de_conv4 = {
        .conv_weight = w->de_conv4_weight, .conv_bias = w->de_conv4_bias,
        .bn_gamma = w->de_bn4_gamma, .bn_beta = w->de_bn4_beta,
        .bn_mean = w->de_bn4_mean, .bn_var = w->de_bn4_var,
        .prelu_alpha = NULL,  /* 使用Tanh,不用PReLU */
        .in_ch = 16, .out_ch = 2, .kh = 1, .kw = 5,
        .sh = 1, .sw = 2, .ph = 0, .pw = 2, .groups = 1,
        .is_deconv = 1, .use_tanh = 1
    };
    conv_block_forward(&de_conv4, buf2, buf1, scratch, batch, time, freq_65, time, freq_erb);

    /* 步骤16: ERB扩展 (B, 2, T, 129) -> (B, 2, T, 257) */
    erb_bs(w, buf1, buf2, batch, 2, time);

    /* 步骤17: 应用复数比率掩码 */
    for (int t = 0; t < time; t++) {
        for (int f = 0; f < freq_in; f++) {
            gtcrn_float mr = buf2[GTCRN_IDX4(0, 0, t, f, 2, time, freq_in)];
            gtcrn_float mi = buf2[GTCRN_IDX4(0, 1, t, f, 2, time, freq_in)];
            gtcrn_float sr = spec_real[t * freq_in + f];
            gtcrn_float si = spec_imag[t * freq_in + f];
            out_real[t * freq_in + f] = sr * mr - si * mi;
            out_imag[t * freq_in + f] = si * mr + sr * mi;
        }
    }
}
