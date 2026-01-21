/// <file>gtcrn_layers.c</file>
/// <summary>GTCRN神经网络层实现</summary>
/// <author>江月希 李文轩</author>

#include "gtcrn_layers.h"
#include <string.h>
#include <stdlib.h>

// Conv2d层实现
void gtcrn_conv2d_forward(const gtcrn_conv2d_t* layer,
                          const gtcrn_float* input,
                          gtcrn_float* output,
                          int batch, int in_h, int in_w,
                          int out_h, int out_w) {
    int in_ch = layer->in_channels;
    int out_ch = layer->out_channels;
    int kH = layer->kernel_h;
    int kW = layer->kernel_w;
    int sH = layer->stride_h;
    int sW = layer->stride_w;
    int pH = layer->padding_h;
    int pW = layer->padding_w;
    int dH = layer->dilation_h;
    int dW = layer->dilation_w;
    int groups = layer->groups;

    int in_ch_per_group = in_ch / groups;
    int out_ch_per_group = out_ch / groups;

    /* 先清零输出 */
    gtcrn_vec_zero(output, batch * out_ch * out_h * out_w);

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < groups; g++) {
            for (int oc = 0; oc < out_ch_per_group; oc++) {
                int oc_abs = g * out_ch_per_group + oc;

                for (int oh = 0; oh < out_h; oh++) {
                    for (int ow = 0; ow < out_w; ow++) {
                        gtcrn_float sum = layer->bias ? layer->bias[oc_abs] : 0.0f;

                        for (int ic = 0; ic < in_ch_per_group; ic++) {
                            int ic_abs = g * in_ch_per_group + ic;

                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int ih = oh * sH - pH + kh * dH;
                                    int iw = ow * sW - pW + kw * dW;

                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        int in_idx = GTCRN_IDX4(b, ic_abs, ih, iw, in_ch, in_h, in_w);
                                        int w_idx = ((oc_abs * in_ch_per_group + ic) * kH + kh) * kW + kw;
                                        sum += input[in_idx] * layer->weight[w_idx];
                                    }
                                }
                            }
                        }

                        int out_idx = GTCRN_IDX4(b, oc_abs, oh, ow, out_ch, out_h, out_w);
                        output[out_idx] = sum;
                    }
                }
            }
        }
    }
}

void gtcrn_conv_transpose2d_forward(const gtcrn_conv2d_t* layer,
                                    const gtcrn_float* input,
                                    gtcrn_float* output,
                                    int batch, int in_h, int in_w,
                                    int out_h, int out_w) {
    int in_ch = layer->in_channels;
    int out_ch = layer->out_channels;
    int kH = layer->kernel_h;
    int kW = layer->kernel_w;
    int sH = layer->stride_h;
    int sW = layer->stride_w;
    int pH = layer->padding_h;
    int pW = layer->padding_w;
    int dH = layer->dilation_h;
    int dW = layer->dilation_w;
    int groups = layer->groups;

    int in_ch_per_group = in_ch / groups;
    int out_ch_per_group = out_ch / groups;

    /* 用偏置初始化输出 */
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_ch; oc++) {
            gtcrn_float bias_val = layer->bias ? layer->bias[oc] : 0.0f;
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int out_idx = GTCRN_IDX4(b, oc, oh, ow, out_ch, out_h, out_w);
                    output[out_idx] = bias_val;
                }
            }
        }
    }

    /* ConvTranspose2d: 从输入散射到输出 */
    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < groups; g++) {
            for (int ic = 0; ic < in_ch_per_group; ic++) {
                int ic_abs = g * in_ch_per_group + ic;

                for (int ih = 0; ih < in_h; ih++) {
                    for (int iw = 0; iw < in_w; iw++) {
                        gtcrn_float in_val = input[GTCRN_IDX4(b, ic_abs, ih, iw, in_ch, in_h, in_w)];

                        for (int oc = 0; oc < out_ch_per_group; oc++) {
                            int oc_abs = g * out_ch_per_group + oc;

                            for (int kh = 0; kh < kH; kh++) {
                                for (int kw = 0; kw < kW; kw++) {
                                    int oh = ih * sH - pH + kh * dH;
                                    int ow = iw * sW - pW + kw * dW;

                                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                                        /* 权重布局: (in_ch, out_ch/groups, kH, kW) */
                                        int w_idx = ((ic_abs * out_ch_per_group + oc) * kH + kh) * kW + kw;
                                        int out_idx = GTCRN_IDX4(b, oc_abs, oh, ow, out_ch, out_h, out_w);
                                        output[out_idx] += in_val * layer->weight[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// BatchNorm2d层实现
void gtcrn_batchnorm2d_forward(const gtcrn_batchnorm2d_t* layer,
                               gtcrn_float* x,
                               int batch, int height, int width) {
    int C = layer->num_features;
    int spatial = height * width;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < C; c++) {
            gtcrn_float mean = layer->running_mean[c];
            gtcrn_float var = layer->running_var[c];
            gtcrn_float gamma = layer->gamma[c];
            gtcrn_float beta = layer->beta[c];

            gtcrn_float inv_std = 1.0f / sqrtf(var + layer->eps);
            gtcrn_float scale = gamma * inv_std;
            gtcrn_float shift = beta - mean * scale;

            for (int s = 0; s < spatial; s++) {
                int idx = GTCRN_IDX4(b, c, s / width, s % width, C, height, width);
                x[idx] = x[idx] * scale + shift;
            }
        }
    }
}

// LayerNorm实现
void gtcrn_layernorm_forward(const gtcrn_layernorm_t* layer,
                             gtcrn_float* x,
                             int batch) {
    int n = layer->normalized_size;

    for (int b = 0; b < batch; b++) {
        gtcrn_float* xb = x + b * n;

        /* 计算均值 */
        gtcrn_float mean = gtcrn_vec_mean(xb, n);

        /* 计算方差 */
        gtcrn_float var = gtcrn_vec_var(xb, mean, n);

        /* 归一化并缩放 */
        gtcrn_float inv_std = 1.0f / sqrtf(var + layer->eps);

        for (int i = 0; i < n; i++) {
            xb[i] = (xb[i] - mean) * inv_std * layer->gamma[i] + layer->beta[i];
        }
    }
}

// GRU层实现

// 单个GRU单元前向传播
static void gtcrn_gru_cell(const gtcrn_float* weight_ih,
                           const gtcrn_float* weight_hh,
                           const gtcrn_float* bias_ih,
                           const gtcrn_float* bias_hh,
                           const gtcrn_float* x,
                           const gtcrn_float* h_prev,
                           gtcrn_float* h_next,
                           int input_size, int hidden_size,
                           gtcrn_float* workspace) {
    /* 工作空间布局: [r, z, n, rh] = 4 * hidden_size */
    gtcrn_float* gates_ih = workspace;
    gtcrn_float* gates_hh = workspace + 3 * hidden_size;

    /* 计算输入门: W_ih @ x + b_ih */
    for (int i = 0; i < 3 * hidden_size; i++) {
        gtcrn_float sum = bias_ih ? bias_ih[i] : 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += weight_ih[i * input_size + j] * x[j];
        }
        gates_ih[i] = sum;
    }

    /* 计算隐藏门: W_hh @ h + b_hh */
    for (int i = 0; i < 3 * hidden_size; i++) {
        gtcrn_float sum = bias_hh ? bias_hh[i] : 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += weight_hh[i * hidden_size + j] * h_prev[j];
        }
        gates_hh[i] = sum;
    }

    /* r = sigmoid(gates_ih[0:H] + gates_hh[0:H]) - 重置门 */
    /* z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H]) - 更新门 */
    /* n = tanh(gates_ih[2H:3H] + r * gates_hh[2H:3H]) - 新门 */
    /* h = (1 - z) * n + z * h_prev - 隐藏状态更新 */

    for (int i = 0; i < hidden_size; i++) {
        gtcrn_float r = gtcrn_sigmoid(gates_ih[i] + gates_hh[i]);
        gtcrn_float z = gtcrn_sigmoid(gates_ih[hidden_size + i] + gates_hh[hidden_size + i]);
        gtcrn_float n = gtcrn_tanh(gates_ih[2 * hidden_size + i] + r * gates_hh[2 * hidden_size + i]);
        h_next[i] = (1.0f - z) * n + z * h_prev[i];
    }
}

void gtcrn_gru_forward(const gtcrn_gru_t* layer,
                       const gtcrn_float* input,
                       const gtcrn_float* h_0,
                       gtcrn_float* output,
                       gtcrn_float* h_n,
                       int batch, int seq_len,
                       gtcrn_float* workspace) {
    int input_size = layer->input_size;
    int hidden_size = layer->hidden_size;

    for (int b = 0; b < batch; b++) {
        /* 初始化隐藏状态 */
        const gtcrn_float* h_prev;
        if (h_0) {
            h_prev = h_0 + b * hidden_size;
        } else {
            /* 使用零值 - 工作空间作为临时零缓冲区 */
            gtcrn_vec_zero(workspace + 6 * hidden_size, hidden_size);
            h_prev = workspace + 6 * hidden_size;
        }

        for (int t = 0; t < seq_len; t++) {
            const gtcrn_float* x_t = input + (b * seq_len + t) * input_size;
            gtcrn_float* h_t = output + (b * seq_len + t) * hidden_size;

            gtcrn_gru_cell(layer->weight_ih, layer->weight_hh,
                          layer->bias_ih, layer->bias_hh,
                          x_t, h_prev, h_t,
                          input_size, hidden_size, workspace);

            h_prev = h_t;
        }

        /* 复制最终隐藏状态 */
        if (h_n) {
            gtcrn_vec_copy(h_prev, h_n + b * hidden_size, hidden_size);
        }
    }
}

void gtcrn_gru_bidirectional_forward(const gtcrn_gru_t* layer,
                                     const gtcrn_float* input,
                                     const gtcrn_float* h_0,
                                     gtcrn_float* output,
                                     gtcrn_float* h_n,
                                     int batch, int seq_len,
                                     gtcrn_float* workspace) {
    int input_size = layer->input_size;
    int hidden_size = layer->hidden_size;

    /* 前向方向 */
    for (int b = 0; b < batch; b++) {
        const gtcrn_float* h_prev;
        if (h_0) {
            h_prev = h_0 + b * hidden_size;
        } else {
            gtcrn_vec_zero(workspace + 6 * hidden_size, hidden_size);
            h_prev = workspace + 6 * hidden_size;
        }

        for (int t = 0; t < seq_len; t++) {
            const gtcrn_float* x_t = input + (b * seq_len + t) * input_size;
            /* 输出布局: (batch, seq, 2*hidden) - 前向结果在前半部分 */
            gtcrn_float* h_t = output + (b * seq_len + t) * 2 * hidden_size;

            gtcrn_gru_cell(layer->weight_ih, layer->weight_hh,
                          layer->bias_ih, layer->bias_hh,
                          x_t, h_prev, h_t,
                          input_size, hidden_size, workspace);

            h_prev = h_t;
        }

        if (h_n) {
            gtcrn_vec_copy(h_prev, h_n + b * hidden_size, hidden_size);
        }
    }

    /* 后向方向 */
    for (int b = 0; b < batch; b++) {
        const gtcrn_float* h_prev;
        if (h_0) {
            h_prev = h_0 + (batch + b) * hidden_size;
        } else {
            gtcrn_vec_zero(workspace + 6 * hidden_size, hidden_size);
            h_prev = workspace + 6 * hidden_size;
        }

        for (int t = seq_len - 1; t >= 0; t--) {
            const gtcrn_float* x_t = input + (b * seq_len + t) * input_size;
            /* 输出布局: 后向结果在后半部分 */
            gtcrn_float* h_t = output + (b * seq_len + t) * 2 * hidden_size + hidden_size;

            gtcrn_gru_cell(layer->weight_ih_reverse, layer->weight_hh_reverse,
                          layer->bias_ih_reverse, layer->bias_hh_reverse,
                          x_t, h_prev, h_t,
                          input_size, hidden_size, workspace);

            h_prev = h_t;
        }

        if (h_n) {
            gtcrn_vec_copy(h_prev, h_n + (batch + b) * hidden_size, hidden_size);
        }
    }
}

// PReLU实现
void gtcrn_prelu_forward(const gtcrn_prelu_t* layer,
                         gtcrn_float* x,
                         int batch, int channels, int spatial) {
    if (layer->num_parameters == 1) {
        /* 共享alpha */
        gtcrn_float alpha = layer->alpha[0];
        int total = batch * channels * spatial;
        for (int i = 0; i < total; i++) {
            x[i] = x[i] > 0 ? x[i] : alpha * x[i];
        }
    } else {
        /* 每通道alpha */
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                gtcrn_float alpha = layer->alpha[c];
                for (int s = 0; s < spatial; s++) {
                    int idx = (b * channels + c) * spatial + s;
                    x[idx] = x[idx] > 0 ? x[idx] : alpha * x[idx];
                }
            }
        }
    }
}

// 线性层实现
void gtcrn_linear_forward(const gtcrn_linear_t* layer,
                          const gtcrn_float* input,
                          gtcrn_float* output,
                          int batch) {
    gtcrn_linear(layer->weight, layer->bias,
                 input, output,
                 batch, layer->in_features, layer->out_features);
}
