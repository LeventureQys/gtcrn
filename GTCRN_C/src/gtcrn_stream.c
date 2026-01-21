/// <file>gtcrn_stream.c</file>
/// <summary>GTCRN流式推理实现</summary>
/// <author>江月希 李文轩</author>
/// <remarks>逐帧流式推理,显式状态管理</remarks>

#include "gtcrn_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef GTCRN_EPS
#define GTCRN_EPS 1e-12f
#endif

/* 调试标志 - 设为1启用调试输出 */
#ifndef GTCRN_STREAM_DEBUG
#define GTCRN_STREAM_DEBUG 0
#endif

/* 全局帧计数器用于调试 */
static int g_stream_frame_count = 0;

/* 调试辅助函数 */
#if GTCRN_STREAM_DEBUG
static double debug_sum(const gtcrn_float* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}
#define STREAM_DEBUG_PRINT(name, data, size) \
    if (g_stream_frame_count == 6) printf("  [C Stream] %s: sum=%.6f\n", name, debug_sum(data, size))
#else
#define STREAM_DEBUG_PRINT(name, data, size)
#endif

// 流式辅助函数

/// <summary>单帧流式Conv2d(带缓存)</summary>
/// <remarks>缓存布局: (channels, cache_t, freq) 其中cache_t = (kernel_t - 1) * dilation_t + 1。处理后将缓存左移1帧。</remarks>
static void conv2d_stream_frame(const gtcrn_float* weight,
                                const gtcrn_float* bias,
                                gtcrn_float* cache,
                                const gtcrn_float* input,
                                gtcrn_float* output,
                                int in_ch, int out_ch,
                                int kernel_t, int kernel_f,
                                int stride_f, int pad_f,
                                int dilation_t,
                                int cache_t, int freq_in, int freq_out,
                                int groups) {
    int in_ch_per_group = in_ch / groups;
    int out_ch_per_group = out_ch / groups;

    /* 更新缓存: 左移1帧,追加新帧 */
    /* 如果cache为NULL且kernel_t=1,则不需要缓存 */
    const gtcrn_float* cache_ptr = input;  /* 默认: 直接使用输入 */
    if (cache != NULL && kernel_t > 1) {
        /* 缓存布局: (in_ch, cache_t, freq_in) - 匹配Python的(C,T,F)布局 */
        /* 对每个通道，左移时间维度 */
        for (int c = 0; c < in_ch; c++) {
            /* 左移: 将帧[1:cache_t]移动到[0:cache_t-1] */
            memmove(cache + c * cache_t * freq_in,
                    cache + c * cache_t * freq_in + freq_in,
                    (cache_t - 1) * freq_in * sizeof(gtcrn_float));
            /* 追加新帧到最后 */
            memcpy(cache + c * cache_t * freq_in + (cache_t - 1) * freq_in,
                   input + c * freq_in,
                   freq_in * sizeof(gtcrn_float));
        }
        cache_ptr = cache;
    }

    /* 计算单帧卷积输出 */
    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < out_ch_per_group; oc++) {
            int oc_abs = g * out_ch_per_group + oc;

            for (int of = 0; of < freq_out; of++) {
                gtcrn_float sum = bias ? bias[oc_abs] : 0.0f;

                #if GTCRN_STREAM_DEBUG
                // Debug for oc_abs=8, of=0 on frame 6 - add freq_in to distinguish calls
                int debug_this = (g_stream_frame_count == 6 && oc_abs == 8 && of == 0 && groups == 2 && freq_in == 65);
                if (debug_this) {
                    printf("  [conv2d_stream EnConv1] g=%d oc=%d oc_abs=%d of=%d groups=%d freq_in=%d\n", g, oc, oc_abs, of, groups, freq_in);
                    printf("  [conv2d_stream EnConv1] in_ch=%d in_ch_per_group=%d cache_ptr=%p input=%p\n", in_ch, in_ch_per_group, (void*)cache_ptr, (void*)input);
                    printf("  [conv2d_stream EnConv1] input[585]=%.6f cache_ptr[585]=%.6f\n", input[585], cache_ptr[585]);
                }
                #endif

                for (int ic = 0; ic < in_ch_per_group; ic++) {
                    int ic_abs = g * in_ch_per_group + ic;

                    for (int kt = 0; kt < kernel_t; kt++) {
                        /* 对于kernel_t=1,始终使用当前帧(t_idx=0) */
                        int t_idx = 0;
                        if (kernel_t > 1) {
                            /* 缓存中t_idx用于在t=cache_t-1处的输出 */
                            t_idx = cache_t - 1 - (kernel_t - 1 - kt) * dilation_t;
                            if (t_idx < 0 || t_idx >= cache_t) continue;
                        }

                        for (int kf = 0; kf < kernel_f; kf++) {
                            int if_idx = of * stride_f - pad_f + kf;
                            if (if_idx < 0 || if_idx >= freq_in) continue;

                            /* cache layout: (in_ch, cache_t, freq_in) - 匹配Python的(C,T,F) */
                            int cache_idx;
                            if (cache != NULL && kernel_t > 1) {
                                cache_idx = (ic_abs * cache_t + t_idx) * freq_in + if_idx;
                            } else {
                                cache_idx = ic_abs * freq_in + if_idx;
                            }
                            int w_idx = ((oc_abs * in_ch_per_group + ic) * kernel_t + kt) * kernel_f + kf;

                            #if GTCRN_STREAM_DEBUG
                            if (debug_this && kf == 2 && ic < 3) {
                                printf("    ic=%d ic_abs=%d kf=%d cache_idx=%d w_idx=%d inp=%.4f wt=%.4f\n",
                                       ic, ic_abs, kf, cache_idx, w_idx, cache_ptr[cache_idx], weight[w_idx]);
                            }
                            #endif

                            sum += cache_ptr[cache_idx] * weight[w_idx];
                        }
                    }
                }

                output[oc_abs * freq_out + of] = sum;

                #if GTCRN_STREAM_DEBUG
                if (debug_this) {
                    printf("  [conv2d_stream EnConv1] Final sum=%.6f\n", sum);
                }
                #endif
            }
        }
    }
}

/// <summary>单帧流式ConvTranspose2d</summary>
/// <remarks>
/// 实现ConvTranspose2d的流式版本:
/// 1. 将当前帧追加到时间缓存
/// 2. 在频率维度做upsampling (插入零)
/// 3. 在频率维度做非对称padding
/// 4. 执行2D卷积 (时间+频率维度)
/// 5. 取最后一个时间步作为输出
/// </remarks>
static void conv_transpose2d_stream_frame(const gtcrn_float* weight,
                                          const gtcrn_float* bias,
                                          gtcrn_float* cache,
                                          const gtcrn_float* input,
                                          gtcrn_float* output,
                                          int in_ch, int out_ch,
                                          int kernel_t, int kernel_f,
                                          int stride_f, int pad_f,
                                          int dilation_t,
                                          int cache_t, int freq_in, int freq_out,
                                          int groups,
                                          gtcrn_float* workspace) {
    int in_ch_per_group = in_ch / groups;
    int out_ch_per_group = out_ch / groups;

    /* 对于kernel_t=1，不需要缓存，直接处理当前帧 */
    if (kernel_t == 1) {
        /* 检查是否需要上采样 */
        if (stride_f > 1) {
            /* 步骤1: Upsampling - 在频率维度插入零 */
            int freq_upsampled = (freq_in - 1) * stride_f + 1;
            gtcrn_float* upsampled = workspace;

            for (int c = 0; c < in_ch; c++) {
                for (int f = 0; f < freq_in; f++) {
                    int uf = f * stride_f;
                    upsampled[c * freq_upsampled + uf] = input[c * freq_in + f];
                    for (int s = 1; s < stride_f && uf + s < freq_upsampled; s++) {
                        upsampled[c * freq_upsampled + uf + s] = 0.0f;
                    }
                }
            }

            /* 步骤2: 对称 Padding */
            /* 对于普通 ConvTranspose2d，使用对称 padding = pad_f */
            int pad_left = pad_f;
            int pad_right = pad_f;
            int freq_padded = freq_upsampled + pad_left + pad_right;
            gtcrn_float* padded = workspace + in_ch * freq_upsampled;

            for (int c = 0; c < in_ch; c++) {
                for (int f = 0; f < pad_left; f++) {
                    padded[c * freq_padded + f] = 0.0f;
                }
                for (int f = 0; f < freq_upsampled; f++) {
                    padded[c * freq_padded + pad_left + f] = upsampled[c * freq_upsampled + f];
                }
                for (int f = 0; f < pad_right; f++) {
                    padded[c * freq_padded + pad_left + freq_upsampled + f] = 0.0f;
                }
            }

            /* 步骤3: Conv2d (使用Conv2d权重布局) */
            gtcrn_vec_zero(output, out_ch * freq_out);

            for (int g = 0; g < groups; g++) {
                for (int oc = 0; oc < out_ch_per_group; oc++) {
                    int oc_abs = g * out_ch_per_group + oc;

                    for (int of = 0; of < freq_out; of++) {
                        gtcrn_float sum = bias ? bias[oc_abs] : 0.0f;

                        for (int ic = 0; ic < in_ch_per_group; ic++) {
                            int ic_abs = g * in_ch_per_group + ic;

                            for (int kf = 0; kf < kernel_f; kf++) {
                                int if_idx = of + kf;
                                if (if_idx >= 0 && if_idx < freq_padded) {
                                    int w_idx = (oc_abs * in_ch_per_group + ic) * kernel_f + kf;
                                    int in_idx = ic_abs * freq_padded + if_idx;
                                    sum += padded[in_idx] * weight[w_idx];
                                }
                            }
                        }

                        output[oc_abs * freq_out + of] = sum;
                    }
                }
            }
        } else {
            /* stride_f == 1: 普通1x1卷积，无需上采样 */
            /* 直接做卷积 */
            gtcrn_vec_zero(output, out_ch * freq_out);

            for (int g = 0; g < groups; g++) {
                for (int oc = 0; oc < out_ch_per_group; oc++) {
                    int oc_abs = g * out_ch_per_group + oc;

                    for (int of = 0; of < freq_out; of++) {
                        gtcrn_float sum = bias ? bias[oc_abs] : 0.0f;

                        for (int ic = 0; ic < in_ch_per_group; ic++) {
                            int ic_abs = g * in_ch_per_group + ic;

                            for (int kf = 0; kf < kernel_f; kf++) {
                                int if_idx = of - pad_f + kf;
                                if (if_idx >= 0 && if_idx < freq_in) {
                                    int w_idx = (oc_abs * in_ch_per_group + ic) * kernel_f + kf;
                                    int in_idx = ic_abs * freq_in + if_idx;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }

                        output[oc_abs * freq_out + of] = sum;
                    }
                }
            }
        }
        return;
    }

    /* 对于kernel_t > 1，需要时间维度的卷积 */
    /* Python逻辑：inp = cat([cache, x]) 然后用 Conv2d(dilation_t) 处理 */
    /* cache_t 是缓存帧数 = (kernel_t-1) * dilation_t，需要额外存储当前帧 */
    /* 但调用者传入的cache_t已经是正确的缓存大小，我们需要在缓存最后追加当前帧 */

    /* 临时缓冲区存储完整的输入序列 (包括cache + 当前帧) */
    int full_t = cache_t + 1;  /* 完整时间序列长度 */

#if GTCRN_STREAM_DEBUG
    /* Debug cache update */
    int is_degt0 = (cache_t == 10 && in_ch == 16 && groups == 16);
    if (is_degt0) {
        double input_sum = 0, cache_before_sum = 0;
        for (int i = 0; i < in_ch * freq_in; i++) input_sum += input[i];
        for (int i = 0; i < in_ch * cache_t * freq_in; i++) cache_before_sum += cache[i];
        printf("    [convT2d] kernel_t=%d cache_t=%d full_t=%d: input sum=%.2f, cache_before sum=%.2f\n",
               kernel_t, cache_t, full_t, input_sum, cache_before_sum);
        printf("    [convT2d] workspace=%p, input=%p, cache=%p\n", (void*)workspace, (void*)input, (void*)cache);
    }
#endif

    /* 步骤0: 构建完整输入序列 = [cache, current_frame] */
    /* 布局: (in_ch, full_t, freq_in) */
    gtcrn_float* full_input = workspace;
    for (int c = 0; c < in_ch; c++) {
        /* 复制缓存 */
        memcpy(full_input + c * full_t * freq_in,
               cache + c * cache_t * freq_in,
               cache_t * freq_in * sizeof(gtcrn_float));
        /* 追加当前帧 */
        memcpy(full_input + c * full_t * freq_in + cache_t * freq_in,
               input + c * freq_in,
               freq_in * sizeof(gtcrn_float));
    }

#if GTCRN_STREAM_DEBUG
    if (is_degt0) {
        double full_input_sum = 0;
        for (int i = 0; i < in_ch * full_t * freq_in; i++) full_input_sum += full_input[i];
        /* Check the last frame (should be h1) */
        double last_frame_sum = 0;
        for (int c = 0; c < in_ch; c++) {
            for (int f = 0; f < freq_in; f++) {
                last_frame_sum += full_input[c * full_t * freq_in + cache_t * freq_in + f];
            }
        }
        /* Re-check input sum after memcpy */
        double input_after = 0;
        for (int i = 0; i < in_ch * freq_in; i++) input_after += input[i];
        printf("    [convT2d] full_input sum=%.2f, last_frame sum=%.2f, input_after=%.2f\n",
               full_input_sum, last_frame_sum, input_after);
        /* Check if workspace overlaps with input */
        ptrdiff_t offset = (const gtcrn_float*)input - (const gtcrn_float*)workspace;
        printf("    [convT2d] input offset from workspace=%td (input_size=%d, full_input_size=%d)\n",
               offset, in_ch * freq_in, in_ch * full_t * freq_in);
    }
#endif

    /* 更新缓存: 取后cache_t帧作为新缓存 */
    for (int c = 0; c < in_ch; c++) {
        memcpy(cache + c * cache_t * freq_in,
               full_input + c * full_t * freq_in + freq_in,  /* 从第1帧开始 */
               cache_t * freq_in * sizeof(gtcrn_float));
    }

#if GTCRN_STREAM_DEBUG
    if (is_degt0) {
        double cache_after_sum = 0;
        for (int i = 0; i < in_ch * cache_t * freq_in; i++) cache_after_sum += cache[i];
        /* Check the source region that was copied */
        double source_sum = 0;
        for (int c = 0; c < in_ch; c++) {
            for (int t = 0; t < cache_t; t++) {
                for (int f = 0; f < freq_in; f++) {
                    source_sum += full_input[c * full_t * freq_in + freq_in + t * freq_in + f];
                }
            }
        }
        printf("    [convT2d] cache_after sum=%.2f, source region sum=%.2f\n",
               cache_after_sum, source_sum);
    }
#endif

    /* 步骤1: 在频率维度做padding */
    /* StreamConvTranspose2d (stride_f=1): pad = [(F_size-1)*F_dilation - F_pad, same] (对称) */
    /* 不需要upsampling因为stride_f=1 */
    int dilation_f = 1;
    int pad_amount = (kernel_f - 1) * dilation_f - pad_f;
    int freq_padded = freq_in + pad_amount * 2;
    gtcrn_float* padded = workspace + in_ch * full_t * freq_in;  /* (in_ch, full_t, freq_padded) */

    for (int c = 0; c < in_ch; c++) {
        for (int t = 0; t < full_t; t++) {
            /* 左侧填充 */
            for (int f = 0; f < pad_amount; f++) {
                padded[(c * full_t + t) * freq_padded + f] = 0.0f;
            }
            /* 复制输入 */
            for (int f = 0; f < freq_in; f++) {
                padded[(c * full_t + t) * freq_padded + pad_amount + f] =
                    full_input[(c * full_t + t) * freq_in + f];
            }
            /* 右侧填充 */
            for (int f = 0; f < pad_amount; f++) {
                padded[(c * full_t + t) * freq_padded + pad_amount + freq_in + f] = 0.0f;
            }
        }
    }

    /* 步骤2: 2D卷积，输出最后一个时间步 */
    /* 流式权重布局: (out_ch, in_ch/groups, kernel_t, kernel_f) - 已被convert_to_stream翻转 */
    /* 输入布局: padded (in_ch, full_t, freq_padded) */
    /* Conv2d with dilation: 对于输出位置0，访问输入位置 kt * dilation_t */
    gtcrn_vec_zero(output, out_ch * freq_out);

    for (int g = 0; g < groups; g++) {
        for (int oc = 0; oc < out_ch_per_group; oc++) {
            int oc_abs = g * out_ch_per_group + oc;

            for (int of = 0; of < freq_out; of++) {
                gtcrn_float sum = bias ? bias[oc_abs] : 0.0f;

                for (int ic = 0; ic < in_ch_per_group; ic++) {
                    int ic_abs = g * in_ch_per_group + ic;

                    for (int kt = 0; kt < kernel_t; kt++) {
                        /* 时间索引: Conv2d with dilation_t 访问 t_in = kt * dilation_t */
                        int t_idx = kt * dilation_t;
                        if (t_idx < 0 || t_idx >= full_t) continue;

                        for (int kf = 0; kf < kernel_f; kf++) {
                            int if_idx = of + kf;
                            if (if_idx >= 0 && if_idx < freq_padded) {
                                /* 使用Conv2d标准布局，权重已翻转无需再翻转 */
                                int w_idx = ((oc_abs * in_ch_per_group + ic) * kernel_t + kt) * kernel_f + kf;
                                int in_idx = (ic_abs * full_t + t_idx) * freq_padded + if_idx;
                                sum += padded[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }

                output[oc_abs * freq_out + of] = sum;
            }
        }
    }
}

/// <summary>流式SFE (单帧)</summary>
static void sfe_stream(const gtcrn_float* input,
                       gtcrn_float* output,
                       int channels, int freq) {
    int out_ch = channels * 3;

    for (int c = 0; c < channels; c++) {
        for (int f = 0; f < freq; f++) {
            int f_left = f - 1;
            int f_right = f + 1;

            gtcrn_float v_left = (f_left >= 0) ? input[c * freq + f_left] : 0.0f;
            gtcrn_float v_center = input[c * freq + f];
            gtcrn_float v_right = (f_right < freq) ? input[c * freq + f_right] : 0.0f;

            output[(c * 3 + 0) * freq + f] = v_left;
            output[(c * 3 + 1) * freq + f] = v_center;
            output[(c * 3 + 2) * freq + f] = v_right;
        }
    }
}

/// <summary>流式ERB压缩 (单帧)</summary>
static void erb_bm_stream(const gtcrn_weights_t* w,
                          const gtcrn_float* input,
                          gtcrn_float* output,
                          int channels) {
    int erb_sub1 = GTCRN_ERB_SUBBAND_1;
    int erb_sub2 = GTCRN_ERB_SUBBAND_2;
    int freq_in = GTCRN_FREQ_BINS;
    int freq_out = GTCRN_ERB_TOTAL;

    for (int c = 0; c < channels; c++) {
        /* 低频: 直接复制 */
        for (int f = 0; f < erb_sub1; f++) {
            output[c * freq_out + f] = input[c * freq_in + f];
        }
        /* 高频: ERB压缩 */
        for (int fo = 0; fo < erb_sub2; fo++) {
            gtcrn_float sum = 0.0f;
            for (int fi = 0; fi < freq_in - erb_sub1; fi++) {
                sum += w->erb_fc_weight[fo * (freq_in - erb_sub1) + fi] *
                       input[c * freq_in + erb_sub1 + fi];
            }
            output[c * freq_out + erb_sub1 + fo] = sum;
        }
    }
}

/// <summary>流式ERB扩展 (单帧)</summary>
static void erb_bs_stream(const gtcrn_weights_t* w,
                          const gtcrn_float* input,
                          gtcrn_float* output,
                          int channels) {
    int erb_sub1 = GTCRN_ERB_SUBBAND_1;
    int erb_sub2 = GTCRN_ERB_SUBBAND_2;
    int freq_in = GTCRN_ERB_TOTAL;
    int freq_out = GTCRN_FREQ_BINS;

    for (int c = 0; c < channels; c++) {
        /* 低频: 直接复制 */
        for (int f = 0; f < erb_sub1; f++) {
            output[c * freq_out + f] = input[c * freq_in + f];
        }
        /* 高频: ERB扩展 */
        for (int fo = 0; fo < freq_out - erb_sub1; fo++) {
            gtcrn_float sum = 0.0f;
            for (int fi = 0; fi < erb_sub2; fi++) {
                sum += w->ierb_fc_weight[fo * erb_sub2 + fi] *
                       input[c * freq_in + erb_sub1 + fi];
            }
            output[c * freq_out + erb_sub1 + fo] = sum;
        }
    }
}

/// <summary>对单帧应用BatchNorm2d (原地操作)</summary>
static void bn_stream(const gtcrn_float* gamma,
                      const gtcrn_float* beta,
                      const gtcrn_float* mean,
                      const gtcrn_float* var,
                      gtcrn_float* x,
                      int channels, int freq) {
    for (int c = 0; c < channels; c++) {
        gtcrn_float inv_std = 1.0f / sqrtf(var[c] + 1e-5f);
        gtcrn_float g = gamma[c];
        gtcrn_float b = beta[c];
        gtcrn_float m = mean[c];
        for (int f = 0; f < freq; f++) {
            x[c * freq + f] = (x[c * freq + f] - m) * inv_std * g + b;
        }
    }
}

/// <summary>对单帧应用PReLU (原地操作)</summary>
static void prelu_stream(const gtcrn_float* alpha, gtcrn_float* x, int channels, int freq) {
    gtcrn_float a = alpha[0];  /* 共享alpha */
    for (int c = 0; c < channels; c++) {
        for (int f = 0; f < freq; f++) {
            gtcrn_float val = x[c * freq + f];
            x[c * freq + f] = val > 0 ? val : a * val;
        }
    }
}

/// <summary>对单帧应用Tanh (原地操作)</summary>
static void tanh_stream(gtcrn_float* x, int channels, int freq) {
    for (int i = 0; i < channels * freq; i++) {
        x[i] = tanhf(x[i]);
    }
}

/// <summary>单帧通道混洗: 交错x1和x2</summary>
static void channel_shuffle_stream(const gtcrn_float* x1, const gtcrn_float* x2,
                                    gtcrn_float* output, int half_ch, int freq) {
    for (int c = 0; c < half_ch; c++) {
        for (int f = 0; f < freq; f++) {
            output[(c * 2) * freq + f] = x1[c * freq + f];
            output[(c * 2 + 1) * freq + f] = x2[c * freq + f];
        }
    }
}

/// <summary>流式TRA GRU步骤</summary>
static void tra_gru_step(const gtcrn_float* weight_ih,
                         const gtcrn_float* weight_hh,
                         const gtcrn_float* bias_ih,
                         const gtcrn_float* bias_hh,
                         const gtcrn_float* fc_weight,
                         const gtcrn_float* fc_bias,
                         gtcrn_float* h_cache,  /* (hidden_size,) */
                         gtcrn_float* x,        /* In-place: (channels, freq) */
                         int channels, int freq,
                         gtcrn_float* workspace) {
    int hidden = channels * 2;

    /* 计算zt = mean(x^2, dim=-1) */
    gtcrn_float* zt = workspace;
    for (int c = 0; c < channels; c++) {
        gtcrn_float sum = 0.0f;
        for (int f = 0; f < freq; f++) {
            gtcrn_float val = x[c * freq + f];
            sum += val * val;
        }
        zt[c] = sum / freq;
    }

    /* GRU单元 */
    gtcrn_float* gates_ih = workspace + channels;
    gtcrn_float* gates_hh = gates_ih + 3 * hidden;
    gtcrn_float* h_new = gates_hh + 3 * hidden;

    /* W_ih @ zt + b_ih */
    for (int i = 0; i < 3 * hidden; i++) {
        gtcrn_float sum = bias_ih ? bias_ih[i] : 0.0f;
        for (int j = 0; j < channels; j++) {
            sum += weight_ih[i * channels + j] * zt[j];
        }
        gates_ih[i] = sum;
    }

    /* W_hh @ h + b_hh */
    for (int i = 0; i < 3 * hidden; i++) {
        gtcrn_float sum = bias_hh ? bias_hh[i] : 0.0f;
        for (int j = 0; j < hidden; j++) {
            sum += weight_hh[i * hidden + j] * h_cache[j];
        }
        gates_hh[i] = sum;
    }

    /* 计算新隐藏状态 */
    for (int i = 0; i < hidden; i++) {
        gtcrn_float r = gtcrn_sigmoid(gates_ih[i] + gates_hh[i]);
        gtcrn_float z = gtcrn_sigmoid(gates_ih[hidden + i] + gates_hh[hidden + i]);
        gtcrn_float n = gtcrn_tanh(gates_ih[2 * hidden + i] + r * gates_hh[2 * hidden + i]);
        h_new[i] = (1.0f - z) * n + z * h_cache[i];
    }
    memcpy(h_cache, h_new, hidden * sizeof(gtcrn_float));

    /* 全连接层 + sigmoid */
    gtcrn_float* at = workspace;
    for (int c = 0; c < channels; c++) {
        gtcrn_float sum = fc_bias[c];
        for (int j = 0; j < hidden; j++) {
            sum += fc_weight[c * hidden + j] * h_cache[j];
        }
        at[c] = gtcrn_sigmoid(sum);
    }

    /* 应用注意力 */
    for (int c = 0; c < channels; c++) {
        for (int f = 0; f < freq; f++) {
            x[c * freq + f] *= at[c];
        }
    }
}
/// <summary>单个位置的GRU单元计算</summary>
static void gru_cell_single(const gtcrn_float* weight_ih,
                            const gtcrn_float* weight_hh,
                            const gtcrn_float* bias_ih,
                            const gtcrn_float* bias_hh,
                            const gtcrn_float* x,       /* (input_size,) */
                            gtcrn_float* h,             /* (hidden_size,) - in/out */
                            gtcrn_float* y,             /* (hidden_size,) - output, can be NULL */
                            int input_size, int hidden_size,
                            gtcrn_float* workspace) {
    gtcrn_float* gates_ih = workspace;
    gtcrn_float* gates_hh = gates_ih + 3 * hidden_size;
    gtcrn_float* h_new = gates_hh + 3 * hidden_size;

    /* W_ih @ x + b_ih */
    for (int i = 0; i < 3 * hidden_size; i++) {
        gtcrn_float sum = bias_ih ? bias_ih[i] : 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += weight_ih[i * input_size + j] * x[j];
        }
        gates_ih[i] = sum;
    }

    /* W_hh @ h + b_hh */
    for (int i = 0; i < 3 * hidden_size; i++) {
        gtcrn_float sum = bias_hh ? bias_hh[i] : 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += weight_hh[i * hidden_size + j] * h[j];
        }
        gates_hh[i] = sum;
    }

    /* GRU更新: r = 重置门, z = 更新门, n = 新门 */
    for (int i = 0; i < hidden_size; i++) {
        gtcrn_float r = gtcrn_sigmoid(gates_ih[i] + gates_hh[i]);
        gtcrn_float z = gtcrn_sigmoid(gates_ih[hidden_size + i] + gates_hh[hidden_size + i]);
        gtcrn_float n = gtcrn_tanh(gates_ih[2 * hidden_size + i] + r * gates_hh[2 * hidden_size + i]);
        h_new[i] = (1.0f - z) * n + z * h[i];
    }

    memcpy(h, h_new, hidden_size * sizeof(gtcrn_float));
    if (y) {
        memcpy(y, h_new, hidden_size * sizeof(gtcrn_float));
    }
}

/// <summary>帧内RNN的双向分组GRU (处理频率维度)</summary>
/// <remarks>Python等价: GRNN(input_size=16, hidden_size=8, bidirectional=True), rnn1: GRU(input=8, hidden=4, bidirectional=True), rnn2: GRU(input=8, hidden=4, bidirectional=True)。输入: (B*T, F, C) = (1, 33, 16), 输出: (B*T, F, hidden*2) = (1, 33, 16)。</remarks>
static void intra_grnn_bidirectional(
    /* rnn1 forward weights */
    const gtcrn_float* rnn1_ih, const gtcrn_float* rnn1_hh,
    const gtcrn_float* rnn1_bih, const gtcrn_float* rnn1_bhh,
    /* rnn1 reverse weights */
    const gtcrn_float* rnn1_ih_rev, const gtcrn_float* rnn1_hh_rev,
    const gtcrn_float* rnn1_bih_rev, const gtcrn_float* rnn1_bhh_rev,
    /* rnn2 forward weights */
    const gtcrn_float* rnn2_ih, const gtcrn_float* rnn2_hh,
    const gtcrn_float* rnn2_bih, const gtcrn_float* rnn2_bhh,
    /* rnn2 reverse weights */
    const gtcrn_float* rnn2_ih_rev, const gtcrn_float* rnn2_hh_rev,
    const gtcrn_float* rnn2_bih_rev, const gtcrn_float* rnn2_bhh_rev,
    /* I/O */
    const gtcrn_float* input,   /* (freq, input_size) = (33, 16) */
    gtcrn_float* output,        /* (freq, hidden_size) = (33, 16) */
    int freq, int input_size, int hidden_size,
    gtcrn_float* workspace) {

    /* input_size = 16, hidden_size = 16 (双向输出) */
    /* 每组处理input_size/2 = 8, 每组每方向hidden = 4 */
    int half_input = input_size / 2;    /* 8 */
    int half_hidden = hidden_size / 2;  /* 8 */
    int quarter_hidden = half_hidden / 2; /* 4 */

    /* 工作空间布局:
     * h1_fwd: (4) - rnn1前向隐藏状态
     * h1_bwd: (4) - rnn1后向隐藏状态
     * h2_fwd: (4) - rnn2前向隐藏状态
     * h2_bwd: (4) - rnn2后向隐藏状态
     * y1_fwd: (freq, 4) - rnn1前向输出
     * y1_bwd: (freq, 4) - rnn1后向输出
     * y2_fwd: (freq, 4) - rnn2前向输出
     * y2_bwd: (freq, 4) - rnn2后向输出
     * gru_work: gru_cell的临时空间
     */
    gtcrn_float* h1_fwd = workspace;
    gtcrn_float* h1_bwd = h1_fwd + quarter_hidden;
    gtcrn_float* h2_fwd = h1_bwd + quarter_hidden;
    gtcrn_float* h2_bwd = h2_fwd + quarter_hidden;
    gtcrn_float* y1_fwd = h2_bwd + quarter_hidden;
    gtcrn_float* y1_bwd = y1_fwd + freq * quarter_hidden;
    gtcrn_float* y2_fwd = y1_bwd + freq * quarter_hidden;
    gtcrn_float* y2_bwd = y2_fwd + freq * quarter_hidden;
    gtcrn_float* gru_work = y2_bwd + freq * quarter_hidden;

    /* 初始化隐藏状态为零 */
    memset(h1_fwd, 0, quarter_hidden * sizeof(gtcrn_float));
    memset(h1_bwd, 0, quarter_hidden * sizeof(gtcrn_float));
    memset(h2_fwd, 0, quarter_hidden * sizeof(gtcrn_float));
    memset(h2_bwd, 0, quarter_hidden * sizeof(gtcrn_float));

    /* rnn1和rnn2的前向传播 */
    for (int f = 0; f < freq; f++) {
        const gtcrn_float* x = input + f * input_size;
        /* rnn1处理输入的前半部分(0:8) */
        gru_cell_single(rnn1_ih, rnn1_hh, rnn1_bih, rnn1_bhh,
                        x, h1_fwd, y1_fwd + f * quarter_hidden,
                        half_input, quarter_hidden, gru_work);
        /* rnn2处理输入的后半部分(8:16) */
        gru_cell_single(rnn2_ih, rnn2_hh, rnn2_bih, rnn2_bhh,
                        x + half_input, h2_fwd, y2_fwd + f * quarter_hidden,
                        half_input, quarter_hidden, gru_work);
    }

    /* rnn1和rnn2的后向传播 */
    memset(h1_bwd, 0, quarter_hidden * sizeof(gtcrn_float));
    memset(h2_bwd, 0, quarter_hidden * sizeof(gtcrn_float));
    for (int f = freq - 1; f >= 0; f--) {
        const gtcrn_float* x = input + f * input_size;
        /* rnn1反向处理输入的前半部分 */
        gru_cell_single(rnn1_ih_rev, rnn1_hh_rev, rnn1_bih_rev, rnn1_bhh_rev,
                        x, h1_bwd, y1_bwd + f * quarter_hidden,
                        half_input, quarter_hidden, gru_work);
        /* rnn2反向处理输入的后半部分 */
        gru_cell_single(rnn2_ih_rev, rnn2_hh_rev, rnn2_bih_rev, rnn2_bhh_rev,
                        x + half_input, h2_bwd, y2_bwd + f * quarter_hidden,
                        half_input, quarter_hidden, gru_work);
    }

    /* 拼接输出: [y1_fwd, y1_bwd, y2_fwd, y2_bwd] -> (freq, 16) */
    /* 输出布局: (freq, hidden_size) 其中hidden_size = 16 */
    for (int f = 0; f < freq; f++) {
        gtcrn_float* out = output + f * hidden_size;
        /* 第一组: [y1_fwd(4), y1_bwd(4)] */
        memcpy(out, y1_fwd + f * quarter_hidden, quarter_hidden * sizeof(gtcrn_float));
        memcpy(out + quarter_hidden, y1_bwd + f * quarter_hidden, quarter_hidden * sizeof(gtcrn_float));
        /* 第二组: [y2_fwd(4), y2_bwd(4)] */
        memcpy(out + half_hidden, y2_fwd + f * quarter_hidden, quarter_hidden * sizeof(gtcrn_float));
        memcpy(out + half_hidden + quarter_hidden, y2_bwd + f * quarter_hidden, quarter_hidden * sizeof(gtcrn_float));
    }
}

/// <summary>帧间RNN的单向分组GRU (处理时间维度)</summary>
/// <remarks>Python等价: GRNN(input_size=16, hidden_size=16, bidirectional=False), rnn1: GRU(input=8, hidden=8), rnn2: GRU(input=8, hidden=8)。使用缓存处理单个时间步。</remarks>
static void inter_grnn_step(
    /* rnn1 weights */
    const gtcrn_float* rnn1_ih, const gtcrn_float* rnn1_hh,
    const gtcrn_float* rnn1_bih, const gtcrn_float* rnn1_bhh,
    /* rnn2 weights */
    const gtcrn_float* rnn2_ih, const gtcrn_float* rnn2_hh,
    const gtcrn_float* rnn2_bih, const gtcrn_float* rnn2_bhh,
    /* I/O */
    gtcrn_float* h_cache,       /* (freq, hidden_size) = (33, 16) */
    const gtcrn_float* input,   /* (freq, input_size) = (33, 16) */
    gtcrn_float* output,        /* (freq, hidden_size) = (33, 16) */
    int freq, int input_size, int hidden_size,
    gtcrn_float* workspace) {

    /* input_size = 16, hidden_size = 16 */
    /* 每组: input/2=8, hidden/2=8 */
    int half_input = input_size / 2;    /* 8 */
    int half_hidden = hidden_size / 2;  /* 8 */

    gtcrn_float* gru_work = workspace;

    /* 独立处理每个频率bin */
    for (int f = 0; f < freq; f++) {
        const gtcrn_float* x = input + f * input_size;
        gtcrn_float* h = h_cache + f * hidden_size;
        gtcrn_float* y = output + f * hidden_size;

        /* rnn1处理前半部分: x[0:8] -> y[0:8] */
        gru_cell_single(rnn1_ih, rnn1_hh, rnn1_bih, rnn1_bhh,
                        x, h, y,
                        half_input, half_hidden, gru_work);

        /* rnn2处理后半部分: x[8:16] -> y[8:16] */
        gru_cell_single(rnn2_ih, rnn2_hh, rnn2_bih, rnn2_bhh,
                        x + half_input, h + half_hidden, y + half_hidden,
                        half_input, half_hidden, gru_work);
    }
}

// 主要流式前向传播

gtcrn_status_t gtcrn_process_frame_impl(gtcrn_t* model,
                                        const gtcrn_float* spec_real,
                                        const gtcrn_float* spec_imag,
                                        gtcrn_float* out_real,
                                        gtcrn_float* out_imag) {
    if (!model || !spec_real || !spec_imag || !out_real || !out_imag) {
        return GTCRN_ERROR_NULL_POINTER;
    }
    if (!model->is_initialized) {
        return GTCRN_ERROR_NOT_INITIALIZED;
    }

    gtcrn_weights_t* w = model->weights;
    gtcrn_state_t* s = model->state;
    gtcrn_float* work = model->workspace;

    /* 增加帧计数器 */
    g_stream_frame_count++;

    int freq_in = GTCRN_FREQ_BINS;      /* 257 */
    int freq_erb = GTCRN_ERB_TOTAL;     /* 129 */
    int freq_65 = 65;
    int freq_down = GTCRN_DPGRNN_WIDTH; /* 33 */

    /* 单帧处理的工作空间布局 */
    /* 为中间缓冲区分配足够空间 */
    /* STFT/ISTFT缓冲区布局 (来自gtcrn_model.c):
     * - stft_window: 512 floats
     * - spec_real: 257 floats
     * - spec_imag: 257 floats
     * - out_spec_real: 257 floats
     * - out_spec_imag: 257 floats
     * - istft_frame: 512 floats
     * 总计: 512 + 4*257 + 512 = 2052 floats
     */
    int stft_buffer_size = GTCRN_WIN_SIZE + 4 * GTCRN_FREQ_BINS + GTCRN_WIN_SIZE;
    gtcrn_float* feat = work + stft_buffer_size;  /* 跳过STFT/ISTFT缓冲区 */
    gtcrn_float* buf1 = feat + 9 * freq_erb;         /* SFE后(9, 129) - feat需要9*129=1161 floats */
    gtcrn_float* buf2 = buf1 + 3 * freq_erb;         /* ERB后(3, 129) - buf1需要3*129=387 floats */
    gtcrn_float* buf3 = buf2 + 16 * freq_65;         /* EnConv0后(16, 65) - buf2需要16*65=1040 floats [FIXED!] */
    gtcrn_float* buf4 = buf3 + 16 * freq_down;       /* EnConv1后(16, 33) - buf3需要16*33=528 floats */
    gtcrn_float* buf5 = buf4 + 16 * freq_down;       /* GTConv用(16, 33) */
    gtcrn_float* buf6 = buf5 + 16 * freq_down;       /* DPGRNN用(16, 33) */
    gtcrn_float* buf7 = buf6 + 16 * freq_down;       /* 解码器GTConv(16, 33) */
    gtcrn_float* buf8 = buf7 + 16 * freq_down;       /* DeConv3输出(16, 65) - buf7需要16*33=528 */
    gtcrn_float* mask = buf8 + 16 * freq_65;         /* DeConv4输出(2, 129), ERB扩展后复用为(2, 257) - buf8需要16*65=1040 */
    gtcrn_float* scratch = mask + 2 * freq_in;       /* 临时空间 - mask需要max(2*129, 2*257)=514 floats */

    /* 步骤1: 创建特征张量 (3, 257) = [mag, real, imag] */
    for (int f = 0; f < freq_in; f++) {
        gtcrn_float r = spec_real[f];
        gtcrn_float i = spec_imag[f];
        gtcrn_float mag = sqrtf(r * r + i * i + GTCRN_EPS);
        feat[0 * freq_in + f] = mag;
        feat[1 * freq_in + f] = r;
        feat[2 * freq_in + f] = i;
    }

    STREAM_DEBUG_PRINT("Input feat mag", feat, freq_in);
    STREAM_DEBUG_PRINT("Input feat real", feat + freq_in, freq_in);
    STREAM_DEBUG_PRINT("Input feat imag", feat + 2 * freq_in, freq_in);

    /* 步骤2: ERB压缩 (3, 257) -> (3, 129) */
    erb_bm_stream(w, feat, buf1, 3);

    STREAM_DEBUG_PRINT("After ERB (3, 129)", buf1, 3 * freq_erb);

    /* 步骤3: SFE (3, 129) -> (9, 129) */
    sfe_stream(buf1, feat, 3, freq_erb);

    STREAM_DEBUG_PRINT("After SFE (9, 129)", feat, 9 * freq_erb);

    /* 步骤4: 编码器ConvBlock 0: Conv2d(9, 16, (1,5), stride=(1,2), padding=(0,2)) */
    /* 输入: (9, 129), 输出: (16, 65) */
    /* 前两个卷积不需要缓存(kernel_t=1) */
    conv2d_stream_frame(w->en_conv0_weight, w->en_conv0_bias,
                        NULL, feat,
                        buf2, 9, 16, 1, 5, 2, 2, 1, 1, 129, 65, 1);
    bn_stream(w->en_bn0_gamma, w->en_bn0_beta, w->en_bn0_mean, w->en_bn0_var,
              buf2, 16, 65);
    prelu_stream(w->en_prelu0, buf2, 16, 65);

    STREAM_DEBUG_PRINT("After EnConv0 (16, 65)", buf2, 16 * 65);
#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] EnConv0 first 5: %.6f %.6f %.6f %.6f %.6f\n",
               buf2[0], buf2[1], buf2[2], buf2[3], buf2[4]);
        // Print channel 0 and 8 inputs to conv1
        printf("  [C Stream] EnConv0 ch0 first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
               buf2[0], buf2[1], buf2[2], buf2[3], buf2[4],
               buf2[5], buf2[6], buf2[7], buf2[8], buf2[9]);
        printf("  [C Stream] EnConv0 ch8 first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
               buf2[8*65+0], buf2[8*65+1], buf2[8*65+2], buf2[8*65+3], buf2[8*65+4],
               buf2[8*65+5], buf2[8*65+6], buf2[8*65+7], buf2[8*65+8], buf2[8*65+9]);
        double ch0sum = 0, ch8sum = 0;
        for (int f = 0; f < 65; f++) { ch0sum += buf2[0*65+f]; ch8sum += buf2[8*65+f]; }
        printf("  [C Stream] EnConv0 ch0 sum: %.6f, ch8 sum: %.6f\n", ch0sum, ch8sum);
        // Print channels 9-15 first 3 values
        printf("  [C Stream] EnConv0 ch9 first 3: %.6f %.6f %.6f\n",
               buf2[9*65+0], buf2[9*65+1], buf2[9*65+2]);
        printf("  [C Stream] EnConv0 ch10 first 3: %.6f %.6f %.6f\n",
               buf2[10*65+0], buf2[10*65+1], buf2[10*65+2]);
        printf("  [C Stream] EnConv0 ch13 first 3: %.6f %.6f %.6f\n",
               buf2[13*65+0], buf2[13*65+1], buf2[13*65+2]);
        printf("  [C Stream] EnConv0 ch15 first 3: %.6f %.6f %.6f\n",
               buf2[15*65+0], buf2[15*65+1], buf2[15*65+2]);
    }
#endif

    /* 保存跳跃连接 */
    memcpy(s->en_out0, buf2, 16 * 65 * sizeof(gtcrn_float));

    /* 步骤5: 编码器ConvBlock 1: Conv2d(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2) */
    /* 输入: (16, 65), 输出: (16, 33) */
    /* 不需要缓存(kernel_t=1) */
#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] en_conv1 weight[0:5]: %.6f %.6f %.6f %.6f %.6f\n",
               w->en_conv1_weight[0], w->en_conv1_weight[1], w->en_conv1_weight[2],
               w->en_conv1_weight[3], w->en_conv1_weight[4]);
        printf("  [C Stream] en_conv1 weight[320:325] (oc=8): %.6f %.6f %.6f %.6f %.6f\n",
               w->en_conv1_weight[320], w->en_conv1_weight[321], w->en_conv1_weight[322],
               w->en_conv1_weight[323], w->en_conv1_weight[324]);
        printf("  [C Stream] en_conv1 bias[0:5]: %.6f %.6f %.6f %.6f %.6f\n",
               w->en_conv1_bias[0], w->en_conv1_bias[1], w->en_conv1_bias[2],
               w->en_conv1_bias[3], w->en_conv1_bias[4]);
        printf("  [C Stream] en_conv1 bias[8:13]: %.6f %.6f %.6f %.6f %.6f\n",
               w->en_conv1_bias[8], w->en_conv1_bias[9], w->en_conv1_bias[10],
               w->en_conv1_bias[11], w->en_conv1_bias[12]);
        double wsum = 0, bsum = 0;
        for (int i = 0; i < 16 * 8 * 1 * 5; i++) wsum += w->en_conv1_weight[i];
        for (int i = 0; i < 16; i++) bsum += w->en_conv1_bias[i];
        printf("  [C Stream] en_conv1 weight_sum=%.6f, bias_sum=%.6f\n", wsum, bsum);

        // Manually compute output for oc=8, of=0 BEFORE calling conv2d
        int oc_abs = 8;
        int of = 0;
        int groups = 2;
        int in_ch_per_group = 8;
        int out_ch_per_group = 8;
        int stride_f = 2;
        int pad_f = 2;
        int kernel_f = 5;
        int freq_in_local = 65;

        gtcrn_float sum = w->en_conv1_bias[oc_abs];
        printf("  [C Stream] Manual oc=8, of=0: bias=%.6f\n", sum);

        int g = oc_abs / out_ch_per_group;  // g = 8/8 = 1
        int oc_local = oc_abs % out_ch_per_group;  // oc_local = 0

        for (int ic = 0; ic < in_ch_per_group; ic++) {
            int ic_abs = g * in_ch_per_group + ic;  // Group 1: 8+ic
            double ic_contrib = 0;
            printf("    ic_local=%d (ic_abs=%d): ", ic, ic_abs);
            for (int kf = 0; kf < kernel_f; kf++) {
                int if_idx = of * stride_f - pad_f + kf;  // 0*2 - 2 + kf = kf - 2
                if (if_idx >= 0 && if_idx < freq_in_local) {
                    int cache_idx = ic_abs * freq_in_local + if_idx;
                    // Weight index - this is the key part
                    int w_idx = ((oc_abs * in_ch_per_group + ic) * 1 + 0) * kernel_f + kf;
                    gtcrn_float inp = buf2[cache_idx];
                    gtcrn_float wt = w->en_conv1_weight[w_idx];
                    ic_contrib += inp * wt;
                    printf("kf=%d w_idx=%d wt=%.4f inp=%.4f; ", kf, w_idx, wt, inp);
                }
            }
            sum += ic_contrib;
            printf("\n      contrib=%.6f, sum=%.6f\n", ic_contrib, sum);
        }
        printf("  [C Stream] Manual result (before conv): %.6f (Python: 0.197774)\n", sum);
        printf("  [C Stream] buf2 address=%p, buf2[585]=%.6f\n", (void*)buf2, buf2[585]);
        printf("  [C Stream] MARKER: About to call EnConv1 conv2d_stream_frame\n");
    }
#endif
    conv2d_stream_frame(w->en_conv1_weight, w->en_conv1_bias,
                        NULL, buf2,
                        buf3, 16, 16, 1, 5, 2, 2, 1, 1, 65, 33, 2);
#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] Actual buf3[8*33+0] = %.6f\n", buf3[8*33+0]);
    }
#endif
    STREAM_DEBUG_PRINT("EnConv1 after conv", buf3, 16 * 33);
#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] en_bn1 gamma[0:5]: %.6f %.6f %.6f %.6f %.6f\n",
               w->en_bn1_gamma[0], w->en_bn1_gamma[1], w->en_bn1_gamma[2],
               w->en_bn1_gamma[3], w->en_bn1_gamma[4]);
        printf("  [C Stream] en_bn1 beta[0:5]: %.6f %.6f %.6f %.6f %.6f\n",
               w->en_bn1_beta[0], w->en_bn1_beta[1], w->en_bn1_beta[2],
               w->en_bn1_beta[3], w->en_bn1_beta[4]);
        printf("  [C Stream] en_bn1 mean[0:5]: %.6f %.6f %.6f %.6f %.6f\n",
               w->en_bn1_mean[0], w->en_bn1_mean[1], w->en_bn1_mean[2],
               w->en_bn1_mean[3], w->en_bn1_mean[4]);
        printf("  [C Stream] en_bn1 var[0:5]: %.6f %.6f %.6f %.6f %.6f\n",
               w->en_bn1_var[0], w->en_bn1_var[1], w->en_bn1_var[2],
               w->en_bn1_var[3], w->en_bn1_var[4]);
    }
#endif
    bn_stream(w->en_bn1_gamma, w->en_bn1_beta, w->en_bn1_mean, w->en_bn1_var,
              buf3, 16, 33);
    STREAM_DEBUG_PRINT("EnConv1 after BN", buf3, 16 * 33);
#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] en_prelu1 alpha = %.6f\n", w->en_prelu1[0]);
        // Count positive/negative values
        int pos = 0, neg = 0;
        double pos_sum = 0, neg_sum = 0;
        for (int i = 0; i < 16 * 33; i++) {
            if (buf3[i] > 0) { pos++; pos_sum += buf3[i]; }
            else { neg++; neg_sum += buf3[i]; }
        }
        printf("  [C Stream] Before PReLU: pos=%d (sum=%.2f), neg=%d (sum=%.2f)\n",
               pos, pos_sum, neg, neg_sum);
    }
#endif
    prelu_stream(w->en_prelu1, buf3, 16, 33);

#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] After EnConv1 first 5: %.6f %.6f %.6f %.6f %.6f\n",
               buf3[0], buf3[1], buf3[2], buf3[3], buf3[4]);
        // Print channel 0 (first 33 values) and channel 8
        printf("  [C Stream] EnConv1 ch0 [0:10]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
               buf3[0], buf3[1], buf3[2], buf3[3], buf3[4],
               buf3[5], buf3[6], buf3[7], buf3[8], buf3[9]);
        printf("  [C Stream] EnConv1 ch8 [0:10]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
               buf3[8*33+0], buf3[8*33+1], buf3[8*33+2], buf3[8*33+3], buf3[8*33+4],
               buf3[8*33+5], buf3[8*33+6], buf3[8*33+7], buf3[8*33+8], buf3[8*33+9]);
        // Print per-channel sums
        printf("  [C Stream] Per-channel sums:\n");
        for (int c = 0; c < 16; c++) {
            double ch_sum = 0;
            for (int f = 0; f < 33; f++) ch_sum += buf3[c * 33 + f];
            printf("    ch%d: %.4f\n", c, ch_sum);
        }
    }
#endif
    STREAM_DEBUG_PRINT("After EnConv1 (16, 33)", buf3, 16 * 33);

    /* 保存跳跃连接 */
    memcpy(s->en_out1, buf3, 16 * 33 * sizeof(gtcrn_float));

    /* 步骤6-8: 编码器GTConvBlocks流式处理 */
    /* GTConvBlock 2 (dilation=1): cache_t = 3 */
    /* 公式: cache_t = (kernel_t - 1) * dilation_t + 1 = (3-1)*1 + 1 = 3 */
    {
        int half_ch = 8;
        gtcrn_float* x1 = scratch;
        gtcrn_float* x2 = x1 + half_ch * freq_down;
        gtcrn_float* x1_sfe = x2 + half_ch * freq_down;
        gtcrn_float* h1 = x1_sfe + 24 * freq_down;
        gtcrn_float* h1_out = h1 + 16 * freq_down;

        /* 通道分割 */
        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < freq_down; f++) {
                x1[c * freq_down + f] = buf3[c * freq_down + f];
                x2[c * freq_down + f] = buf3[(c + half_ch) * freq_down + f];
            }
        }

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double x1_sum = 0, x2_sum = 0;
            for (int i = 0; i < half_ch * freq_down; i++) { x1_sum += fabs(x1[i]); x2_sum += fabs(x2[i]); }
            printf("  [EnGT2] x1 abs_sum: %.6f\n", x1_sum);
            printf("  [EnGT2] x2 abs_sum: %.6f\n", x2_sum);
        }
#endif

        /* x1的SFE */
        sfe_stream(x1, x1_sfe, half_ch, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double sfe_sum = 0;
            for (int i = 0; i < 24 * freq_down; i++) { sfe_sum += fabs(x1_sfe[i]); }
            printf("  [EnGT2] After SFE abs_sum: %.6f\n", sfe_sum);
            printf("  [EnGT2] SFE first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   x1_sfe[0], x1_sfe[1], x1_sfe[2], x1_sfe[3], x1_sfe[4],
                   x1_sfe[5], x1_sfe[6], x1_sfe[7], x1_sfe[8], x1_sfe[9]);
        }
#endif

        /* 点卷积1 */
        conv2d_stream_frame(w->en_gt2_pc1_weight, w->en_gt2_pc1_bias,
                            NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double pc1_sum = 0;
            for (int i = 0; i < 16 * freq_down; i++) { pc1_sum += fabs(h1[i]); }
            printf("  [EnGT2] After PointConv1 abs_sum: %.6f\n", pc1_sum);
        }
#endif

        bn_stream(w->en_gt2_bn1_gamma, w->en_gt2_bn1_beta, w->en_gt2_bn1_mean, w->en_gt2_bn1_var,
                  h1, 16, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double bn1_sum = 0;
            for (int i = 0; i < 16 * freq_down; i++) { bn1_sum += fabs(h1[i]); }
            printf("  [EnGT2] After BN1 abs_sum: %.6f\n", bn1_sum);
        }
#endif

        prelu_stream(w->en_gt2_prelu1, h1, 16, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double prelu1_sum = 0;
            for (int i = 0; i < 16 * freq_down; i++) { prelu1_sum += fabs(h1[i]); }
            printf("  [EnGT2] After PReLU1 abs_sum: %.6f\n", prelu1_sum);
            printf("  [EnGT2] PReLU1 first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   h1[0], h1[1], h1[2], h1[3], h1[4], h1[5], h1[6], h1[7], h1[8], h1[9]);
        }
#endif

        /* 带缓存的深度卷积(dilation=1, cache_t=3) */
        /* 缓存布局: (16, 3, 33) - 独立缓存 */
        conv2d_stream_frame(w->en_gt2_dc_weight, w->en_gt2_dc_bias,
                            s->en_gt2_cache, h1,
                            buf4, 16, 16, 3, 3, 1, 1, 1, 3, freq_down, freq_down, 16);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double dc_sum = 0;
            for (int i = 0; i < 16 * freq_down; i++) { dc_sum += fabs(buf4[i]); }
            printf("  [EnGT2] After DepthConv abs_sum: %.6f\n", dc_sum);
            printf("  [EnGT2] DepthConv first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   buf4[0], buf4[1], buf4[2], buf4[3], buf4[4], buf4[5], buf4[6], buf4[7], buf4[8], buf4[9]);
        }
#endif

        bn_stream(w->en_gt2_bn2_gamma, w->en_gt2_bn2_beta, w->en_gt2_bn2_mean, w->en_gt2_bn2_var,
                  buf4, 16, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double bn2_sum = 0;
            for (int i = 0; i < 16 * freq_down; i++) { bn2_sum += fabs(buf4[i]); }
            printf("  [EnGT2] After DepthBN abs_sum: %.6f\n", bn2_sum);
        }
#endif

        prelu_stream(w->en_gt2_prelu2, buf4, 16, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double prelu2_sum = 0;
            for (int i = 0; i < 16 * freq_down; i++) { prelu2_sum += fabs(buf4[i]); }
            printf("  [EnGT2] After DepthPReLU abs_sum: %.6f\n", prelu2_sum);
            printf("  [EnGT2] DepthPReLU first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   buf4[0], buf4[1], buf4[2], buf4[3], buf4[4], buf4[5], buf4[6], buf4[7], buf4[8], buf4[9]);
        }
#endif

        /* 点卷积2 */
        conv2d_stream_frame(w->en_gt2_pc2_weight, w->en_gt2_pc2_bias,
                            NULL, buf4, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double pc2_sum = 0;
            for (int i = 0; i < 8 * freq_down; i++) { pc2_sum += fabs(h1_out[i]); }
            printf("  [EnGT2] After PointConv2 abs_sum: %.6f\n", pc2_sum);
        }
#endif

        bn_stream(w->en_gt2_bn3_gamma, w->en_gt2_bn3_beta, w->en_gt2_bn3_mean, w->en_gt2_bn3_var,
                  h1_out, 8, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double bn3_sum = 0;
            for (int i = 0; i < 8 * freq_down; i++) { bn3_sum += fabs(h1_out[i]); }
            printf("  [EnGT2] After PointBN2 abs_sum: %.6f\n", bn3_sum);
            printf("  [EnGT2] PointBN2 first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   h1_out[0], h1_out[1], h1_out[2], h1_out[3], h1_out[4],
                   h1_out[5], h1_out[6], h1_out[7], h1_out[8], h1_out[9]);
        }
#endif

        /* TRA注意力 */
        tra_gru_step(w->en_gt2_tra_gru_ih, w->en_gt2_tra_gru_hh,
                     w->en_gt2_tra_gru_bih, w->en_gt2_tra_gru_bhh,
                     w->en_gt2_tra_fc_weight, w->en_gt2_tra_fc_bias,
                     s->en_tra_h2, h1_out, 8, freq_down, scratch + 16 * freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            double tra_sum = 0;
            for (int i = 0; i < 8 * freq_down; i++) { tra_sum += fabs(h1_out[i]); }
            printf("  [EnGT2] After TRA abs_sum: %.6f\n", tra_sum);
            printf("  [EnGT2] TRA first 10: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                   h1_out[0], h1_out[1], h1_out[2], h1_out[3], h1_out[4],
                   h1_out[5], h1_out[6], h1_out[7], h1_out[8], h1_out[9]);
        }
#endif

        /* 通道混洗 */
        channel_shuffle_stream(h1_out, x2, buf4, 8, freq_down);
    }

    STREAM_DEBUG_PRINT("After EnGT2 (16, 33)", buf4, 16 * 33);

    memcpy(s->en_out2, buf4, 16 * 33 * sizeof(gtcrn_float));

    /* GTConvBlock 3 (dilation=2): cache_t = 5 */
    /* 公式: cache_t = (kernel_t - 1) * dilation_t + 1 = (3-1)*2 + 1 = 5 */
    {
        int half_ch = 8;
        gtcrn_float* x1 = scratch;
        gtcrn_float* x2 = x1 + half_ch * freq_down;
        gtcrn_float* x1_sfe = x2 + half_ch * freq_down;
        gtcrn_float* h1 = x1_sfe + 24 * freq_down;
        gtcrn_float* h1_out = h1 + 16 * freq_down;

        /* Split channels */
        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < freq_down; f++) {
                x1[c * freq_down + f] = buf4[c * freq_down + f];
                x2[c * freq_down + f] = buf4[(c + half_ch) * freq_down + f];
            }
        }

        /* SFE on x1 */
        sfe_stream(x1, x1_sfe, half_ch, freq_down);

        /* Point conv 1 */
        conv2d_stream_frame(w->en_gt3_pc1_weight, w->en_gt3_pc1_bias,
                            NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1);
        bn_stream(w->en_gt3_bn1_gamma, w->en_gt3_bn1_beta, w->en_gt3_bn1_mean, w->en_gt3_bn1_var,
                  h1, 16, freq_down);
        prelu_stream(w->en_gt3_prelu1, h1, 16, freq_down);

        /* 带缓存的深度卷积(dilation=2, cache_t=5) */
        /* 缓存布局: (16, 5, 33) - 独立缓存 */
        conv2d_stream_frame(w->en_gt3_dc_weight, w->en_gt3_dc_bias,
                            s->en_gt3_cache, h1,
                            buf4, 16, 16, 3, 3, 1, 1, 2, 5, freq_down, freq_down, 16);
        bn_stream(w->en_gt3_bn2_gamma, w->en_gt3_bn2_beta, w->en_gt3_bn2_mean, w->en_gt3_bn2_var,
                  buf4, 16, freq_down);
        prelu_stream(w->en_gt3_prelu2, buf4, 16, freq_down);

        /* Point conv 2 */
        conv2d_stream_frame(w->en_gt3_pc2_weight, w->en_gt3_pc2_bias,
                            NULL, buf4, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1);
        bn_stream(w->en_gt3_bn3_gamma, w->en_gt3_bn3_beta, w->en_gt3_bn3_mean, w->en_gt3_bn3_var,
                  h1_out, 8, freq_down);

        /* TRA */
        tra_gru_step(w->en_gt3_tra_gru_ih, w->en_gt3_tra_gru_hh,
                     w->en_gt3_tra_gru_bih, w->en_gt3_tra_gru_bhh,
                     w->en_gt3_tra_fc_weight, w->en_gt3_tra_fc_bias,
                     s->en_tra_h3, h1_out, 8, freq_down, scratch + 16 * freq_down);

        /* Channel shuffle */
        channel_shuffle_stream(h1_out, x2, buf4, 8, freq_down);
    }

    STREAM_DEBUG_PRINT("After EnGT3 (16, 33)", buf4, 16 * 33);

    memcpy(s->en_out3, buf4, 16 * 33 * sizeof(gtcrn_float));

    /* GTConvBlock 4 (dilation=5): cache_t = 11 */
    /* 公式: cache_t = (kernel_t - 1) * dilation_t + 1 = (3-1)*5 + 1 = 11 */
    {
        int half_ch = 8;
        gtcrn_float* x1 = scratch;
        gtcrn_float* x2 = x1 + half_ch * freq_down;
        gtcrn_float* x1_sfe = x2 + half_ch * freq_down;
        gtcrn_float* h1 = x1_sfe + 24 * freq_down;
        gtcrn_float* h1_out = h1 + 16 * freq_down;

        /* Split channels */
        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < freq_down; f++) {
                x1[c * freq_down + f] = buf4[c * freq_down + f];
                x2[c * freq_down + f] = buf4[(c + half_ch) * freq_down + f];
            }
        }

        /* SFE on x1 */
        sfe_stream(x1, x1_sfe, half_ch, freq_down);

        /* Point conv 1 */
        conv2d_stream_frame(w->en_gt4_pc1_weight, w->en_gt4_pc1_bias,
                            NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1);
        bn_stream(w->en_gt4_bn1_gamma, w->en_gt4_bn1_beta, w->en_gt4_bn1_mean, w->en_gt4_bn1_var,
                  h1, 16, freq_down);
        prelu_stream(w->en_gt4_prelu1, h1, 16, freq_down);

        /* 带缓存的深度卷积(dilation=5, cache_t=11) */
        /* 缓存布局: (16, 11, 33) - 独立缓存 */
        conv2d_stream_frame(w->en_gt4_dc_weight, w->en_gt4_dc_bias,
                            s->en_gt4_cache, h1,
                            buf4, 16, 16, 3, 3, 1, 1, 5, 11, freq_down, freq_down, 16);
        bn_stream(w->en_gt4_bn2_gamma, w->en_gt4_bn2_beta, w->en_gt4_bn2_mean, w->en_gt4_bn2_var,
                  buf4, 16, freq_down);
        prelu_stream(w->en_gt4_prelu2, buf4, 16, freq_down);

        /* Point conv 2 */
        conv2d_stream_frame(w->en_gt4_pc2_weight, w->en_gt4_pc2_bias,
                            NULL, buf4, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1);
        bn_stream(w->en_gt4_bn3_gamma, w->en_gt4_bn3_beta, w->en_gt4_bn3_mean, w->en_gt4_bn3_var,
                  h1_out, 8, freq_down);

        /* TRA */
        tra_gru_step(w->en_gt4_tra_gru_ih, w->en_gt4_tra_gru_hh,
                     w->en_gt4_tra_gru_bih, w->en_gt4_tra_gru_bhh,
                     w->en_gt4_tra_fc_weight, w->en_gt4_tra_fc_bias,
                     s->en_tra_h4, h1_out, 8, freq_down, scratch + 16 * freq_down);

        /* Channel shuffle */
        channel_shuffle_stream(h1_out, x2, buf5, 8, freq_down);
    }

    STREAM_DEBUG_PRINT("After EnGT4 (16, 33)", buf5, 16 * 33);

    memcpy(s->en_out4, buf5, 16 * 33 * sizeof(gtcrn_float));

    /* 步骤9-10: DPGRNN1和DPGRNN2 */
    /* DPGRNN处理: 帧内RNN(双向分组) + 帧间RNN(单向分组) */
    /* 帧内RNN用双向分组GRU处理频率维度 */
    /* 帧间RNN用单向分组GRU处理时间维度(带缓存) */

    /* DPGRNN1 */
    {
        /* 重塑: (16, 33) -> (33, 16) 用于RNN处理 */
        gtcrn_float* rnn_in = scratch;
        gtcrn_float* rnn_out = scratch + 33 * 16;
        gtcrn_float* rnn_work = scratch + 33 * 32;

        for (int f = 0; f < 33; f++) {
            for (int c = 0; c < 16; c++) {
                rnn_in[f * 16 + c] = buf5[c * 33 + f];
            }
        }

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DPGRNN1] Input (rnn_in) sum: %.6f\n", debug_sum(rnn_in, 33 * 16));
        }
#endif

        /* 帧内RNN: 双向分组GRU */
        intra_grnn_bidirectional(
            /* rnn1前向 */
            w->dp1_intra_rnn1_ih, w->dp1_intra_rnn1_hh,
            w->dp1_intra_rnn1_bih, w->dp1_intra_rnn1_bhh,
            /* rnn1反向 */
            w->dp1_intra_rnn1_ih_rev, w->dp1_intra_rnn1_hh_rev,
            w->dp1_intra_rnn1_bih_rev, w->dp1_intra_rnn1_bhh_rev,
            /* rnn2前向 */
            w->dp1_intra_rnn2_ih, w->dp1_intra_rnn2_hh,
            w->dp1_intra_rnn2_bih, w->dp1_intra_rnn2_bhh,
            /* rnn2反向 */
            w->dp1_intra_rnn2_ih_rev, w->dp1_intra_rnn2_hh_rev,
            w->dp1_intra_rnn2_bih_rev, w->dp1_intra_rnn2_bhh_rev,
            /* 输入/输出 */
            rnn_in, rnn_out, 33, 16, 16, rnn_work);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DPGRNN1] After intra_rnn (rnn_out) sum: %.6f\n", debug_sum(rnn_out, 33 * 16));
        }
#endif

        /* 全连接层 */
        gtcrn_float* fc_buf = rnn_work;  /* 复用工作空间用于FC输出(33*16) */
        for (int f = 0; f < 33; f++) {
            for (int c = 0; c < 16; c++) {
                gtcrn_float sum = w->dp1_intra_fc_bias[c];
                for (int j = 0; j < 16; j++) {
                    sum += w->dp1_intra_fc_weight[c * 16 + j] * rnn_out[f * 16 + j];
                }
                fc_buf[f * 16 + c] = sum;  /* FC output, no residual yet */
            }
        }

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DPGRNN1] After intra_fc (fc_buf) sum: %.6f\n", debug_sum(fc_buf, 33 * 16));
        }
#endif

        /* 对整个(33, 16) = 528个元素进行层归一化 */
        {
            gtcrn_float mean = 0.0f, var = 0.0f;
            for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
            mean /= (33 * 16);
            for (int i = 0; i < 33 * 16; i++) {
                gtcrn_float diff = fc_buf[i] - mean;
                var += diff * diff;
            }
            var /= (33 * 16);
            gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
            /* LayerNorm then add residual */
            for (int i = 0; i < 33 * 16; i++) {
                gtcrn_float ln_out = (fc_buf[i] - mean) * inv_std *
                    w->dp1_intra_ln_gamma[i] + w->dp1_intra_ln_beta[i];
                rnn_in[i] = rnn_in[i] + ln_out;  /* 残差连接 AFTER LayerNorm */
            }
        }

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DPGRNN1] After intra_ln+residual (rnn_in) sum: %.6f\n", debug_sum(rnn_in, 33 * 16));
        }
#endif

        /* 帧间RNN: 带缓存的单向分组GRU */
        inter_grnn_step(
            /* rnn1 */
            w->dp1_inter_rnn1_ih, w->dp1_inter_rnn1_hh,
            w->dp1_inter_rnn1_bih, w->dp1_inter_rnn1_bhh,
            /* rnn2 */
            w->dp1_inter_rnn2_ih, w->dp1_inter_rnn2_hh,
            w->dp1_inter_rnn2_bih, w->dp1_inter_rnn2_bhh,
            /* 输入/输出 */
            s->dp1_inter_h, rnn_in, rnn_out, 33, 16, 16, rnn_work);

        /* 全连接层 */
        for (int f = 0; f < 33; f++) {
            for (int c = 0; c < 16; c++) {
                gtcrn_float sum = w->dp1_inter_fc_bias[c];
                for (int j = 0; j < 16; j++) {
                    sum += w->dp1_inter_fc_weight[c * 16 + j] * rnn_out[f * 16 + j];
                }
                fc_buf[f * 16 + c] = sum;  /* FC output, no residual yet */
            }
        }

        /* LayerNorm over entire (33, 16) = 528 elements, then add residual */
        {
            gtcrn_float mean = 0.0f, var = 0.0f;
            for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
            mean /= (33 * 16);
            for (int i = 0; i < 33 * 16; i++) {
                gtcrn_float diff = fc_buf[i] - mean;
                var += diff * diff;
            }
            var /= (33 * 16);
            gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
            /* LayerNorm then add residual, then reshape to output */
            for (int f = 0; f < 33; f++) {
                for (int c = 0; c < 16; c++) {
                    gtcrn_float ln_out = (fc_buf[f * 16 + c] - mean) * inv_std *
                        w->dp1_inter_ln_gamma[f * 16 + c] + w->dp1_inter_ln_beta[f * 16 + c];
                    buf5[c * 33 + f] = rnn_in[f * 16 + c] + ln_out;  /* 残差连接 AFTER LayerNorm */
                }
            }
        }
    }

    STREAM_DEBUG_PRINT("After DPGRNN1 (16, 33)", buf5, 16 * 33);

    /* DPGRNN2 */
    {
        /* Reshape: (16, 33) -> (33, 16) for RNN processing */
        gtcrn_float* rnn_in = scratch;
        gtcrn_float* rnn_out = scratch + 33 * 16;
        gtcrn_float* rnn_work = scratch + 33 * 32;

        for (int f = 0; f < 33; f++) {
            for (int c = 0; c < 16; c++) {
                rnn_in[f * 16 + c] = buf5[c * 33 + f];
            }
        }

        /* Intra RNN: bidirectional grouped GRU */
        intra_grnn_bidirectional(
            /* rnn1 forward */
            w->dp2_intra_rnn1_ih, w->dp2_intra_rnn1_hh,
            w->dp2_intra_rnn1_bih, w->dp2_intra_rnn1_bhh,
            /* rnn1 reverse */
            w->dp2_intra_rnn1_ih_rev, w->dp2_intra_rnn1_hh_rev,
            w->dp2_intra_rnn1_bih_rev, w->dp2_intra_rnn1_bhh_rev,
            /* rnn2 forward */
            w->dp2_intra_rnn2_ih, w->dp2_intra_rnn2_hh,
            w->dp2_intra_rnn2_bih, w->dp2_intra_rnn2_bhh,
            /* rnn2 reverse */
            w->dp2_intra_rnn2_ih_rev, w->dp2_intra_rnn2_hh_rev,
            w->dp2_intra_rnn2_bih_rev, w->dp2_intra_rnn2_bhh_rev,
            /* I/O */
            rnn_in, rnn_out, 33, 16, 16, rnn_work);

        /* FC (no residual yet) */
        gtcrn_float* fc_buf = rnn_work;  /* Reuse workspace for FC output (33*16) */
        for (int f = 0; f < 33; f++) {
            for (int c = 0; c < 16; c++) {
                gtcrn_float sum = w->dp2_intra_fc_bias[c];
                for (int j = 0; j < 16; j++) {
                    sum += w->dp2_intra_fc_weight[c * 16 + j] * rnn_out[f * 16 + j];
                }
                fc_buf[f * 16 + c] = sum;  /* FC output, no residual yet */
            }
        }

        /* LayerNorm over entire (33, 16) = 528 elements, then add residual */
        {
            gtcrn_float mean = 0.0f, var = 0.0f;
            for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
            mean /= (33 * 16);
            for (int i = 0; i < 33 * 16; i++) {
                gtcrn_float diff = fc_buf[i] - mean;
                var += diff * diff;
            }
            var /= (33 * 16);
            gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
            /* LayerNorm then add residual */
            for (int i = 0; i < 33 * 16; i++) {
                gtcrn_float ln_out = (fc_buf[i] - mean) * inv_std *
                    w->dp2_intra_ln_gamma[i] + w->dp2_intra_ln_beta[i];
                rnn_in[i] = rnn_in[i] + ln_out;  /* 残差连接 AFTER LayerNorm */
            }
        }

        /* Inter RNN: unidirectional grouped GRU with cache */
        inter_grnn_step(
            /* rnn1 */
            w->dp2_inter_rnn1_ih, w->dp2_inter_rnn1_hh,
            w->dp2_inter_rnn1_bih, w->dp2_inter_rnn1_bhh,
            /* rnn2 */
            w->dp2_inter_rnn2_ih, w->dp2_inter_rnn2_hh,
            w->dp2_inter_rnn2_bih, w->dp2_inter_rnn2_bhh,
            /* I/O */
            s->dp2_inter_h, rnn_in, rnn_out, 33, 16, 16, rnn_work);

        /* FC (no residual yet) */
        for (int f = 0; f < 33; f++) {
            for (int c = 0; c < 16; c++) {
                gtcrn_float sum = w->dp2_inter_fc_bias[c];
                for (int j = 0; j < 16; j++) {
                    sum += w->dp2_inter_fc_weight[c * 16 + j] * rnn_out[f * 16 + j];
                }
                fc_buf[f * 16 + c] = sum;  /* FC output, no residual yet */
            }
        }

        /* 对整个(33, 16) = 528个元素进行层归一化, then add residual */
        {
            gtcrn_float mean = 0.0f, var = 0.0f;
            for (int i = 0; i < 33 * 16; i++) mean += fc_buf[i];
            mean /= (33 * 16);
            for (int i = 0; i < 33 * 16; i++) {
                gtcrn_float diff = fc_buf[i] - mean;
                var += diff * diff;
            }
            var /= (33 * 16);
            gtcrn_float inv_std = 1.0f / sqrtf(var + 1e-8f);
            /* LayerNorm then add residual, then reshape to output */
            for (int f = 0; f < 33; f++) {
                for (int c = 0; c < 16; c++) {
                    gtcrn_float ln_out = (fc_buf[f * 16 + c] - mean) * inv_std *
                        w->dp2_inter_ln_gamma[f * 16 + c] + w->dp2_inter_ln_beta[f * 16 + c];
                    buf6[c * 33 + f] = rnn_in[f * 16 + c] + ln_out;  /* 残差连接 AFTER LayerNorm */
                }
            }
        }
    }


    STREAM_DEBUG_PRINT("After DPGRNN2 (16, 33)", buf6, 16 * 33);

    /* 步骤11-15: 带跳跃连接的解码器 */
    /* 解码器GTConvBlock 0 (dilation=5): cache_t = 10 */
    /* 公式: cache_t = (kernel_t - 1) * dilation_t = (3-1)*5 = 10 (不含当前帧) */
    {
#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] DPGRNN2 output sum: %.6f (Python: 84.713)\n", debug_sum(buf6, 16*33));
            printf("  [DeGT0] en_out4 sum: %.6f (Python: 34.861)\n", debug_sum(s->en_out4, 16*33));
        }
#endif

        /* 添加跳跃连接 */
        for (int i = 0; i < 16 * 33; i++) {
            buf6[i] += s->en_out4[i];
        }

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After skip sum: %.6f (Python: 119.574)\n", debug_sum(buf6, 16*33));
        }
#endif

        int half_ch = 8;
        gtcrn_float* x1 = scratch;
        gtcrn_float* x2 = x1 + half_ch * freq_down;
        gtcrn_float* x1_sfe = x2 + half_ch * freq_down;
        gtcrn_float* h1 = x1_sfe + 24 * freq_down;
        gtcrn_float* h1_out = h1 + 16 * freq_down;
        gtcrn_float* conv_workspace = h1_out + 8 * freq_down;  /* DeGT0 workspace after all intermediate buffers */

        /* Split channels */
        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < freq_down; f++) {
                x1[c * freq_down + f] = buf6[c * freq_down + f];
                x2[c * freq_down + f] = buf6[(c + half_ch) * freq_down + f];
            }
        }

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] x1 sum: %.6f (Python: 89.450)\n", debug_sum(x1, 8*33));
            printf("  [DeGT0] x2 sum: %.6f (Python: 30.124)\n", debug_sum(x2, 8*33));
        }
#endif

        /* SFE on x1 */
        sfe_stream(x1, x1_sfe, half_ch, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After SFE sum: %.6f (Python: 275.544)\n", debug_sum(x1_sfe, 24*33));
        }
#endif

        /* Point conv 1 */
        conv_transpose2d_stream_frame(w->de_gt0_pc1_weight, w->de_gt0_pc1_bias,
                                       NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1, conv_workspace);
        bn_stream(w->de_gt0_bn1_gamma, w->de_gt0_bn1_beta, w->de_gt0_bn1_mean, w->de_gt0_bn1_var,
                  h1, 16, freq_down);
        prelu_stream(w->de_gt0_prelu1, h1, 16, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After PointConv1+BN+PReLU sum: %.6f (Python: 13.986)\n", debug_sum(h1, 16*33));
            printf("  [DeGT0] h1 first 10: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                   h1[0], h1[1], h1[2], h1[3], h1[4], h1[5], h1[6], h1[7], h1[8], h1[9]);
            printf("  [DeGT0] de_gt0_cache sum: %.6f (Python: 246.983)\n",
                   debug_sum(s->de_gt0_cache, 10*16*33));
        }
        /* Track cache at every frame */
        printf("  [DeGT0-F%d] BEFORE depth_conv: h1 sum=%.2f, cache sum=%.2f\n",
               g_stream_frame_count, debug_sum(h1, 16*33), debug_sum(s->de_gt0_cache, 10*16*33));
#endif

        /* 带缓存的深度卷积(dilation=5, cache_t=10) */
        /* 缓存布局: (16, 10, 33) - 独立缓存 */
        /* 公式: cache_t = (kernel_t - 1) * dilation_t = (3-1)*5 = 10 (不含当前帧) */
        conv_transpose2d_stream_frame(w->de_gt0_dc_weight, w->de_gt0_dc_bias,
                                       s->de_gt0_cache, h1,
                                       buf6, 16, 16, 3, 3, 1, 1, 5, 10, freq_down, freq_down, 16, conv_workspace);

#if GTCRN_STREAM_DEBUG
        printf("  [DeGT0-F%d] AFTER depth_conv: cache sum=%.2f\n",
               g_stream_frame_count, debug_sum(s->de_gt0_cache, 10*16*33));
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After DepthConv sum: %.6f (Python: -149.039)\n", debug_sum(buf6, 16*33));
            printf("  [DeGT0] buf6 first 10: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                   buf6[0], buf6[1], buf6[2], buf6[3], buf6[4], buf6[5], buf6[6], buf6[7], buf6[8], buf6[9]);
        }
#endif

        bn_stream(w->de_gt0_bn2_gamma, w->de_gt0_bn2_beta, w->de_gt0_bn2_mean, w->de_gt0_bn2_var,
                  buf6, 16, freq_down);
        prelu_stream(w->de_gt0_prelu2, buf6, 16, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After DepthBN+PReLU sum: %.6f (Python: 228.857)\n", debug_sum(buf6, 16*33));
        }
#endif

        /* Point conv 2 */
        conv_transpose2d_stream_frame(w->de_gt0_pc2_weight, w->de_gt0_pc2_bias,
                                       NULL, buf6, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1, conv_workspace);
        bn_stream(w->de_gt0_bn3_gamma, w->de_gt0_bn3_beta, w->de_gt0_bn3_mean, w->de_gt0_bn3_var,
                  h1_out, 8, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After PointConv2+BN sum: %.6f (Python: -40.126)\n", debug_sum(h1_out, 8*33));
        }
#endif

        /* TRA */
        tra_gru_step(w->de_gt0_tra_gru_ih, w->de_gt0_tra_gru_hh,
                     w->de_gt0_tra_gru_bih, w->de_gt0_tra_gru_bhh,
                     w->de_gt0_tra_fc_weight, w->de_gt0_tra_fc_bias,
                     s->de_tra_h0, h1_out, 8, freq_down, conv_workspace);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After TRA sum: %.6f (Python: -3.952)\n", debug_sum(h1_out, 8*33));
            printf("  [DeGT0] x2 sum (for shuffle): %.6f (Python: 30.124)\n", debug_sum(x2, 8*33));
        }
#endif

        /* Channel shuffle */
        channel_shuffle_stream(h1_out, x2, buf6, 8, freq_down);

#if GTCRN_STREAM_DEBUG
        if (g_stream_frame_count == 6) {
            printf("  [DeGT0] After Shuffle (output) sum: %.6f (Python: 26.172)\n", debug_sum(buf6, 16*33));
        }
#endif
    }

    STREAM_DEBUG_PRINT("After DeGT0 (16, 33)", buf6, 16 * 33);

    /* 解码器GTConvBlock 1 (dilation=2): cache_t = 5 */
    /* 公式: cache_t = (kernel_t - 1) * dilation_t + 1 = (3-1)*2 + 1 = 5 */
    {
        /* 添加跳跃连接 */
        for (int i = 0; i < 16 * 33; i++) {
            buf6[i] += s->en_out3[i];
        }

        int half_ch = 8;
        gtcrn_float* x1 = scratch;
        gtcrn_float* x2 = x1 + half_ch * freq_down;
        gtcrn_float* x1_sfe = x2 + half_ch * freq_down;
        gtcrn_float* h1 = x1_sfe + 24 * freq_down;
        gtcrn_float* h1_out = h1 + 16 * freq_down;
        gtcrn_float* conv_workspace = h1_out + 8 * freq_down;  /* DeGT1 workspace */

        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < freq_down; f++) {
                x1[c * freq_down + f] = buf6[c * freq_down + f];
                x2[c * freq_down + f] = buf6[(c + half_ch) * freq_down + f];
            }
        }

        sfe_stream(x1, x1_sfe, half_ch, freq_down);

        conv_transpose2d_stream_frame(w->de_gt1_pc1_weight, w->de_gt1_pc1_bias,
                                       NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1, conv_workspace);
        bn_stream(w->de_gt1_bn1_gamma, w->de_gt1_bn1_beta, w->de_gt1_bn1_mean, w->de_gt1_bn1_var,
                  h1, 16, freq_down);
        prelu_stream(w->de_gt1_prelu1, h1, 16, freq_down);

        /* 缓存布局: (16, 4, 33) - 独立缓存 */
        /* 公式: cache_t = (kernel_t - 1) * dilation_t = (3-1)*2 = 4 (不含当前帧) */
        conv_transpose2d_stream_frame(w->de_gt1_dc_weight, w->de_gt1_dc_bias,
                                       s->de_gt1_cache, h1,
                                       buf6, 16, 16, 3, 3, 1, 1, 2, 4, freq_down, freq_down, 16, conv_workspace);
        bn_stream(w->de_gt1_bn2_gamma, w->de_gt1_bn2_beta, w->de_gt1_bn2_mean, w->de_gt1_bn2_var,
                  buf6, 16, freq_down);
        prelu_stream(w->de_gt1_prelu2, buf6, 16, freq_down);

        conv_transpose2d_stream_frame(w->de_gt1_pc2_weight, w->de_gt1_pc2_bias,
                                       NULL, buf6, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1, conv_workspace);
        bn_stream(w->de_gt1_bn3_gamma, w->de_gt1_bn3_beta, w->de_gt1_bn3_mean, w->de_gt1_bn3_var,
                  h1_out, 8, freq_down);

        tra_gru_step(w->de_gt1_tra_gru_ih, w->de_gt1_tra_gru_hh,
                     w->de_gt1_tra_gru_bih, w->de_gt1_tra_gru_bhh,
                     w->de_gt1_tra_fc_weight, w->de_gt1_tra_fc_bias,
                     s->de_tra_h1, h1_out, 8, freq_down, conv_workspace);

        channel_shuffle_stream(h1_out, x2, buf6, 8, freq_down);
    }

    STREAM_DEBUG_PRINT("After DeGT1 (16, 33)", buf6, 16 * 33);

    /* 解码器GTConvBlock 2 (dilation=1): cache_t = 3 */
    /* 公式: cache_t = (kernel_t - 1) * dilation_t + 1 = (3-1)*1 + 1 = 3 */
    {
        /* 添加跳跃连接 */
        for (int i = 0; i < 16 * 33; i++) {
            buf6[i] += s->en_out2[i];
        }

        int half_ch = 8;
        gtcrn_float* x1 = scratch;
        gtcrn_float* x2 = x1 + half_ch * freq_down;
        gtcrn_float* x1_sfe = x2 + half_ch * freq_down;
        gtcrn_float* h1 = x1_sfe + 24 * freq_down;
        gtcrn_float* h1_out = h1 + 16 * freq_down;
        gtcrn_float* conv_workspace = h1_out + 8 * freq_down;  /* DeGT2 workspace */

        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < freq_down; f++) {
                x1[c * freq_down + f] = buf6[c * freq_down + f];
                x2[c * freq_down + f] = buf6[(c + half_ch) * freq_down + f];
            }
        }

        sfe_stream(x1, x1_sfe, half_ch, freq_down);

        conv_transpose2d_stream_frame(w->de_gt2_pc1_weight, w->de_gt2_pc1_bias,
                                       NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1, conv_workspace);
        bn_stream(w->de_gt2_bn1_gamma, w->de_gt2_bn1_beta, w->de_gt2_bn1_mean, w->de_gt2_bn1_var,
                  h1, 16, freq_down);
        prelu_stream(w->de_gt2_prelu1, h1, 16, freq_down);

        /* 缓存布局: (16, 2, 33) - 独立缓存 */
        /* 公式: cache_t = (kernel_t - 1) * dilation_t = (3-1)*1 = 2 (不含当前帧) */
        conv_transpose2d_stream_frame(w->de_gt2_dc_weight, w->de_gt2_dc_bias,
                                       s->de_gt2_cache, h1,
                                       buf6, 16, 16, 3, 3, 1, 1, 1, 2, freq_down, freq_down, 16, conv_workspace);
        bn_stream(w->de_gt2_bn2_gamma, w->de_gt2_bn2_beta, w->de_gt2_bn2_mean, w->de_gt2_bn2_var,
                  buf6, 16, freq_down);
        prelu_stream(w->de_gt2_prelu2, buf6, 16, freq_down);

        conv_transpose2d_stream_frame(w->de_gt2_pc2_weight, w->de_gt2_pc2_bias,
                                       NULL, buf6, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, freq_down, freq_down, 1, conv_workspace);
        bn_stream(w->de_gt2_bn3_gamma, w->de_gt2_bn3_beta, w->de_gt2_bn3_mean, w->de_gt2_bn3_var,
                  h1_out, 8, freq_down);

        tra_gru_step(w->de_gt2_tra_gru_ih, w->de_gt2_tra_gru_hh,
                     w->de_gt2_tra_gru_bih, w->de_gt2_tra_gru_bhh,
                     w->de_gt2_tra_fc_weight, w->de_gt2_tra_fc_bias,
                     s->de_tra_h2, h1_out, 8, freq_down, conv_workspace);

        channel_shuffle_stream(h1_out, x2, buf7, 8, freq_down);
    }

    STREAM_DEBUG_PRINT("After DeGT2 (16, 33)", buf7, 16 * 33);

    /* 解码器ConvBlock 3: ConvTranspose2d(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2) */
    /* 添加跳跃连接 */
    for (int i = 0; i < 16 * 33; i++) {
        buf7[i] += s->en_out1[i];
    }
    /* 输入: (16, 33), 输出: (16, 65) */
    conv_transpose2d_stream_frame(w->de_conv3_weight, w->de_conv3_bias,
                                   NULL, buf7,
                                   buf8, 16, 16, 1, 5, 2, 2, 1, 1, 33, 65, 2, scratch);
    bn_stream(w->de_bn3_gamma, w->de_bn3_beta, w->de_bn3_mean, w->de_bn3_var,
              buf8, 16, 65);
    prelu_stream(w->de_prelu3, buf8, 16, 65);

    STREAM_DEBUG_PRINT("After DeConv3 (16, 65)", buf8, 16 * 65);

    /* 解码器ConvBlock 4: ConvTranspose2d(16, 2, (1,5), stride=(1,2), padding=(0,2)) */
    /* 添加跳跃连接 */
    for (int i = 0; i < 16 * 65; i++) {
        buf8[i] += s->en_out0[i];
    }
    /* 输入: (16, 65), 输出: (2, 129) */
    conv_transpose2d_stream_frame(w->de_conv4_weight, w->de_conv4_bias,
                                   NULL, buf8,
                                   mask, 16, 2, 1, 5, 2, 2, 1, 1, 65, freq_erb, 1, scratch);
    bn_stream(w->de_bn4_gamma, w->de_bn4_beta, w->de_bn4_mean, w->de_bn4_var,
              mask, 2, freq_erb);
    tanh_stream(mask, 2, freq_erb);

#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] Mask before ERB (2, 129) - after tanh:\n");
        printf("    mask_real first 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", mask[i]);
        printf("\n    mask_imag first 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", mask[freq_erb + i]);
        printf("\n    mask_real sum: %.6f\n", debug_sum(mask, freq_erb));
        printf("    mask_imag sum: %.6f\n", debug_sum(mask + freq_erb, freq_erb));
    }
#endif

    /* 步骤16: ERB扩展 (2, 129) -> (2, 257) */
    erb_bs_stream(w, mask, buf8, 2);

#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] Mask after ERB expansion (2, 257):\n");
        printf("    mask_real first 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", buf8[i]);
        printf("\n    mask_imag first 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", buf8[freq_in + i]);
        printf("\n    mask_real sum: %.6f\n", debug_sum(buf8, freq_in));
        printf("    mask_imag sum: %.6f\n", debug_sum(buf8 + freq_in, freq_in));
        printf("  [C Stream] Spec input:\n");
        printf("    spec_real first 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", spec_real[i]);
        printf("\n    spec_imag first 10: ");
        for (int i = 0; i < 10; i++) printf("%.6f ", spec_imag[i]);
        printf("\n");
    }
#endif

    /* 步骤17: 应用掩码 */
    /* mask: (2, 257) = [mask_real, mask_imag] */
    /* spec: (257,) = [real, imag] */
    for (int f = 0; f < freq_in; f++) {
        gtcrn_float m_real = buf8[0 * freq_in + f];
        gtcrn_float m_imag = buf8[1 * freq_in + f];
        gtcrn_float s_real = spec_real[f];
        gtcrn_float s_imag = spec_imag[f];
        
        /* 复数乘法: mask * spec */
        out_real[f] = s_real * m_real - s_imag * m_imag;
        out_imag[f] = s_imag * m_real + s_real * m_imag;
    }

#if GTCRN_STREAM_DEBUG
    if (g_stream_frame_count == 6) {
        printf("  [C Stream] Output spectrum (after mask):\n");
        printf("    out_real first 10: ");
        for (int i = 0; i < 10; i++) printf("%.8f ", out_real[i]);
        printf("\n    out_imag first 10: ");
        for (int i = 0; i < 10; i++) printf("%.8f ", out_imag[i]);
        printf("\n    out_real sum: %.6f\n", debug_sum(out_real, freq_in));
        printf("    out_imag sum: %.6f\n", debug_sum(out_imag, freq_in));
    }
#endif

    return GTCRN_OK;
}
