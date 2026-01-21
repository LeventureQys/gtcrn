/// <file>gtcrn_model.c</file>
/// <summary>GTCRN模型实现</summary>
/// <author>江月希 李文轩</author>

#include "gtcrn_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 从gtcrn_forward.c前向声明完整前向传播函数 */
extern void gtcrn_forward_complete_with_workspace(gtcrn_t* model,
                                                   const gtcrn_float* spec_real,
                                                   const gtcrn_float* spec_imag,
                                                   gtcrn_float* out_real,
                                                   gtcrn_float* out_imag,
                                                   int n_frames,
                                                   gtcrn_float* workspace);

// 内部辅助函数

/* SFE: 使用unfold的子带特征提取 */
static void gtcrn_sfe_forward(const gtcrn_float* input,
                              gtcrn_float* output,
                              int batch, int channels, int time, int freq) {
    /* 使用kernel=3, padding=1的unfold: 输出通道数 = 输入通道数 * 3 */
    int out_channels = channels * 3;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < freq; f++) {
                    /* 带填充的邻域索引 */
                    int f_left = f - 1;
                    int f_right = f + 1;

                    gtcrn_float v_left = (f_left >= 0) ?
                        input[GTCRN_IDX4(b, c, t, f_left, channels, time, freq)] : 0.0f;
                    gtcrn_float v_center = input[GTCRN_IDX4(b, c, t, f, channels, time, freq)];
                    gtcrn_float v_right = (f_right < freq) ?
                        input[GTCRN_IDX4(b, c, t, f_right, channels, time, freq)] : 0.0f;

                    /* 输出布局: (B, C*3, T, F) */
                    output[GTCRN_IDX4(b, c * 3 + 0, t, f, out_channels, time, freq)] = v_left;
                    output[GTCRN_IDX4(b, c * 3 + 1, t, f, out_channels, time, freq)] = v_center;
                    output[GTCRN_IDX4(b, c * 3 + 2, t, f, out_channels, time, freq)] = v_right;
                }
            }
        }
    }
}

/* ERB压缩: bm (频带映射) */
static void gtcrn_erb_bm(const gtcrn_weights_t* w,
                         const gtcrn_float* input,
                         gtcrn_float* output,
                         int batch, int channels, int time) {
    int erb_sub1 = GTCRN_ERB_SUBBAND_1;  /* 65 */
    int erb_sub2 = GTCRN_ERB_SUBBAND_2;  /* 64 */
    int freq_in = GTCRN_FREQ_BINS;       /* 257 */
    int freq_out = GTCRN_ERB_TOTAL;      /* 129 */

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                /* 直接复制低频bin */
                for (int f = 0; f < erb_sub1; f++) {
                    output[GTCRN_IDX4(b, c, t, f, channels, time, freq_out)] =
                        input[GTCRN_IDX4(b, c, t, f, channels, time, freq_in)];
                }

                /* 对高频应用ERB压缩 */
                /* erb_fc: (erb_sub2, freq_in - erb_sub1) = (64, 192) */
                for (int f_out = 0; f_out < erb_sub2; f_out++) {
                    gtcrn_float sum = 0.0f;
                    for (int f_in = 0; f_in < freq_in - erb_sub1; f_in++) {
                        sum += w->erb_fc_weight[f_out * (freq_in - erb_sub1) + f_in] *
                               input[GTCRN_IDX4(b, c, t, erb_sub1 + f_in, channels, time, freq_in)];
                    }
                    output[GTCRN_IDX4(b, c, t, erb_sub1 + f_out, channels, time, freq_out)] = sum;
                }
            }
        }
    }
}

/* ERB扩展: bs (频带合成) */
static void gtcrn_erb_bs(const gtcrn_weights_t* w,
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
                /* 直接复制低频bin */
                for (int f = 0; f < erb_sub1; f++) {
                    output[GTCRN_IDX4(b, c, t, f, channels, time, freq_out)] =
                        input[GTCRN_IDX4(b, c, t, f, channels, time, freq_in)];
                }

                /* 对高频应用逆ERB */
                /* ierb_fc: (freq_out - erb_sub1, erb_sub2) = (192, 64) */
                for (int f_out = 0; f_out < freq_out - erb_sub1; f_out++) {
                    gtcrn_float sum = 0.0f;
                    for (int f_in = 0; f_in < erb_sub2; f_in++) {
                        sum += w->ierb_fc_weight[f_out * erb_sub2 + f_in] *
                               input[GTCRN_IDX4(b, c, t, erb_sub1 + f_in, channels, time, freq_in)];
                    }
                    output[GTCRN_IDX4(b, c, t, erb_sub1 + f_out, channels, time, freq_out)] = sum;
                }
            }
        }
    }
}

/* GTConvBlock的通道混洗 */
static void gtcrn_channel_shuffle(const gtcrn_float* x1, const gtcrn_float* x2,
                                  gtcrn_float* output,
                                  int batch, int half_channels, int time, int freq) {
    int out_channels = half_channels * 2;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < half_channels; c++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < freq; f++) {
                    /* 交错: [c0_x1, c0_x2, c1_x1, c1_x2, ...] */
                    output[GTCRN_IDX4(b, c * 2, t, f, out_channels, time, freq)] =
                        x1[GTCRN_IDX4(b, c, t, f, half_channels, time, freq)];
                    output[GTCRN_IDX4(b, c * 2 + 1, t, f, out_channels, time, freq)] =
                        x2[GTCRN_IDX4(b, c, t, f, half_channels, time, freq)];
                }
            }
        }
    }
}

/* 复数比率掩码应用 */
static void gtcrn_apply_mask(const gtcrn_float* mask,
                             const gtcrn_float* spec_real,
                             const gtcrn_float* spec_imag,
                             gtcrn_float* out_real,
                             gtcrn_float* out_imag,
                             int batch, int time, int freq) {
    /* mask: (B, 2, T, F) 其中mask[:,0]是实部掩码, mask[:,1]是虚部掩码 */
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < time; t++) {
            for (int f = 0; f < freq; f++) {
                gtcrn_float mr = mask[GTCRN_IDX4(b, 0, t, f, 2, time, freq)];
                gtcrn_float mi = mask[GTCRN_IDX4(b, 1, t, f, 2, time, freq)];
                gtcrn_float sr = spec_real[b * time * freq + t * freq + f];
                gtcrn_float si = spec_imag[b * time * freq + t * freq + f];

                /* 复数乘法: (sr + j*si) * (mr + j*mi) */
                out_real[b * time * freq + t * freq + f] = sr * mr - si * mi;
                out_imag[b * time * freq + t * freq + f] = si * mr + sr * mi;
            }
        }
    }
}

// API实现
size_t gtcrn_get_workspace_size(void) {
    /* gtcrn_process的工作空间布局:
     * 1. STFT/ISTFT缓冲区: 4 * max_frames * 257 (spec_real, spec_imag, out_real, out_imag)
     * 2. 前向传播缓冲区: 4 * 16 * max_frames * 257 (buf1, buf2, buf3, scratch)
     *    加上GTConvBlock工作空间的额外空间
     */
    size_t max_frames = 2048;  /* 支持最多约32秒 */
    size_t stft_buffers = 4 * max_frames * GTCRN_FREQ_BINS * sizeof(gtcrn_float);
    size_t forward_buffers = 4 * 16 * max_frames * GTCRN_FREQ_BINS * sizeof(gtcrn_float);
    size_t extra = 4 * 1024 * 1024;  /* GTConvBlock工作空间额外4MB */
    return stft_buffers + forward_buffers + extra;
}

gtcrn_t* gtcrn_create(void) {
    gtcrn_t* model = (gtcrn_t*)calloc(1, sizeof(gtcrn_t));
    if (!model) return NULL;

    model->weights = (gtcrn_weights_t*)calloc(1, sizeof(gtcrn_weights_t));
    model->state = (gtcrn_state_t*)calloc(1, sizeof(gtcrn_state_t));
    model->stft = gtcrn_stft_create(GTCRN_FFT_SIZE, GTCRN_HOP_SIZE, GTCRN_WIN_SIZE);

    model->workspace_size = gtcrn_get_workspace_size();
    model->workspace = (gtcrn_float*)malloc(model->workspace_size);

    if (!model->weights || !model->state || !model->stft || !model->workspace) {
        gtcrn_destroy(model);
        return NULL;
    }

    model->is_initialized = 0;
    return model;
}

void gtcrn_destroy(gtcrn_t* model) {
    if (model) {
        free(model->weights);
        free(model->state);
        gtcrn_stft_destroy(model->stft);
        free(model->workspace);
        free(model);
    }
}

gtcrn_status_t gtcrn_load_weights(gtcrn_t* model, const char* filepath) {
    if (!model || !filepath) return GTCRN_ERROR_NULL_POINTER;

    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open weights file: %s\n", filepath);
        return GTCRN_ERROR_FILE_IO;
    }

    /* 读取文件头(魔数 + 版本) */
    char magic[4];
    uint32_t version;
    if (fread(magic, 1, 4, fp) != 4 || fread(&version, 4, 1, fp) != 1) {
        fclose(fp);
        return GTCRN_ERROR_INVALID_FORMAT;
    }

    /* Accept both 'GTCN' (complete) and 'GTCS' (stream) format */
    if (memcmp(magic, "GTCN", 4) != 0 && memcmp(magic, "GTCS", 4) != 0) {
        fprintf(stderr, "Error: Invalid weights file format (expected GTCN or GTCS)\n");
        fclose(fp);
        return GTCRN_ERROR_INVALID_FORMAT;
    }

    /* 读取所有权重 */
    size_t weights_size = sizeof(gtcrn_weights_t);
    size_t read_size = fread(model->weights, 1, weights_size, fp);
    fclose(fp);

    if (read_size != weights_size) {
        fprintf(stderr, "Error: Incomplete weights file (read %zu, expected %zu)\n",
                read_size, weights_size);
        return GTCRN_ERROR_INVALID_FORMAT;
    }

    model->is_initialized = 1;
    return GTCRN_OK;
}

void gtcrn_reset_state(gtcrn_t* model) {
    if (model && model->state) {
        memset(model->state, 0, sizeof(gtcrn_state_t));
        /* 标记为第一帧,用于流式ISTFT重叠相加初始化 */
        model->state->first_frame = 1;
    }
}

/* Forward declaration of internal processing functions */
static void gtcrn_forward_offline(gtcrn_t* model,
                                  const gtcrn_float* spec_real,
                                  const gtcrn_float* spec_imag,
                                  gtcrn_float* out_real,
                                  gtcrn_float* out_imag,
                                  int n_frames);

gtcrn_status_t gtcrn_process(gtcrn_t* model,
                             const gtcrn_float* input, int input_len,
                             gtcrn_float* output, int* output_len) {
    if (!model || !input || !output || !output_len) {
        return GTCRN_ERROR_NULL_POINTER;
    }
    if (!model->is_initialized) {
        return GTCRN_ERROR_NOT_INITIALIZED;
    }

    /* 计算帧数 */
    int n_frames = gtcrn_stft_num_frames(input_len, GTCRN_FFT_SIZE, GTCRN_HOP_SIZE);
    if (n_frames <= 0) {
        *output_len = 0;
        return GTCRN_OK;
    }

    /* 检查帧数是否超过最大支持 */
    size_t max_frames = 2048;
    if ((size_t)n_frames > max_frames) {
        fprintf(stderr, "Error: Audio too long (%d frames > %zu max)\n", n_frames, max_frames);
        return GTCRN_ERROR_INVALID_PARAM;
    }

    /* 工作空间布局:
     * - spec_real:    [0, n_frames * 257)
     * - spec_imag:    [n_frames * 257, 2 * n_frames * 257)
     * - out_real:     [2 * n_frames * 257, 3 * n_frames * 257)
     * - out_imag:     [3 * n_frames * 257, 4 * n_frames * 257)
     * - forward_work: [4 * n_frames * 257, ...)  -- 用于神经网络前向传播
     */
    int n_freqs = GTCRN_FREQ_BINS;
    size_t spec_buffer_offset = 4 * n_frames * n_freqs;  /* Reserve space for spectrums */

    gtcrn_float* spec_real = model->workspace;
    gtcrn_float* spec_imag = spec_real + n_frames * n_freqs;
    gtcrn_float* out_real = spec_imag + n_frames * n_freqs;
    gtcrn_float* out_imag = out_real + n_frames * n_freqs;
    gtcrn_float* forward_workspace = out_imag + n_frames * n_freqs;

    /* STFT */
    gtcrn_stft_forward(model->stft, input, input_len, spec_real, spec_imag);

    /* 神经网络前向传播 - 传递独立的前向工作空间 */
    gtcrn_forward_complete_with_workspace(model, spec_real, spec_imag, out_real, out_imag,
                                          n_frames, forward_workspace);

    /* ISTFT */
    *output_len = input_len;
    gtcrn_istft(model->stft, out_real, out_imag, n_frames, output, *output_len);

    return GTCRN_OK;
}

// 流式帧处理

// 前向声明
gtcrn_status_t gtcrn_process_frame_impl(gtcrn_t* model,
                                        const gtcrn_float* spec_real,
                                        const gtcrn_float* spec_imag,
                                        gtcrn_float* out_real,
                                        gtcrn_float* out_imag);

gtcrn_status_t gtcrn_process_frame(gtcrn_t* model,
                                   const gtcrn_float* input_frame,
                                   gtcrn_float* output_frame) {
    if (!model || !input_frame || !output_frame) {
        return GTCRN_ERROR_NULL_POINTER;
    }
    if (!model->is_initialized) {
        return GTCRN_ERROR_NOT_INITIALIZED;
    }

    gtcrn_state_t* state = model->state;

    /* 为STFT/ISTFT分配临时缓冲区 */
    gtcrn_float* stft_window = model->workspace;  /* 512 samples for STFT input */
    gtcrn_float* spec_real = stft_window + GTCRN_WIN_SIZE;
    gtcrn_float* spec_imag = spec_real + GTCRN_FREQ_BINS;
    gtcrn_float* out_spec_real = spec_imag + GTCRN_FREQ_BINS;
    gtcrn_float* out_spec_imag = out_spec_real + GTCRN_FREQ_BINS;
    gtcrn_float* istft_frame = out_spec_imag + GTCRN_FREQ_BINS;  /* 512 samples ISTFT output */

    /* 步骤1: 构建512样本窗口用于STFT */
    /* 窗口 = [前一帧的256样本] + [当前帧的256样本] */
    memcpy(stft_window, state->stft_input_buffer, GTCRN_HOP_SIZE * sizeof(gtcrn_float));
    memcpy(stft_window + GTCRN_HOP_SIZE, input_frame, GTCRN_HOP_SIZE * sizeof(gtcrn_float));

    /* 更新输入缓冲区为下一帧 */
    memcpy(state->stft_input_buffer, input_frame, GTCRN_HOP_SIZE * sizeof(gtcrn_float));

    /* 步骤2: STFT - 将时域帧转换为频域 */
    gtcrn_stft_frame(model->stft, stft_window, spec_real, spec_imag);

    /* 步骤3: 在频域处理单帧 */
    gtcrn_status_t status = gtcrn_process_frame_impl(model, spec_real, spec_imag,
                                                     out_spec_real, out_spec_imag);
    if (status != GTCRN_OK) {
        return status;
    }

    /* 步骤4: ISTFT - 得到512样本输出 */
    gtcrn_istft_frame(model->stft, out_spec_real, out_spec_imag, istft_frame);

    /* 步骤5: 重叠相加生成256样本输出 */
    /* 输出 = OLA_buffer[0:256] + istft_frame[0:256] */
    /* 然后更新OLA_buffer = istft_frame[256:512] */
    if (state->first_frame) {
        /* 第一帧: 直接输出前半部分 */
        for (int i = 0; i < GTCRN_HOP_SIZE; i++) {
            output_frame[i] = istft_frame[i];
        }
        state->first_frame = 0;
    } else {
        /* 后续帧: 重叠相加 */
        for (int i = 0; i < GTCRN_HOP_SIZE; i++) {
            output_frame[i] = state->ola_buffer[i] + istft_frame[i];
        }
    }

    /* 保存后半部分用于下一帧的重叠相加 */
    memcpy(state->ola_buffer, istft_frame + GTCRN_HOP_SIZE, GTCRN_HOP_SIZE * sizeof(gtcrn_float));

    return GTCRN_OK;
}
