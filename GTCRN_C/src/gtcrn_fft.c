/// <file>gtcrn_fft.c</file>
/// <summary>GTCRN FFT/STFT实现</summary>
/// <author>江月希 李文轩</author>

#include "gtcrn_fft.h"
#include "gtcrn_math.h"
#include <stdlib.h>
#include <string.h>

// FFT实现 (Cooley-Tukey基2算法)

static int is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static int bit_reverse_index(int x, int log2n) {
    int result = 0;
    for (int i = 0; i < log2n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

gtcrn_fft_plan_t* gtcrn_fft_plan_create(int n) {
    if (!is_power_of_two(n)) {
        return NULL;
    }

    gtcrn_fft_plan_t* plan = (gtcrn_fft_plan_t*)malloc(sizeof(gtcrn_fft_plan_t));
    if (!plan) return NULL;

    plan->n = n;
    plan->cos_table = (gtcrn_float*)malloc(n / 2 * sizeof(gtcrn_float));
    plan->sin_table = (gtcrn_float*)malloc(n / 2 * sizeof(gtcrn_float));
    plan->bit_reverse = (int*)malloc(n * sizeof(int));

    if (!plan->cos_table || !plan->sin_table || !plan->bit_reverse) {
        gtcrn_fft_plan_destroy(plan);
        return NULL;
    }

    /* 构建旋转因子表 */
    for (int i = 0; i < n / 2; i++) {
        gtcrn_float angle = -2.0f * (gtcrn_float)M_PI * i / n;
        plan->cos_table[i] = cosf(angle);
        plan->sin_table[i] = sinf(angle);
    }

    /* 构建位反转表 */
    int log2n = 0;
    int temp = n;
    while (temp > 1) {
        log2n++;
        temp >>= 1;
    }

    for (int i = 0; i < n; i++) {
        plan->bit_reverse[i] = bit_reverse_index(i, log2n);
    }

    return plan;
}

void gtcrn_fft_plan_destroy(gtcrn_fft_plan_t* plan) {
    if (plan) {
        free(plan->cos_table);
        free(plan->sin_table);
        free(plan->bit_reverse);
        free(plan);
    }
}

void gtcrn_fft_forward(gtcrn_fft_plan_t* plan,
                       gtcrn_float* real, gtcrn_float* imag) {
    int n = plan->n;

    /* 位反转置换 */
    for (int i = 0; i < n; i++) {
        int j = plan->bit_reverse[i];
        if (i < j) {
            gtcrn_float tr = real[i];
            gtcrn_float ti = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = tr;
            imag[j] = ti;
        }
    }

    /* Cooley-Tukey FFT蝶形运算 */
    for (int size = 2; size <= n; size *= 2) {
        int halfsize = size / 2;
        int tablestep = n / size;

        for (int i = 0; i < n; i += size) {
            for (int j = 0, k = 0; j < halfsize; j++, k += tablestep) {
                int idx1 = i + j;
                int idx2 = i + j + halfsize;

                gtcrn_float tpre = real[idx2] * plan->cos_table[k] - imag[idx2] * plan->sin_table[k];
                gtcrn_float tpim = real[idx2] * plan->sin_table[k] + imag[idx2] * plan->cos_table[k];

                real[idx2] = real[idx1] - tpre;
                imag[idx2] = imag[idx1] - tpim;
                real[idx1] += tpre;
                imag[idx1] += tpim;
            }
        }
    }
}

void gtcrn_fft_inverse(gtcrn_fft_plan_t* plan,
                       gtcrn_float* real, gtcrn_float* imag) {
    int n = plan->n;

    /* 共轭 */
    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i];
    }

    /* 正向FFT */
    gtcrn_fft_forward(plan, real, imag);

    /* 再次共轭并缩放 */
    gtcrn_float scale = 1.0f / n;
    for (int i = 0; i < n; i++) {
        real[i] *= scale;
        imag[i] = -imag[i] * scale;
    }
}

// STFT实现

gtcrn_stft_t* gtcrn_stft_create(int n_fft, int hop_length, int win_length) {
    gtcrn_stft_t* stft = (gtcrn_stft_t*)malloc(sizeof(gtcrn_stft_t));
    if (!stft) return NULL;

    stft->n_fft = n_fft;
    stft->hop_length = hop_length;
    stft->win_length = win_length;

    stft->window = (gtcrn_float*)malloc(win_length * sizeof(gtcrn_float));
    stft->fft_buffer = (gtcrn_float*)malloc(2 * n_fft * sizeof(gtcrn_float));
    stft->fft_plan = gtcrn_fft_plan_create(n_fft);

    if (!stft->window || !stft->fft_buffer || !stft->fft_plan) {
        gtcrn_stft_destroy(stft);
        return NULL;
    }

    /* 创建平方根汉宁窗 */
    for (int i = 0; i < win_length; i++) {
        gtcrn_float hann = 0.5f * (1.0f - cosf(2.0f * (gtcrn_float)M_PI * i / win_length));
        stft->window[i] = sqrtf(hann);
    }

    return stft;
}

void gtcrn_stft_destroy(gtcrn_stft_t* stft) {
    if (stft) {
        free(stft->window);
        free(stft->fft_buffer);
        gtcrn_fft_plan_destroy(stft->fft_plan);
        free(stft);
    }
}

int gtcrn_stft_num_frames(int signal_len, int n_fft, int hop_length) {
    return (signal_len - n_fft) / hop_length + 1;
}

void gtcrn_stft_frame(gtcrn_stft_t* stft,
                      const gtcrn_float* frame,
                      gtcrn_float* spec_real, gtcrn_float* spec_imag) {
    int n_fft = stft->n_fft;
    int win_length = stft->win_length;
    gtcrn_float* real = stft->fft_buffer;
    gtcrn_float* imag = stft->fft_buffer + n_fft;

    /* 零填充并加窗 */
    int pad_left = (n_fft - win_length) / 2;
    memset(real, 0, n_fft * sizeof(gtcrn_float));
    memset(imag, 0, n_fft * sizeof(gtcrn_float));

    for (int i = 0; i < win_length; i++) {
        real[pad_left + i] = frame[i] * stft->window[i];
    }

    /* 计算FFT */
    gtcrn_fft_forward(stft->fft_plan, real, imag);

    /* 复制正频率分量(包含直流和奈奎斯特频率) */
    int n_freqs = n_fft / 2 + 1;
    memcpy(spec_real, real, n_freqs * sizeof(gtcrn_float));
    memcpy(spec_imag, imag, n_freqs * sizeof(gtcrn_float));
}

int gtcrn_stft_forward(gtcrn_stft_t* stft,
                       const gtcrn_float* signal, int signal_len,
                       gtcrn_float* spec_real, gtcrn_float* spec_imag) {
    int n_fft = stft->n_fft;
    int hop_length = stft->hop_length;
    int n_freqs = n_fft / 2 + 1;

    int n_frames = gtcrn_stft_num_frames(signal_len, n_fft, hop_length);

    for (int t = 0; t < n_frames; t++) {
        const gtcrn_float* frame = signal + t * hop_length;
        gtcrn_stft_frame(stft, frame,
                        spec_real + t * n_freqs,
                        spec_imag + t * n_freqs);
    }

    return n_frames;
}

void gtcrn_istft_frame(gtcrn_stft_t* stft,
                       const gtcrn_float* spec_real, const gtcrn_float* spec_imag,
                       gtcrn_float* output) {
    int n_fft = stft->n_fft;
    int win_length = stft->win_length;
    int n_freqs = n_fft / 2 + 1;
    gtcrn_float* real = stft->fft_buffer;
    gtcrn_float* imag = stft->fft_buffer + n_fft;

    /* 复制频谱并保持共轭对称性 */
    memcpy(real, spec_real, n_freqs * sizeof(gtcrn_float));
    memcpy(imag, spec_imag, n_freqs * sizeof(gtcrn_float));

    /* 重构负频率分量(共轭对称) */
    for (int i = 1; i < n_fft / 2; i++) {
        real[n_fft - i] = real[i];
        imag[n_fft - i] = -imag[i];
    }

    /* 逆FFT */
    gtcrn_fft_inverse(stft->fft_plan, real, imag);

    /* 加窗并提取 */
    int pad_left = (n_fft - win_length) / 2;
    for (int i = 0; i < win_length; i++) {
        output[i] = real[pad_left + i] * stft->window[i];
    }
}

void gtcrn_istft(gtcrn_stft_t* stft,
                 const gtcrn_float* spec_real, const gtcrn_float* spec_imag,
                 int n_frames,
                 gtcrn_float* signal, int signal_len) {
    int n_fft = stft->n_fft;
    int hop_length = stft->hop_length;
    int win_length = stft->win_length;
    int n_freqs = n_fft / 2 + 1;

    /* 分配临时帧缓冲区 */
    gtcrn_float* frame_out = (gtcrn_float*)malloc(win_length * sizeof(gtcrn_float));
    gtcrn_float* window_sum = (gtcrn_float*)calloc(signal_len, sizeof(gtcrn_float));

    if (!frame_out || !window_sum) {
        free(frame_out);
        free(window_sum);
        return;
    }

    /* 清零输出 */
    memset(signal, 0, signal_len * sizeof(gtcrn_float));

    /* 重叠相加 */
    for (int t = 0; t < n_frames; t++) {
        gtcrn_istft_frame(stft,
                         spec_real + t * n_freqs,
                         spec_imag + t * n_freqs,
                         frame_out);

        int start = t * hop_length;
        for (int i = 0; i < win_length && start + i < signal_len; i++) {
            signal[start + i] += frame_out[i];
            window_sum[start + i] += stft->window[i] * stft->window[i];
        }
    }

    /* 按窗函数和归一化 */
    for (int i = 0; i < signal_len; i++) {
        if (window_sum[i] > 1e-8f) {
            signal[i] /= window_sum[i];
        }
    }

    free(frame_out);
    free(window_sum);
}
