/// <file>debug_forward.c</file>
/// <summary>逐层对比C实现与PyTorch输出,调试GTCRN前向传播</summary>
/// <author>李文轩</author>

#include "gtcrn_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 前向声明 */
extern void gtcrn_forward_complete_with_workspace(gtcrn_t* model,
                                                   const gtcrn_float* spec_real,
                                                   const gtcrn_float* spec_imag,
                                                   gtcrn_float* out_real,
                                                   gtcrn_float* out_imag,
                                                   int n_frames,
                                                   gtcrn_float* workspace);

/* 张量求和用于调试 */
static double tensor_sum(const gtcrn_float* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

/* 张量统计 */
static void tensor_stats(const gtcrn_float* data, int size, double* sum, double* min, double* max) {
    *sum = 0.0;
    *min = data[0];
    *max = data[0];
    for (int i = 0; i < size; i++) {
        *sum += data[i];
        if (data[i] < *min) *min = data[i];
        if (data[i] > *max) *max = data[i];
    }
}

/* Load binary tensor */
static gtcrn_float* load_tensor(const char* filepath, int expected_size) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;

    gtcrn_float* data = (gtcrn_float*)malloc(expected_size * sizeof(gtcrn_float));
    size_t read = fread(data, sizeof(gtcrn_float), expected_size, f);
    fclose(f);

    if (read != (size_t)expected_size) {
        free(data);
        return NULL;
    }
    return data;
}

/* Compare tensors */
static void compare_tensors(const char* name, const gtcrn_float* c_data,
                            const gtcrn_float* py_data, int size) {
    double max_diff = 0.0, mean_diff = 0.0;
    int max_idx = 0;

    for (int i = 0; i < size; i++) {
        double diff = fabs(c_data[i] - py_data[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
        mean_diff += diff;
    }
    mean_diff /= size;

    const char* status = max_diff < 0.001 ? "PASS" : (max_diff < 0.1 ? "WARN" : "FAIL");
    printf("%s: max_diff=%.6f (at %d), mean_diff=%.6f [%s]\n",
           name, max_diff, max_idx, mean_diff, status);

    if (max_diff >= 0.001) {
        printf("  C[%d]=%.6f, Py[%d]=%.6f, diff=%.6f\n",
               max_idx, c_data[max_idx], max_idx, py_data[max_idx], max_diff);

        /* Print first few values */
        printf("  First 5 C values: ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.4f ", c_data[i]);
        printf("\n");
        printf("  First 5 Py values: ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.4f ", py_data[i]);
        printf("\n");
    }
}

/* ============================================================================
 * Step-by-Step Forward for Debugging
 * ============================================================================ */

/* SFE forward (from gtcrn_forward.c) */
static void sfe_forward_debug(const gtcrn_float* input,
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

/* ERB bm forward */
static void erb_bm_debug(const gtcrn_weights_t* w,
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
                /* Low freq: direct copy */
                for (int f = 0; f < erb_sub1; f++) {
                    output[GTCRN_IDX4(b, c, t, f, channels, time, freq_out)] =
                        input[GTCRN_IDX4(b, c, t, f, channels, time, freq_in)];
                }
                /* High freq: ERB compression */
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

int main(int argc, char** argv) {
    const char* weights_path = "weights/gtcrn_weights.bin";
    const char* test_dir = "test_data";

    printf("=== GTCRN Layer-by-Layer Debug ===\n\n");

    /* Create and load model */
    gtcrn_t* model = gtcrn_create();
    if (!model || gtcrn_load_weights(model, weights_path) != GTCRN_OK) {
        printf("Failed to load model\n");
        return 1;
    }

    printf("Model loaded\n\n");

    /* Load input */
    int n_frames = 10;
    int freq_in = 257;
    int freq_erb = 129;

    char path[256];
    snprintf(path, sizeof(path), "%s/spec_real.bin", test_dir);
    gtcrn_float* spec_real = load_tensor(path, n_frames * freq_in);

    snprintf(path, sizeof(path), "%s/spec_imag.bin", test_dir);
    gtcrn_float* spec_imag = load_tensor(path, n_frames * freq_in);

    if (!spec_real || !spec_imag) {
        printf("无法加载输入。请先运行create_test_cases.py。\n");
        gtcrn_destroy(model);
        return 1;
    }

    printf("Input loaded: spec_real sum=%.6f, spec_imag sum=%.6f\n\n",
           tensor_sum(spec_real, n_frames * freq_in),
           tensor_sum(spec_imag, n_frames * freq_in));

    /* Allocate workspace */
    size_t buf_size = 16 * n_frames * freq_in;
    gtcrn_float* buf1 = (gtcrn_float*)calloc(buf_size, sizeof(gtcrn_float));
    gtcrn_float* buf2 = (gtcrn_float*)calloc(buf_size, sizeof(gtcrn_float));

    /* Step 1: Feature tensor */
    printf("Step 1: Feature tensor\n");
    for (int t = 0; t < n_frames; t++) {
        for (int f = 0; f < freq_in; f++) {
            gtcrn_float r = spec_real[t * freq_in + f];
            gtcrn_float i = spec_imag[t * freq_in + f];
            gtcrn_float mag = sqrtf(r * r + i * i + GTCRN_EPS);
            buf1[GTCRN_IDX4(0, 0, t, f, 3, n_frames, freq_in)] = mag;
            buf1[GTCRN_IDX4(0, 1, t, f, 3, n_frames, freq_in)] = r;
            buf1[GTCRN_IDX4(0, 2, t, f, 3, n_frames, freq_in)] = i;
        }
    }
    printf("  C feat sum: %.6f\n", tensor_sum(buf1, 3 * n_frames * freq_in));

    snprintf(path, sizeof(path), "%s/py_feat.bin", test_dir);
    gtcrn_float* py_feat = load_tensor(path, 3 * n_frames * freq_in);
    if (py_feat) {
        compare_tensors("  feat", buf1, py_feat, 3 * n_frames * freq_in);
        free(py_feat);
    }

    /* Step 2: ERB compression */
    printf("\nStep 2: ERB bm\n");
    erb_bm_debug(model->weights, buf1, buf2, 1, 3, n_frames);
    printf("  C erb sum: %.6f\n", tensor_sum(buf2, 3 * n_frames * freq_erb));

    snprintf(path, sizeof(path), "%s/py_feat_erb.bin", test_dir);
    gtcrn_float* py_erb = load_tensor(path, 3 * n_frames * freq_erb);
    if (py_erb) {
        compare_tensors("  erb", buf2, py_erb, 3 * n_frames * freq_erb);
        free(py_erb);
    }

    /* Step 3: SFE */
    printf("\nStep 3: SFE\n");
    sfe_forward_debug(buf2, buf1, 1, 3, n_frames, freq_erb);
    printf("  C sfe sum: %.6f\n", tensor_sum(buf1, 9 * n_frames * freq_erb));

    snprintf(path, sizeof(path), "%s/py_feat_sfe.bin", test_dir);
    gtcrn_float* py_sfe = load_tensor(path, 9 * n_frames * freq_erb);
    if (py_sfe) {
        compare_tensors("  sfe", buf1, py_sfe, 9 * n_frames * freq_erb);
        free(py_sfe);
    }

    /* Steps 4-13: Full forward pass */
    printf("\n=== Full Forward Pass ===\n");
    gtcrn_float* out_real = (gtcrn_float*)malloc(n_frames * freq_in * sizeof(gtcrn_float));
    gtcrn_float* out_imag = (gtcrn_float*)malloc(n_frames * freq_in * sizeof(gtcrn_float));

    /* Allocate workspace for forward pass */
    size_t forward_ws_size = 4 * 16 * n_frames * freq_in;
    gtcrn_float* forward_workspace = (gtcrn_float*)calloc(forward_ws_size, sizeof(gtcrn_float));

    gtcrn_forward_complete_with_workspace(model, spec_real, spec_imag, out_real, out_imag,
                                          n_frames, forward_workspace);

    free(forward_workspace);

    printf("C output: real sum=%.6f, imag sum=%.6f\n",
           tensor_sum(out_real, n_frames * freq_in),
           tensor_sum(out_imag, n_frames * freq_in));

    /* Compare with PyTorch full output */
    snprintf(path, sizeof(path), "%s/py_full_output.bin", test_dir);
    gtcrn_float* py_output = load_tensor(path, freq_in * n_frames * 2);

    if (py_output) {
        /* Reconstruct C output in same format as PyTorch: (1, 257, 10, 2) */
        gtcrn_float* c_output = (gtcrn_float*)malloc(freq_in * n_frames * 2 * sizeof(gtcrn_float));
        for (int f = 0; f < freq_in; f++) {
            for (int t = 0; t < n_frames; t++) {
                c_output[(f * n_frames + t) * 2 + 0] = out_real[t * freq_in + f];
                c_output[(f * n_frames + t) * 2 + 1] = out_imag[t * freq_in + f];
            }
        }

        printf("\nFull output comparison:\n");
        compare_tensors("  output", c_output, py_output, freq_in * n_frames * 2);

        free(c_output);
        free(py_output);
    }

    /* Cleanup */
    free(buf1);
    free(buf2);
    free(spec_real);
    free(spec_imag);
    free(out_real);
    free(out_imag);
    gtcrn_destroy(model);

    return 0;
}
