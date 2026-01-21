/// <file>debug_full_forward.c</file>
/// <summary>逐步调试完整前向传播</summary>
/// <author>李文轩</author>

#include "gtcrn_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 张量求和用于调试 */
static double tensor_sum(const gtcrn_float* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

/* 加载二进制张量 */
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

    if (max_diff >= 0.001 && size > 0) {
        printf("  C[%d]=%.6f, Py[%d]=%.6f\n",
               max_idx, c_data[max_idx], max_idx, py_data[max_idx]);

        /* Print first few values */
        printf("  First 5 C: ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.6f ", c_data[i]);
        printf("\n");
        printf("  First 5 Py: ");
        for (int i = 0; i < 5 && i < size; i++) printf("%.6f ", py_data[i]);
        printf("\n");
    }
}

/* SFE forward (same as in gtcrn_forward.c) */
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

/* ERB bm forward */
static void erb_bm(const gtcrn_weights_t* w,
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

    printf("=== Full Forward Pass Debug ===\n\n");

    /* Create and load model */
    gtcrn_t* model = gtcrn_create();
    if (!model || gtcrn_load_weights(model, weights_path) != GTCRN_OK) {
        printf("Failed to load model\n");
        return 1;
    }

    gtcrn_weights_t* w = model->weights;

    int n_frames = 10;
    int freq_in = 257;
    int freq_erb = 129;

    /* Load input */
    char path[256];
    snprintf(path, sizeof(path), "%s/spec_real.bin", test_dir);
    gtcrn_float* spec_real = load_tensor(path, n_frames * freq_in);
    snprintf(path, sizeof(path), "%s/spec_imag.bin", test_dir);
    gtcrn_float* spec_imag = load_tensor(path, n_frames * freq_in);

    if (!spec_real || !spec_imag) {
        printf("无法加载输入\n");
        gtcrn_destroy(model);
        return 1;
    }

    /* Allocate buffers */
    size_t buf_size = 16 * n_frames * freq_in;
    gtcrn_float* buf1 = (gtcrn_float*)calloc(buf_size, sizeof(gtcrn_float));
    gtcrn_float* buf2 = (gtcrn_float*)calloc(buf_size, sizeof(gtcrn_float));
    gtcrn_float* scratch = (gtcrn_float*)calloc(buf_size, sizeof(gtcrn_float));

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
    erb_bm(w, buf1, buf2, 1, 3, n_frames);
    printf("  C erb sum: %.6f\n", tensor_sum(buf2, 3 * n_frames * freq_erb));

    snprintf(path, sizeof(path), "%s/py_feat_erb.bin", test_dir);
    gtcrn_float* py_erb = load_tensor(path, 3 * n_frames * freq_erb);
    if (py_erb) {
        compare_tensors("  erb", buf2, py_erb, 3 * n_frames * freq_erb);
        free(py_erb);
    }

    /* Step 3: SFE */
    printf("\nStep 3: SFE\n");
    sfe_forward(buf2, buf1, 1, 3, n_frames, freq_erb);
    printf("  C sfe sum: %.6f\n", tensor_sum(buf1, 9 * n_frames * freq_erb));

    snprintf(path, sizeof(path), "%s/py_feat_sfe.bin", test_dir);
    gtcrn_float* py_sfe = load_tensor(path, 9 * n_frames * freq_erb);
    if (py_sfe) {
        compare_tensors("  sfe", buf1, py_sfe, 9 * n_frames * freq_erb);
        free(py_sfe);
    }

    /* Step 4: EnConv0 - use EXACT same pattern as debug_enconv0 which works */
    printf("\nStep 4: EnConv0\n");

    /* First, reload the PyTorch SFE output to ensure we're using same input */
    snprintf(path, sizeof(path), "%s/py_feat_sfe.bin", test_dir);
    gtcrn_float* sfe_input = load_tensor(path, 9 * n_frames * freq_erb);

    if (sfe_input) {
        printf("  Using PyTorch SFE output as input\n");
        printf("  Input sum: %.6f\n", tensor_sum(sfe_input, 9 * n_frames * freq_erb));

        gtcrn_conv2d_t conv = {
            .weight = w->en_conv0_weight,
            .bias = w->en_conv0_bias,
            .in_channels = 9,
            .out_channels = 16,
            .kernel_h = 1,
            .kernel_w = 5,
            .stride_h = 1,
            .stride_w = 2,
            .padding_h = 0,
            .padding_w = 2,
            .dilation_h = 1,
            .dilation_w = 1,
            .groups = 1
        };

        int out_size = 16 * n_frames * 65;
        gtcrn_conv2d_forward(&conv, sfe_input, buf2, 1, n_frames, freq_erb, n_frames, 65);
        printf("  C Conv output sum: %.6f (expected -9600.52)\n", tensor_sum(buf2, out_size));

        /* Now apply BN */
        gtcrn_batchnorm2d_t bn = {
            .gamma = w->en_bn0_gamma,
            .beta = w->en_bn0_beta,
            .running_mean = w->en_bn0_mean,
            .running_var = w->en_bn0_var,
            .num_features = 16,
            .eps = 1e-5f
        };
        gtcrn_batchnorm2d_forward(&bn, buf2, 1, n_frames, 65);
        printf("  C BN output sum: %.6f (expected -614.83)\n", tensor_sum(buf2, out_size));

        /* PReLU */
        gtcrn_prelu_t prelu = {
            .alpha = w->en_prelu0,
            .num_parameters = 1
        };
        gtcrn_prelu_forward(&prelu, buf2, 1, 16, n_frames * 65);
        printf("  C PReLU output sum: %.6f (expected 1000.50)\n", tensor_sum(buf2, out_size));

        snprintf(path, sizeof(path), "%s/py_en_conv0.bin", test_dir);
        gtcrn_float* py_conv0 = load_tensor(path, out_size);
        if (py_conv0) {
            compare_tensors("  EnConv0", buf2, py_conv0, out_size);
            free(py_conv0);
        }

        free(sfe_input);
    }

    /* Now test with C-computed SFE */
    printf("\nStep 4b: EnConv0 with C-computed SFE\n");
    printf("  Using C SFE output sum: %.6f\n", tensor_sum(buf1, 9 * n_frames * freq_erb));

    {
        gtcrn_conv2d_t conv = {
            .weight = w->en_conv0_weight,
            .bias = w->en_conv0_bias,
            .in_channels = 9,
            .out_channels = 16,
            .kernel_h = 1,
            .kernel_w = 5,
            .stride_h = 1,
            .stride_w = 2,
            .padding_h = 0,
            .padding_w = 2,
            .dilation_h = 1,
            .dilation_w = 1,
            .groups = 1
        };

        int out_size = 16 * n_frames * 65;
        gtcrn_conv2d_forward(&conv, buf1, buf2, 1, n_frames, freq_erb, n_frames, 65);
        printf("  C Conv output sum: %.6f (expected -9600.52)\n", tensor_sum(buf2, out_size));

        gtcrn_batchnorm2d_t bn = {
            .gamma = w->en_bn0_gamma,
            .beta = w->en_bn0_beta,
            .running_mean = w->en_bn0_mean,
            .running_var = w->en_bn0_var,
            .num_features = 16,
            .eps = 1e-5f
        };
        gtcrn_batchnorm2d_forward(&bn, buf2, 1, n_frames, 65);
        printf("  C BN output sum: %.6f (expected -614.83)\n", tensor_sum(buf2, out_size));

        gtcrn_prelu_t prelu = {
            .alpha = w->en_prelu0,
            .num_parameters = 1
        };
        gtcrn_prelu_forward(&prelu, buf2, 1, 16, n_frames * 65);
        printf("  C PReLU output sum: %.6f (expected 1000.50)\n", tensor_sum(buf2, out_size));
    }

    /* Cleanup */
    free(buf1);
    free(buf2);
    free(scratch);
    free(spec_real);
    free(spec_imag);
    gtcrn_destroy(model);

    return 0;
}
