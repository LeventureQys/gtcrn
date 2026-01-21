/// <file>debug_gtconv.c</file>
/// <summary>逐步调试GTConvBlock</summary>
/// <author>李文轩</author>

#include "gtcrn_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double tensor_sum(const gtcrn_float* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) sum += data[i];
    return sum;
}

static gtcrn_float* load_tensor(const char* filepath, int expected_size) {
    FILE* f = fopen(filepath, "rb");
    if (!f) return NULL;
    gtcrn_float* data = (gtcrn_float*)malloc(expected_size * sizeof(gtcrn_float));
    size_t read = fread(data, sizeof(gtcrn_float), expected_size, f);
    fclose(f);
    if (read != (size_t)expected_size) { free(data); return NULL; }
    return data;
}

static void compare_tensors(const char* name, const gtcrn_float* c_data,
                            const gtcrn_float* py_data, int size, double py_sum) {
    double c_sum = tensor_sum(c_data, size);
    double max_diff = 0.0;
    int max_idx = 0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(c_data[i] - py_data[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
    }
    const char* status = max_diff < 0.001 ? "PASS" : (max_diff < 0.1 ? "WARN" : "FAIL");
    printf("%s: C=%.4f, Py=%.4f, max_diff=%.6f [%s]\n", name, c_sum, py_sum, max_diff, status);
    if (max_diff >= 0.001) {
        printf("  C[%d]=%.6f, Py[%d]=%.6f\n", max_idx, c_data[max_idx], max_idx, py_data[max_idx]);
    }
}

/* SFE forward */
static void sfe_forward(const gtcrn_float* input, gtcrn_float* output,
                        int batch, int channels, int time, int freq) {
    int out_ch = channels * 3;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                for (int f = 0; f < freq; f++) {
                    gtcrn_float v_left = (f > 0) ? input[GTCRN_IDX4(b, c, t, f-1, channels, time, freq)] : 0.0f;
                    gtcrn_float v_center = input[GTCRN_IDX4(b, c, t, f, channels, time, freq)];
                    gtcrn_float v_right = (f < freq-1) ? input[GTCRN_IDX4(b, c, t, f+1, channels, time, freq)] : 0.0f;
                    output[GTCRN_IDX4(b, c*3+0, t, f, out_ch, time, freq)] = v_left;
                    output[GTCRN_IDX4(b, c*3+1, t, f, out_ch, time, freq)] = v_center;
                    output[GTCRN_IDX4(b, c*3+2, t, f, out_ch, time, freq)] = v_right;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    const char* weights_path = "weights/gtcrn_weights.bin";
    const char* test_dir = "test_data";

    printf("=== GTConvBlock 2 Step-by-Step Debug ===\n\n");

    gtcrn_t* model = gtcrn_create();
    if (!model || gtcrn_load_weights(model, weights_path) != GTCRN_OK) {
        printf("Failed to load model\n");
        return 1;
    }
    gtcrn_weights_t* w = model->weights;

    int batch = 1, time = 10, freq = 33;
    char path[256];

    /* Load EnConv1 output as input */
    snprintf(path, sizeof(path), "%s/py_en_conv1.bin", test_dir);
    gtcrn_float* en1_out = load_tensor(path, 16 * time * freq);
    if (!en1_out) { printf("无法加载en_conv1\n"); return 1; }
    printf("Input en_conv1: sum=%.6f\n\n", tensor_sum(en1_out, 16 * time * freq));

    /* Allocate buffers */
    size_t buf_size = 32 * 16 * time * freq;
    gtcrn_float* workspace = (gtcrn_float*)calloc(buf_size, sizeof(gtcrn_float));

    int half_ch = 8;

    /* Step 1: Split channels */
    gtcrn_float* x1 = workspace;
    gtcrn_float* x2 = x1 + half_ch * time * freq;

    for (int c = 0; c < half_ch; c++) {
        for (int t = 0; t < time; t++) {
            for (int f = 0; f < freq; f++) {
                x1[GTCRN_IDX4(0, c, t, f, half_ch, time, freq)] =
                    en1_out[GTCRN_IDX4(0, c, t, f, 16, time, freq)];
                x2[GTCRN_IDX4(0, c, t, f, half_ch, time, freq)] =
                    en1_out[GTCRN_IDX4(0, c + half_ch, t, f, 16, time, freq)];
            }
        }
    }

    snprintf(path, sizeof(path), "%s/py_gt2_x1.bin", test_dir);
    gtcrn_float* py_x1 = load_tensor(path, half_ch * time * freq);
    snprintf(path, sizeof(path), "%s/py_gt2_x2.bin", test_dir);
    gtcrn_float* py_x2 = load_tensor(path, half_ch * time * freq);
    if (py_x1) compare_tensors("x1", x1, py_x1, half_ch * time * freq, 166.48);
    if (py_x2) compare_tensors("x2", x2, py_x2, half_ch * time * freq, 438.72);

    /* Step 2: SFE on x1 */
    gtcrn_float* x1_sfe = x2 + half_ch * time * freq;
    sfe_forward(x1, x1_sfe, batch, half_ch, time, freq);

    snprintf(path, sizeof(path), "%s/py_gt2_x1_sfe.bin", test_dir);
    gtcrn_float* py_x1_sfe = load_tensor(path, 24 * time * freq);
    if (py_x1_sfe) compare_tensors("x1_sfe", x1_sfe, py_x1_sfe, 24 * time * freq, 489.69);

    /* Step 3: Point conv 1 */
    gtcrn_float* h1 = x1;
    gtcrn_conv2d_t pc1 = {
        .weight = w->en_gt2_pc1_weight, .bias = w->en_gt2_pc1_bias,
        .in_channels = 24, .out_channels = 16,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1
    };
    gtcrn_conv2d_forward(&pc1, x1_sfe, h1, batch, time, freq, time, freq);

    snprintf(path, sizeof(path), "%s/py_gt2_pc1.bin", test_dir);
    gtcrn_float* py_pc1 = load_tensor(path, 16 * time * freq);
    if (py_pc1) compare_tensors("pc1", h1, py_pc1, 16 * time * freq, -769.81);

    /* Step 4: BN1 */
    gtcrn_batchnorm2d_t bn1 = {
        .gamma = w->en_gt2_bn1_gamma, .beta = w->en_gt2_bn1_beta,
        .running_mean = w->en_gt2_bn1_mean, .running_var = w->en_gt2_bn1_var,
        .num_features = 16, .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn1, h1, batch, time, freq);

    snprintf(path, sizeof(path), "%s/py_gt2_bn1.bin", test_dir);
    gtcrn_float* py_bn1 = load_tensor(path, 16 * time * freq);
    if (py_bn1) compare_tensors("bn1", h1, py_bn1, 16 * time * freq, 579.73);

    /* Step 5: PReLU1 */
    gtcrn_prelu_t prelu1 = { .alpha = w->en_gt2_prelu1, .num_parameters = 1 };
    gtcrn_prelu_forward(&prelu1, h1, batch, 16, time * freq);

    snprintf(path, sizeof(path), "%s/py_gt2_prelu1.bin", test_dir);
    gtcrn_float* py_prelu1 = load_tensor(path, 16 * time * freq);
    if (py_prelu1) compare_tensors("prelu1", h1, py_prelu1, 16 * time * freq, 1695.99);

    /* Step 6: Pad */
    int dilation = 1;
    int pad_t = (3 - 1) * dilation;  /* = 2 */
    int padded_time = time + pad_t;  /* = 12 */
    gtcrn_float* h1_padded = x1_sfe;
    gtcrn_vec_zero(h1_padded, batch * 16 * padded_time * freq);

    for (int c = 0; c < 16; c++) {
        for (int t = 0; t < time; t++) {
            for (int f = 0; f < freq; f++) {
                h1_padded[GTCRN_IDX4(0, c, t + pad_t, f, 16, padded_time, freq)] =
                    h1[GTCRN_IDX4(0, c, t, f, 16, time, freq)];
            }
        }
    }

    snprintf(path, sizeof(path), "%s/py_gt2_padded.bin", test_dir);
    gtcrn_float* py_padded = load_tensor(path, 16 * padded_time * freq);
    if (py_padded) compare_tensors("padded", h1_padded, py_padded, 16 * padded_time * freq, 1695.99);

    /* Step 7: Depth conv */
    gtcrn_float* dc_out = h1;
    gtcrn_conv2d_t dc = {
        .weight = w->en_gt2_dc_weight, .bias = w->en_gt2_dc_bias,
        .in_channels = 16, .out_channels = 16,
        .kernel_h = 3, .kernel_w = 3,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 1,
        .dilation_h = dilation, .dilation_w = 1,
        .groups = 16
    };
    gtcrn_conv2d_forward(&dc, h1_padded, dc_out, batch, padded_time, freq, time, freq);

    snprintf(path, sizeof(path), "%s/py_gt2_dc.bin", test_dir);
    gtcrn_float* py_dc = load_tensor(path, 16 * time * freq);
    if (py_dc) compare_tensors("dc", dc_out, py_dc, 16 * time * freq, 1298.13);

    /* Step 8: BN2 */
    gtcrn_batchnorm2d_t bn2 = {
        .gamma = w->en_gt2_bn2_gamma, .beta = w->en_gt2_bn2_beta,
        .running_mean = w->en_gt2_bn2_mean, .running_var = w->en_gt2_bn2_var,
        .num_features = 16, .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn2, dc_out, batch, time, freq);

    snprintf(path, sizeof(path), "%s/py_gt2_bn2.bin", test_dir);
    gtcrn_float* py_bn2 = load_tensor(path, 16 * time * freq);
    if (py_bn2) compare_tensors("bn2", dc_out, py_bn2, 16 * time * freq, 1193.05);

    /* Step 9: PReLU2 */
    gtcrn_prelu_t prelu2 = { .alpha = w->en_gt2_prelu2, .num_parameters = 1 };
    gtcrn_prelu_forward(&prelu2, dc_out, batch, 16, time * freq);

    snprintf(path, sizeof(path), "%s/py_gt2_prelu2.bin", test_dir);
    gtcrn_float* py_prelu2 = load_tensor(path, 16 * time * freq);
    if (py_prelu2) compare_tensors("prelu2", dc_out, py_prelu2, 16 * time * freq, 2434.24);

    /* Step 10: Point conv 2 */
    gtcrn_float* h1_out = x1_sfe;
    gtcrn_conv2d_t pc2 = {
        .weight = w->en_gt2_pc2_weight, .bias = w->en_gt2_pc2_bias,
        .in_channels = 16, .out_channels = 8,
        .kernel_h = 1, .kernel_w = 1,
        .stride_h = 1, .stride_w = 1,
        .padding_h = 0, .padding_w = 0,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1
    };
    gtcrn_conv2d_forward(&pc2, dc_out, h1_out, batch, time, freq, time, freq);

    snprintf(path, sizeof(path), "%s/py_gt2_pc2.bin", test_dir);
    gtcrn_float* py_pc2 = load_tensor(path, 8 * time * freq);
    if (py_pc2) compare_tensors("pc2", h1_out, py_pc2, 8 * time * freq, -560.05);

    /* Step 11: BN3 */
    gtcrn_batchnorm2d_t bn3 = {
        .gamma = w->en_gt2_bn3_gamma, .beta = w->en_gt2_bn3_beta,
        .running_mean = w->en_gt2_bn3_mean, .running_var = w->en_gt2_bn3_var,
        .num_features = 8, .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn3, h1_out, batch, time, freq);

    snprintf(path, sizeof(path), "%s/py_gt2_bn3.bin", test_dir);
    gtcrn_float* py_bn3 = load_tensor(path, 8 * time * freq);
    if (py_bn3) compare_tensors("bn3", h1_out, py_bn3, 8 * time * freq, 1993.62);

    /* Note: We skip TRA for now as it involves GRU */
    printf("\nSkipping TRA (GRU-based) for this debug.\n");

    /* Cleanup */
    free(en1_out);
    free(workspace);
    if (py_x1) free(py_x1);
    if (py_x2) free(py_x2);
    if (py_x1_sfe) free(py_x1_sfe);
    if (py_pc1) free(py_pc1);
    if (py_bn1) free(py_bn1);
    if (py_prelu1) free(py_prelu1);
    if (py_padded) free(py_padded);
    if (py_dc) free(py_dc);
    if (py_bn2) free(py_bn2);
    if (py_prelu2) free(py_prelu2);
    if (py_pc2) free(py_pc2);
    if (py_bn3) free(py_bn3);
    gtcrn_destroy(model);

    return 0;
}
