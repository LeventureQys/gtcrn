/// <file>debug_enconv0.c</file>
/// <summary>调试EnConv0,与PyTorch对比</summary>
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

    if (max_diff >= 0.001) {
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

int main(int argc, char** argv) {
    const char* weights_path = "weights/gtcrn_weights.bin";
    const char* test_dir = "test_data";

    printf("=== EnConv0 Layer Debug ===\n\n");

    /* Create and load model */
    gtcrn_t* model = gtcrn_create();
    if (!model || gtcrn_load_weights(model, weights_path) != GTCRN_OK) {
        printf("Failed to load model\n");
        return 1;
    }

    gtcrn_weights_t* w = model->weights;

    /* Print weight sums */
    printf("C Weight sums:\n");
    printf("  en_conv0_weight sum: %.6f\n", tensor_sum(w->en_conv0_weight, 16 * 9 * 1 * 5));
    printf("  en_conv0_bias sum: %.6f\n", tensor_sum(w->en_conv0_bias, 16));
    printf("  en_bn0_gamma sum: %.6f\n", tensor_sum(w->en_bn0_gamma, 16));
    printf("  en_bn0_beta sum: %.6f\n", tensor_sum(w->en_bn0_beta, 16));
    printf("  en_bn0_mean sum: %.6f\n", tensor_sum(w->en_bn0_mean, 16));
    printf("  en_bn0_var sum: %.6f\n", tensor_sum(w->en_bn0_var, 16));
    printf("  en_prelu0 value: %.6f\n", w->en_prelu0[0]);

    /* Load SFE input */
    char path[256];
    snprintf(path, sizeof(path), "%s/py_feat_sfe.bin", test_dir);
    gtcrn_float* feat_sfe = load_tensor(path, 9 * 10 * 129);

    if (!feat_sfe) {
        printf("无法加载feat_sfe。请先运行export_all_tensors.py。\n");
        gtcrn_destroy(model);
        return 1;
    }

    printf("\nInput feat_sfe sum: %.6f\n", tensor_sum(feat_sfe, 9 * 10 * 129));

    /* Allocate output buffers */
    int out_size = 16 * 10 * 65;
    gtcrn_float* conv_out = (gtcrn_float*)calloc(out_size, sizeof(gtcrn_float));
    gtcrn_float* bn_out = (gtcrn_float*)calloc(out_size, sizeof(gtcrn_float));

    /* Step 1: Conv2d only */
    printf("\n=== Step 1: Conv2d ===\n");
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

    gtcrn_conv2d_forward(&conv, feat_sfe, conv_out, 1, 10, 129, 10, 65);
    printf("C Conv output sum: %.6f\n", tensor_sum(conv_out, out_size));
    printf("Expected PyTorch: -9600.519531\n");

    /* Compare with PyTorch conv output */
    snprintf(path, sizeof(path), "%s/py_enconv0_conv.bin", test_dir);
    gtcrn_float* py_conv = load_tensor(path, out_size);
    if (py_conv) {
        compare_tensors("Conv", conv_out, py_conv, out_size);
        free(py_conv);
    }

    /* Step 2: BatchNorm */
    printf("\n=== Step 2: BatchNorm ===\n");
    memcpy(bn_out, conv_out, out_size * sizeof(gtcrn_float));

    gtcrn_batchnorm2d_t bn = {
        .gamma = w->en_bn0_gamma,
        .beta = w->en_bn0_beta,
        .running_mean = w->en_bn0_mean,
        .running_var = w->en_bn0_var,
        .num_features = 16,
        .eps = 1e-5f
    };
    gtcrn_batchnorm2d_forward(&bn, bn_out, 1, 10, 65);
    printf("C BN output sum: %.6f\n", tensor_sum(bn_out, out_size));
    printf("Expected PyTorch: -614.833984\n");

    /* Compare with PyTorch BN output */
    snprintf(path, sizeof(path), "%s/py_enconv0_bn.bin", test_dir);
    gtcrn_float* py_bn = load_tensor(path, out_size);
    if (py_bn) {
        compare_tensors("BN", bn_out, py_bn, out_size);
        free(py_bn);
    }

    /* Step 3: PReLU */
    printf("\n=== Step 3: PReLU ===\n");
    gtcrn_prelu_t prelu = {
        .alpha = w->en_prelu0,
        .num_parameters = 1  /* Shared alpha */
    };
    gtcrn_prelu_forward(&prelu, bn_out, 1, 16, 10 * 65);
    printf("C PReLU output sum: %.6f\n", tensor_sum(bn_out, out_size));
    printf("Expected PyTorch: 1000.502075\n");

    /* Compare with PyTorch PReLU output */
    snprintf(path, sizeof(path), "%s/py_enconv0_prelu.bin", test_dir);
    gtcrn_float* py_prelu = load_tensor(path, out_size);
    if (py_prelu) {
        compare_tensors("PReLU", bn_out, py_prelu, out_size);
        free(py_prelu);
    }

    /* Cleanup */
    free(feat_sfe);
    free(conv_out);
    free(bn_out);
    gtcrn_destroy(model);

    return 0;
}
