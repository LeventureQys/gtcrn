/// <file>debug_tra.c</file>
/// <summary>逐步调试TRA(时序循环注意力)</summary>
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
    printf("%s: C=%.6f, Py=%.6f, max_diff=%.6f [%s]\n", name, c_sum, py_sum, max_diff, status);
    if (max_diff >= 0.001) {
        printf("  C[%d]=%.6f, Py[%d]=%.6f\n", max_idx, c_data[max_idx], max_idx, py_data[max_idx]);
    }
}

int main(int argc, char** argv) {
    const char* weights_path = "weights/gtcrn_weights.bin";
    const char* test_dir = "test_data";

    printf("=== TRA (Temporal Recurrent Attention) Debug ===\n\n");

    gtcrn_t* model = gtcrn_create();
    if (!model || gtcrn_load_weights(model, weights_path) != GTCRN_OK) {
        printf("Failed to load model\n");
        return 1;
    }
    gtcrn_weights_t* w = model->weights;

    int batch = 1, channels = 8, time = 10, freq = 33;
    int gru_hidden = channels * 2;  /* 16 */
    char path[256];

    /* Load BN3 output as input to TRA */
    snprintf(path, sizeof(path), "%s/py_gt2_bn3.bin", test_dir);
    gtcrn_float* bn3 = load_tensor(path, channels * time * freq);
    if (!bn3) { printf("无法加载bn3\n"); return 1; }
    printf("Input (bn3): sum=%.6f\n\n", tensor_sum(bn3, channels * time * freq));

    /* Allocate workspace */
    size_t ws_size = 32 * 16 * time * freq;
    gtcrn_float* workspace = (gtcrn_float*)calloc(ws_size, sizeof(gtcrn_float));

    /* Step 1: Compute zt = mean(x^2, dim=-1): (B, C, T) */
    gtcrn_float* zt = workspace;
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                gtcrn_float sum = 0.0f;
                for (int f = 0; f < freq; f++) {
                    gtcrn_float val = bn3[GTCRN_IDX4(b, c, t, f, channels, time, freq)];
                    sum += val * val;
                }
                zt[(b * channels + c) * time + t] = sum / freq;
            }
        }
    }

    snprintf(path, sizeof(path), "%s/py_tra_zt.bin", test_dir);
    gtcrn_float* py_zt = load_tensor(path, channels * time);
    if (py_zt) compare_tensors("zt", zt, py_zt, channels * time, 90.863060);

    /* Debug: print first timestep input */
    printf("\nFirst timestep input (t=0):\n");
    for (int c = 0; c < channels; c++) {
        printf("  zt[%d,0] = %.6f\n", c, zt[c * time + 0]);
    }

    /* Debug: print first few weight values */
    printf("\nWeight W_ih first 5x3:\n");
    for (int i = 0; i < 5; i++) {
        printf("  [%d]: ", i);
        for (int j = 0; j < 3; j++) {
            printf("%.6f ", w->en_gt2_tra_gru_ih[i * channels + j]);
        }
        printf("\n");
    }

    /* Step 2: GRU forward */
    gtcrn_float* gru_out = zt + batch * channels * time;
    gtcrn_float* h_prev = gru_out + batch * time * gru_hidden;
    gtcrn_float* h_curr = h_prev + gru_hidden;
    gtcrn_float* gru_workspace = h_curr + gru_hidden;

    gtcrn_vec_zero(h_prev, gru_hidden);

    const gtcrn_float* tra_gru_ih = w->en_gt2_tra_gru_ih;
    const gtcrn_float* tra_gru_hh = w->en_gt2_tra_gru_hh;
    const gtcrn_float* tra_gru_bih = w->en_gt2_tra_gru_bih;
    const gtcrn_float* tra_gru_bhh = w->en_gt2_tra_gru_bhh;

    for (int t = 0; t < time; t++) {
        /* Prepare input for this timestep: transpose (B, C, T) to (B, T, C)
         * x_t must not overlap with gru_out! Place it after gru_workspace.
         */
        gtcrn_float* x_t = gru_workspace + 3 * gru_hidden * 2;  /* After gates buffers */
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                x_t[b * channels + c] = zt[(b * channels + c) * time + t];
            }
        }

        /* GRU cell for each batch */
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

            /* Debug first timestep */
            if (t == 0) {
                printf("\nTimestep 0 GRU debug:\n");
                printf("  gates_ih[:5]: ");
                for (int i = 0; i < 5; i++) printf("%.6f ", gates_ih[i]);
                printf("\n");
                printf("  gates_hh[:5]: ");
                for (int i = 0; i < 5; i++) printf("%.6f ", gates_hh[i]);
                printf("\n");
                printf("  h_new[:5]: ");
                for (int i = 0; i < 5; i++) printf("%.6f ", h_curr[i]);
                printf("\n");
            }
            /* Debug all timesteps h values */
            printf("t=%d: h[0:3]= %.4f %.4f %.4f, sum=%.6f\n", t,
                   h_curr[0], h_curr[1], h_curr[2], tensor_sum(h_curr, gru_hidden));
            /* Check gru_out storage */
            if (t == 5) {
                printf("  gru_out at t=5: [%d]->%.6f, [%d]->%.6f, [%d]->%.6f\n",
                       5*16+0, gru_out[5*16+0],
                       5*16+1, gru_out[5*16+1],
                       5*16+2, gru_out[5*16+2]);
            }

            /* Swap h_prev and h_curr */
            gtcrn_float* tmp = h_prev;
            h_prev = h_curr;
            h_curr = tmp;
        }
    }

    snprintf(path, sizeof(path), "%s/py_tra_gru_out.bin", test_dir);
    gtcrn_float* py_gru = load_tensor(path, time * gru_hidden);

    /* Debug gru_out before compare */
    printf("\nBefore compare: gru_out[82] = %.6f\n", gru_out[82]);

    if (py_gru) compare_tensors("gru_out", gru_out, py_gru, time * gru_hidden, -1.620263);

    /* Step 3: FC layer + Sigmoid */
    const gtcrn_float* tra_fc_weight = w->en_gt2_tra_fc_weight;
    const gtcrn_float* tra_fc_bias = w->en_gt2_tra_fc_bias;

    gtcrn_float* at = workspace;
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < time; t++) {
            for (int c = 0; c < channels; c++) {
                gtcrn_float sum = tra_fc_bias[c];
                for (int j = 0; j < gru_hidden; j++) {
                    sum += tra_fc_weight[c * gru_hidden + j] * gru_out[(b * time + t) * gru_hidden + j];
                }
                /* Store in (B, C, T) layout to match PyTorch after transpose */
                at[(b * channels + c) * time + t] = gtcrn_sigmoid(sum);
            }
        }
    }

    snprintf(path, sizeof(path), "%s/py_tra_at.bin", test_dir);
    gtcrn_float* py_at = load_tensor(path, channels * time);
    if (py_at) compare_tensors("at (after sigmoid)", at, py_at, channels * time, 30.552059);

    /* Step 4: Apply attention */
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int t = 0; t < time; t++) {
                gtcrn_float alpha = at[(b * channels + c) * time + t];
                for (int f = 0; f < freq; f++) {
                    bn3[GTCRN_IDX4(b, c, t, f, channels, time, freq)] *= alpha;
                }
            }
        }
    }

    snprintf(path, sizeof(path), "%s/py_gt2_tra.bin", test_dir);
    gtcrn_float* py_tra = load_tensor(path, channels * time * freq);
    if (py_tra) compare_tensors("TRA output", bn3, py_tra, channels * time * freq, 815.181396);

    /* Cleanup */
    free(workspace);
    free(bn3);
    if (py_zt) free(py_zt);
    if (py_gru) free(py_gru);
    if (py_at) free(py_at);
    if (py_tra) free(py_tra);
    gtcrn_destroy(model);

    return 0;
}
