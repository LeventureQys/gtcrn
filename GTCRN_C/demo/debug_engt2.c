/// <file>debug_engt2.c</file>
/// <summary>Debug EnGT2 intermediate values for comparison with Python</summary>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gtcrn_model.h"
#include "wav_io.h"

#define FRAME_SIZE 256
#define N_FREQ 257
#define WIN_SIZE 512
#define FREQ_DOWN 33

static double abs_sum(const float* buf, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += fabs(buf[i]);
    }
    return sum;
}

static void print_first_n(const char* name, const float* buf, int n) {
    printf("  %s first %d vals: [", name, n);
    for (int i = 0; i < n; i++) {
        printf("%.6f", buf[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    if (argc != 3) {
        printf("Usage: %s <weights_file> <input_wav>\n", argv[0]);
        return 1;
    }

    const char* weights_path = argv[1];
    const char* input_path = argv[2];

    printf("=== Debug EnGT2 Detail (C) ===\n\n");

    /* Create model */
    gtcrn_t* model = gtcrn_create();
    if (!model) {
        fprintf(stderr, "Error: Failed to create model\n");
        return 1;
    }

    /* Load weights */
    if (gtcrn_load_weights(model, weights_path) != GTCRN_OK) {
        fprintf(stderr, "Error: Failed to load weights\n");
        return 1;
    }
    printf("Weights loaded\n");

    /* Read audio */
    wav_info_t wav_info;
    float* audio = NULL;
    int num_samples = wav_read(input_path, &wav_info, &audio);
    if (num_samples <= 0) {
        fprintf(stderr, "Error: Failed to read audio\n");
        return 1;
    }
    printf("Audio: %d samples\n\n", num_samples);

    /* Reset state */
    gtcrn_reset_state(model);

    /* Process frames 0-5 (frame_count 1-6) to match Python frame 5 */
    float output_frame[FRAME_SIZE];
    int target_c_frame = 6;  /* C frame_count = 6 matches Python frame 5 */

    printf("Processing frames 1 to %d...\n", target_c_frame);

    for (int frame = 0; frame < target_c_frame; frame++) {
        float* input_frame = audio + frame * FRAME_SIZE;
        gtcrn_process_frame(model, input_frame, output_frame);

        if (frame == target_c_frame - 1) {
            printf("\n=== C Frame %d (Python frame %d) ===\n\n", target_c_frame, target_c_frame - 1);

            /* Access internal buffers from model state via pointer */
            gtcrn_state_t* s = model->state;

            /* Print EnConv0 output (16, 65) - stored in en_out0 */
            printf("EnConv0 output abs_sum: %.6f\n", abs_sum(s->en_out0, 16 * 65));

            /* Print EnConv1 output (16, 33) - stored in en_out1 */
            printf("EnConv1 output abs_sum: %.6f\n", abs_sum(s->en_out1, 16 * FREQ_DOWN));

            /* Print EnGT2 output (16, 33) - stored in en_out2 */
            printf("EnGT2 output abs_sum: %.6f\n", abs_sum(s->en_out2, 16 * FREQ_DOWN));

            /* Print EnGT3 output (16, 33) - stored in en_out3 */
            printf("EnGT3 output abs_sum: %.6f\n", abs_sum(s->en_out3, 16 * FREQ_DOWN));

            /* Print per-channel sums for EnGT2 */
            printf("\n--- EnGT2 per-channel sums ---\n");
            for (int c = 0; c < 16; c++) {
                double ch_sum = 0;
                for (int f = 0; f < FREQ_DOWN; f++) {
                    ch_sum += s->en_out2[c * FREQ_DOWN + f];
                }
                printf("  ch%d: %.6f\n", c, ch_sum);
            }

            /* Print first 10 values */
            print_first_n("EnConv1 output", s->en_out1, 10);
            print_first_n("EnGT2 output", s->en_out2, 10);
        }
    }

    printf("\n=== Comparison with Python ===\n");
    printf("Python Frame 5 (C frame_count=6):\n");
    printf("  EnConv0 abs_sum: 151.607483 (Python)\n");
    printf("  EnConv1 abs_sum: 112.684692 (Python)\n");
    printf("  EnGT2 output abs_sum: 142.484085 (Python)\n");
    printf("\n  EnGT2 per-channel sums (Python):\n");
    printf("    ch0: 16.708656\n");
    printf("    ch1: 26.233351\n");
    printf("    ch2: 7.766028\n");
    printf("    ch3: 12.498169\n");
    printf("    ch4: 7.187007\n");
    printf("    ch5: -0.700970\n");
    printf("    ch6: -6.701913\n");
    printf("    ch7: 4.948688\n");
    printf("    ch8: 13.060825\n");
    printf("    ch9: 6.094824\n");
    printf("    ch10: 13.823698\n");
    printf("    ch11: 0.397370\n");
    printf("    ch12: -4.660911\n");
    printf("    ch13: 2.710830\n");
    printf("    ch14: 11.217967\n");
    printf("    ch15: 4.166672\n");

    free(audio);
    gtcrn_destroy(model);

    return 0;
}
