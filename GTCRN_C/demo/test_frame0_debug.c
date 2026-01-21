/// <file>test_frame0_debug.c</file>
/// <summary>Debug frame 0 output comparison with Python</summary>

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

    printf("=== Frame 0 Debug Test ===\n\n");

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

    /* Process frame 0 */
    float* output_frame = (float*)malloc(FRAME_SIZE * sizeof(float));
    gtcrn_process_frame(model, audio, output_frame);

    /* Access internal spectrum buffers */
    float* workspace = model->workspace;
    float* spec_real = workspace + WIN_SIZE;
    float* spec_imag = spec_real + N_FREQ;
    float* out_spec_real = spec_imag + N_FREQ;
    float* out_spec_imag = out_spec_real + N_FREQ;

    /* Input spectrum stats */
    double in_real_sum = 0.0, in_imag_sum = 0.0, in_mag_sum = 0.0;
    for (int f = 0; f < N_FREQ; f++) {
        in_real_sum += spec_real[f];
        in_imag_sum += spec_imag[f];
        in_mag_sum += sqrt(spec_real[f] * spec_real[f] + spec_imag[f] * spec_imag[f]);
    }

    printf("Input spectrum (C):\n");
    printf("  Real sum: %.4f (Python: -0.7160)\n", in_real_sum);
    printf("  Imag sum: %.4f (Python: -0.0000)\n", in_imag_sum);
    printf("  Mag sum:  %.4f (Python: 56.4236)\n\n", in_mag_sum);

    /* Output spectrum stats */
    double out_real_sum = 0.0, out_imag_sum = 0.0, out_mag_sum = 0.0;
    double max_mag = 0.0;
    for (int f = 0; f < N_FREQ; f++) {
        out_real_sum += out_spec_real[f];
        out_imag_sum += out_spec_imag[f];
        double mag = sqrt(out_spec_real[f] * out_spec_real[f] +
                          out_spec_imag[f] * out_spec_imag[f]);
        out_mag_sum += mag;
        if (mag > max_mag) max_mag = mag;
    }

    printf("Output spectrum (C):\n");
    printf("  Real sum: %.4f (Python: 0.7156)\n", out_real_sum);
    printf("  Imag sum: %.4f (Python: -0.0196)\n", out_imag_sum);
    printf("  Mag sum:  %.4f (Python: 9.4084)\n", out_mag_sum);
    printf("  Max mag:  %.4f (Python: 1.1535)\n\n", max_mag);

    printf("First 10 output spectrum values (C):\n");
    printf("Bin | Real       | Imag       | Mag\n");
    printf("----+------------+------------+--------\n");

    /* Python values for comparison:
       0 |  -0.120726 |  -0.001620 | 0.1207
       1 |   0.033122 |  -0.015869 | 0.0367
       2 |   1.153475 |   0.006386 | 1.1535
       3 |  -0.700272 |   0.008012 | 0.7003
       4 |   0.158315 |  -0.002256 | 0.1583
       5 |   0.090616 |  -0.001226 | 0.0906
       6 |   0.048720 |  -0.000074 | 0.0487
       7 |  -0.162887 |  -0.000467 | 0.1629
       8 |  -0.024102 |  -0.000107 | 0.0241
       9 |   0.028414 |   0.000167 | 0.0284
    */
    for (int i = 0; i < 10; i++) {
        double mag = sqrt(out_spec_real[i] * out_spec_real[i] +
                          out_spec_imag[i] * out_spec_imag[i]);
        printf("%3d | %10.6f | %10.6f | %.4f\n",
               i, out_spec_real[i], out_spec_imag[i], mag);
    }

    printf("\nOutput time-domain frame (first 10 samples):\n");
    for (int i = 0; i < 10; i++) {
        printf("  [%d] = %.6f\n", i, output_frame[i]);
    }

    free(output_frame);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
