/// <file>test_stream_spectrum.c</file>
/// <summary>Test C streaming output spectrum vs Python</summary>

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

    printf("=== C Stream Spectrum Test ===\n\n");

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

    /* Access internal spectrum buffers */
    float* workspace = model->workspace;
    float* spec_real = workspace + WIN_SIZE;
    float* spec_imag = spec_real + N_FREQ;
    float* out_spec_real = spec_imag + N_FREQ;
    float* out_spec_imag = out_spec_real + N_FREQ;

    /* Process first 11 frames */
    printf("Output spectrum stats per frame:\n");
    printf("Frame | In Real sum | In Imag sum | Out Real    | Out Imag    | Out Mag\n");
    printf("--------------------------------------------------------------------------\n");

    float* output_frame = (float*)malloc(FRAME_SIZE * sizeof(float));

    for (int frame = 0; frame < 11; frame++) {
        float* input_frame = audio + frame * FRAME_SIZE;
        gtcrn_process_frame(model, input_frame, output_frame);

        /* Calculate input spectrum stats */
        double in_real_sum = 0.0, in_imag_sum = 0.0;
        for (int f = 0; f < N_FREQ; f++) {
            in_real_sum += spec_real[f];
            in_imag_sum += spec_imag[f];
        }

        /* Calculate output stats */
        double real_sum = 0.0, imag_sum = 0.0, mag_sum = 0.0;
        for (int f = 0; f < N_FREQ; f++) {
            real_sum += out_spec_real[f];
            imag_sum += out_spec_imag[f];
            mag_sum += sqrt(out_spec_real[f] * out_spec_real[f] +
                           out_spec_imag[f] * out_spec_imag[f]);
        }

        printf("%5d | %11.4f | %11.4f | %11.4f | %11.4f | %9.4f\n",
               frame, in_real_sum, in_imag_sum, real_sum, imag_sum, mag_sum);

        if (frame == 10) {
            printf("\nFrame 10 first 5 output real: ");
            for (int i = 0; i < 5; i++) {
                printf("%.6f ", out_spec_real[i]);
            }
            printf("\n");
        }
    }

    free(output_frame);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
