/// <file>test_istft_standalone.c</file>
/// <summary>Test C ISTFT with known spectrum values</summary>

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

    printf("=== ISTFT Standalone Test ===\n\n");

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

    /* Test: STFT -> ISTFT should approximately recover the original signal */
    /* For sqrt-Hann window with 50% overlap, this should be perfect at steady state */

    float* stft_window = (float*)malloc(WIN_SIZE * sizeof(float));
    float* spec_real = (float*)malloc(N_FREQ * sizeof(float));
    float* spec_imag = (float*)malloc(N_FREQ * sizeof(float));
    float* istft_frame = (float*)malloc(WIN_SIZE * sizeof(float));

    int frame_idx = 100;
    memcpy(stft_window, audio + 99 * FRAME_SIZE, FRAME_SIZE * sizeof(float));
    memcpy(stft_window + FRAME_SIZE, audio + 100 * FRAME_SIZE, FRAME_SIZE * sizeof(float));

    /* STFT */
    gtcrn_stft_frame(model->stft, stft_window, spec_real, spec_imag);

    double in_mag_sum = 0;
    for (int f = 0; f < N_FREQ; f++) {
        in_mag_sum += sqrt(spec_real[f] * spec_real[f] + spec_imag[f] * spec_imag[f]);
    }
    printf("Input spectrum magnitude sum: %.4f\n", in_mag_sum);

    /* ISTFT (without going through neural network) */
    gtcrn_istft_frame(model->stft, spec_real, spec_imag, istft_frame);

    /* Compare with original windowed signal */
    /* The ISTFT output should be: original_windowed * window */
    printf("\n=== STFT->ISTFT reconstruction test ===\n");

    /* Compute original windowed input */
    float* window = model->stft->window;
    float* original_windowed = (float*)malloc(WIN_SIZE * sizeof(float));
    for (int i = 0; i < WIN_SIZE; i++) {
        original_windowed[i] = stft_window[i] * window[i];
    }

    /* Compare */
    double original_energy = 0, istft_energy = 0, diff_energy = 0;
    for (int i = 0; i < WIN_SIZE; i++) {
        original_energy += original_windowed[i] * original_windowed[i];
        istft_energy += istft_frame[i] * istft_frame[i];
        double diff = istft_frame[i] - original_windowed[i];
        diff_energy += diff * diff;
    }

    printf("Original windowed RMS: %.6f\n", sqrt(original_energy / WIN_SIZE));
    printf("ISTFT output RMS:      %.6f\n", sqrt(istft_energy / WIN_SIZE));
    printf("Difference RMS:        %.6f\n", sqrt(diff_energy / WIN_SIZE));
    printf("Energy ratio:          %.4f\n", sqrt(istft_energy / original_energy));

    /* Sample comparison */
    printf("\n=== First 10 samples ===\n");
    printf("Idx | Original*Win | ISTFT Output | Diff\n");
    printf("----+--------------+--------------+--------\n");
    for (int i = 0; i < 10; i++) {
        printf("%3d | %12.6f | %12.6f | %.6f\n",
               i, original_windowed[i], istft_frame[i],
               istft_frame[i] - original_windowed[i]);
    }

    /* Clean up */
    free(stft_window);
    free(spec_real);
    free(spec_imag);
    free(istft_frame);
    free(original_windowed);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
