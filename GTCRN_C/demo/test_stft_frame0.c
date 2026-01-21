/// <file>test_stft_frame0.c</file>
/// <summary>Compare STFT frame 0 output between C and Python</summary>

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
#define WIN_SIZE 512
#define N_FREQ 257

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    if (argc != 2) {
        printf("Usage: %s <input_wav>\n", argv[0]);
        return 1;
    }

    const char* input_path = argv[1];

    printf("=== STFT Frame 0 Test ===\n\n");

    /* Read audio */
    wav_info_t wav_info;
    float* audio = NULL;
    int num_samples = wav_read(input_path, &wav_info, &audio);
    if (num_samples <= 0) {
        fprintf(stderr, "Error: Failed to read audio\n");
        return 1;
    }
    printf("Audio: %d samples\n\n", num_samples);

    /* Create STFT object */
    gtcrn_stft_t* stft = gtcrn_stft_create(512, 256, 512);
    if (!stft) {
        fprintf(stderr, "Error: Failed to create STFT\n");
        return 1;
    }

    /* Process frame 0 exactly like streaming does */
    float stft_window[WIN_SIZE];
    float spec_real[N_FREQ];
    float spec_imag[N_FREQ];
    float stft_input_buffer[FRAME_SIZE] = {0};  /* Initialize to zeros */

    /* Frame 0: window = [zeros] + [audio[0:256]] */
    memcpy(stft_window, stft_input_buffer, FRAME_SIZE * sizeof(float));
    memcpy(stft_window + FRAME_SIZE, audio, FRAME_SIZE * sizeof(float));

    printf("Frame 0 STFT window:\n");
    printf("  First 10: ");
    for (int i = 0; i < 10; i++) printf("%.6f ", stft_window[i]);
    printf("\n");
    printf("  [256:266]: ");
    for (int i = 256; i < 266; i++) printf("%.6f ", stft_window[i]);
    printf("\n");
    printf("  Window sum: %.6f\n", 0.0);
    double wsum = 0;
    for (int i = 0; i < WIN_SIZE; i++) wsum += stft_window[i];
    printf("  Actual window sum: %.6f\n\n", wsum);

    /* Compute STFT */
    gtcrn_stft_frame(stft, stft_window, spec_real, spec_imag);

    /* Print results */
    double real_sum = 0, imag_sum = 0, mag_sum = 0;
    for (int f = 0; f < N_FREQ; f++) {
        real_sum += spec_real[f];
        imag_sum += spec_imag[f];
        mag_sum += sqrt(spec_real[f]*spec_real[f] + spec_imag[f]*spec_imag[f] + 1e-12);
    }

    printf("Frame 0 STFT output:\n");
    printf("  Real sum: %.6f\n", real_sum);
    printf("  Imag sum: %.6f\n", imag_sum);
    printf("  Mag sum: %.6f\n\n", mag_sum);

    printf("First 5 real: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", spec_real[i]);
    printf("\n");
    printf("First 5 imag: ");
    for (int i = 0; i < 5; i++) printf("%.6f ", spec_imag[i]);
    printf("\n");

    /* Also test frame 6 for comparison with debug output */
    printf("\n=== Frame 6 ===\n");

    /* Simulate processing frames 0-5 to update buffer */
    memcpy(stft_input_buffer, audio, FRAME_SIZE * sizeof(float));  /* Frame 0 */
    for (int f = 1; f <= 5; f++) {
        memcpy(stft_input_buffer, audio + f * FRAME_SIZE, FRAME_SIZE * sizeof(float));
    }

    /* Frame 6 window = [audio[5*256:6*256]] + [audio[6*256:7*256]] */
    memcpy(stft_window, audio + 5 * FRAME_SIZE, FRAME_SIZE * sizeof(float));
    memcpy(stft_window + FRAME_SIZE, audio + 6 * FRAME_SIZE, FRAME_SIZE * sizeof(float));

    gtcrn_stft_frame(stft, stft_window, spec_real, spec_imag);

    real_sum = imag_sum = mag_sum = 0;
    for (int f = 0; f < N_FREQ; f++) {
        real_sum += spec_real[f];
        imag_sum += spec_imag[f];
        mag_sum += sqrt(spec_real[f]*spec_real[f] + spec_imag[f]*spec_imag[f] + 1e-12);
    }

    printf("Frame 6 STFT output:\n");
    printf("  Real sum: %.6f\n", real_sum);
    printf("  Imag sum: %.6f\n", imag_sum);
    printf("  Mag sum: %.6f\n\n", mag_sum);

    printf("Expected from debug (C frame 6):\n");
    printf("  Real sum: 0.067720\n");
    printf("  Imag sum: -0.022344\n");
    printf("  Mag sum: 3.000679\n");

    gtcrn_stft_destroy(stft);
    free(audio);

    return 0;
}
