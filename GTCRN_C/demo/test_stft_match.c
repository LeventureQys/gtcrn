/// <file>test_stft_match.c</file>
/// <summary>Verify C STFT matches Python STFT</summary>

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

    printf("=== STFT Match Test ===\n\n");

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

    /* Build STFT input for frame 100 (C style: [prev_frame, curr_frame]) */
    float* stft_window = (float*)malloc(WIN_SIZE * sizeof(float));
    float* spec_real = (float*)malloc(N_FREQ * sizeof(float));
    float* spec_imag = (float*)malloc(N_FREQ * sizeof(float));

    int frame_idx = 100;
    /* C style: stft_window = [audio[(frame-1)*256 : frame*256], audio[frame*256 : (frame+1)*256]] */
    /* But wait - the streaming code uses input buffer differently! */
    /* In streaming, stft_window = [stft_input_buffer (prev frame), current_frame] */
    /* So for frame 100, the input to frame 100's STFT is: */
    /*   stft_input_buffer = audio[99*256 : 100*256] (from frame 99) */
    /*   current_frame = audio[100*256 : 101*256] */

    memcpy(stft_window, audio + 99 * FRAME_SIZE, FRAME_SIZE * sizeof(float));
    memcpy(stft_window + FRAME_SIZE, audio + 100 * FRAME_SIZE, FRAME_SIZE * sizeof(float));

    /* STFT */
    gtcrn_stft_frame(model->stft, stft_window, spec_real, spec_imag);

    printf("=== Frame 100 INPUT spectrum (C) ===\n");
    double c_in_real_sum = 0, c_in_imag_sum = 0, c_in_mag_sum = 0;
    for (int f = 0; f < N_FREQ; f++) {
        c_in_real_sum += spec_real[f];
        c_in_imag_sum += spec_imag[f];
        c_in_mag_sum += sqrt(spec_real[f] * spec_real[f] + spec_imag[f] * spec_imag[f]);
    }
    printf("Real sum:  %.4f\n", c_in_real_sum);
    printf("Imag sum:  %.4f\n", c_in_imag_sum);
    printf("Mag sum:   %.4f\n", c_in_mag_sum);

    /* From Python save_py_spectrums.py / debug_nn_layers.py:
     * Input spec_real sum: -0.038603
     * Input spec_imag sum: -1.661163
     * Input magnitude sum: 115.842438
     */
    printf("\n=== Expected (from Python) ===\n");
    printf("Real sum:  -0.0386\n");
    printf("Imag sum:  -1.6612\n");
    printf("Mag sum:   115.8424\n");

    printf("\n=== Comparison ===\n");
    printf("Real sum ratio: %.4f\n", c_in_real_sum / -0.0386);
    printf("Imag sum ratio: %.4f\n", c_in_imag_sum / -1.6612);
    printf("Mag sum ratio:  %.4f\n", c_in_mag_sum / 115.8424);

    /* Print first 5 bins */
    printf("\n=== First 5 bins ===\n");
    printf("Bin | C Real     | C Imag\n");
    printf("----+------------+------------\n");
    for (int i = 0; i < 5; i++) {
        printf("%3d | %10.6f | %10.6f\n", i, spec_real[i], spec_imag[i]);
    }

    /* Clean up */
    free(stft_window);
    free(spec_real);
    free(spec_imag);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
