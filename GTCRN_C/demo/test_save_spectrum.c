/// <file>test_save_spectrum.c</file>
/// <summary>Save C streaming output spectrum to binary file for Python comparison</summary>

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

    if (argc != 4) {
        printf("Usage: %s <weights_file> <input_wav> <output_bin>\n", argv[0]);
        printf("  output_bin: Binary file with (n_frames, 257, 2) float32 array\n");
        return 1;
    }

    const char* weights_path = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];

    printf("=== Save C Stream Spectrum ===\n\n");

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

    /* Calculate number of frames */
    int n_frames = num_samples / FRAME_SIZE;
    printf("Processing %d frames\n", n_frames);

    /* Allocate output buffer for all spectrums */
    float* all_spec = (float*)malloc(n_frames * N_FREQ * 2 * sizeof(float));
    if (!all_spec) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }

    /* Reset state */
    gtcrn_reset_state(model);

    /* Access internal spectrum buffers */
    float* workspace = model->workspace;
    float* spec_real = workspace + WIN_SIZE;
    float* spec_imag = spec_real + N_FREQ;
    float* out_spec_real = spec_imag + N_FREQ;
    float* out_spec_imag = out_spec_real + N_FREQ;

    float* output_frame = (float*)malloc(FRAME_SIZE * sizeof(float));

    /* Process all frames */
    for (int frame = 0; frame < n_frames; frame++) {
        float* input_frame = audio + frame * FRAME_SIZE;
        gtcrn_process_frame(model, input_frame, output_frame);

        /* Copy output spectrum to buffer */
        for (int f = 0; f < N_FREQ; f++) {
            all_spec[(frame * N_FREQ + f) * 2 + 0] = out_spec_real[f];
            all_spec[(frame * N_FREQ + f) * 2 + 1] = out_spec_imag[f];
        }

        if ((frame + 1) % 100 == 0) {
            printf("\r  Processed %d / %d frames", frame + 1, n_frames);
            fflush(stdout);
        }
    }
    printf("\n");

    /* Print some statistics for verification */
    printf("\nFrame 100 stats:\n");
    double sum_real = 0, sum_imag = 0, sum_mag2 = 0;
    for (int f = 0; f < N_FREQ; f++) {
        float r = all_spec[(100 * N_FREQ + f) * 2 + 0];
        float i = all_spec[(100 * N_FREQ + f) * 2 + 1];
        sum_real += r;
        sum_imag += i;
        sum_mag2 += r * r + i * i;
    }
    printf("  Sum real: %.6f\n", sum_real);
    printf("  Sum imag: %.6f\n", sum_imag);
    printf("  Sum mag^2: %.6f\n", sum_mag2);

    printf("\nFirst 5 real values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", all_spec[(100 * N_FREQ + i) * 2 + 0]);
    }
    printf("\n");

    printf("First 5 imag values: ");
    for (int i = 0; i < 5; i++) {
        printf("%.6f ", all_spec[(100 * N_FREQ + i) * 2 + 1]);
    }
    printf("\n");

    /* Save to binary file */
    printf("\nSaving to %s...\n", output_path);
    FILE* fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open output file\n");
        return 1;
    }

    /* Write header: n_frames, n_freq */
    fwrite(&n_frames, sizeof(int), 1, fp);
    int n_freq = N_FREQ;
    fwrite(&n_freq, sizeof(int), 1, fp);

    /* Write spectrum data */
    size_t written = fwrite(all_spec, sizeof(float), n_frames * N_FREQ * 2, fp);
    fclose(fp);

    printf("Saved %zu floats (%d frames x %d freq x 2)\n", written, n_frames, N_FREQ);

    free(output_frame);
    free(all_spec);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
