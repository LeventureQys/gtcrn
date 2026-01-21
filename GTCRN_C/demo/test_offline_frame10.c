/// <file>test_offline_frame10.c</file>
/// <summary>Test C offline output for frame 10</summary>

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

    printf("=== C Offline Frame 10 Test ===\n\n");

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

    /* Pad audio at the beginning to match Python */
    int padded_len = num_samples + FRAME_SIZE;
    float* audio_padded = (float*)calloc(padded_len, sizeof(float));
    memcpy(audio_padded + FRAME_SIZE, audio, num_samples * sizeof(float));

    /* Run offline processing */
    float* output = (float*)malloc(padded_len * sizeof(float));
    int output_len;
    gtcrn_forward(model, audio_padded, padded_len, output, &output_len);

    printf("Offline processing done\n");
    printf("Output length: %d samples\n\n", output_len);

    /* The offline output spectrum is in model->workspace after the last frame */
    /* But we need to process the spectrum for frame 10 specifically */
    /* Let's just compare the time-domain output */

    /* For now, let's compare the time-domain samples around frame 10 */
    /* Frame 10 starts at sample 10*256 = 2560 */
    int frame10_start = 10 * FRAME_SIZE;
    printf("Offline output samples at frame 10 (samples %d-%d):\n",
           frame10_start, frame10_start + 10);
    for (int i = 0; i < 10; i++) {
        printf("  [%d] = %.6f\n", frame10_start + i, output[frame10_start + i]);
    }

    free(audio_padded);
    free(output);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
