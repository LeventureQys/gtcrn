/// <file>test_intermediate.c</file>
/// <summary>Compare C streaming intermediate outputs with Python</summary>

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

/* External declaration of stream forward impl for debugging */
extern gtcrn_status_t gtcrn_process_frame_impl(gtcrn_t* model,
    const gtcrn_float* spec_real, const gtcrn_float* spec_imag,
    gtcrn_float* out_spec_real, gtcrn_float* out_spec_imag);

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

    printf("=== C Intermediate Output Test ===\n\n");

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

    /* Process frames 0-4 */
    printf("Processing frames 0-4...\n");
    float* output_frame = (float*)malloc(FRAME_SIZE * sizeof(float));

    for (int frame = 0; frame < 5; frame++) {
        float* input_frame = audio + frame * FRAME_SIZE;
        gtcrn_process_frame(model, input_frame, output_frame);
    }

    /* Now for frame 5, manually trace through and print intermediate values */
    /* At this point, stft_input_buffer contains audio[1024:1280] (frame 4's input) */

    /* Access workspace pointers */
    gtcrn_float* workspace = model->workspace;
    gtcrn_float* stft_window = workspace;
    gtcrn_float* spec_real = stft_window + WIN_SIZE;
    gtcrn_float* spec_imag = spec_real + N_FREQ;
    gtcrn_float* out_spec_real = spec_imag + N_FREQ;
    gtcrn_float* out_spec_imag = out_spec_real + N_FREQ;

    /* Build STFT window for frame 5 */
    /* C frame 5 uses: [prev_frame = audio[4*256:5*256] = audio[1024:1280],
                        curr_frame = audio[5*256:6*256] = audio[1280:1536]] */
    gtcrn_state_t* state = model->state;
    memcpy(stft_window, state->stft_input_buffer, FRAME_SIZE * sizeof(gtcrn_float));
    memcpy(stft_window + FRAME_SIZE, audio + 5 * FRAME_SIZE, FRAME_SIZE * sizeof(gtcrn_float));

    /* Compute STFT */
    gtcrn_stft_frame(model->stft, stft_window, spec_real, spec_imag);

    /* Calculate feature stats */
    double mag_sum = 0.0, real_sum = 0.0, imag_sum = 0.0;
    for (int f = 0; f < N_FREQ; f++) {
        double r = spec_real[f];
        double i = spec_imag[f];
        mag_sum += sqrt(r * r + i * i + 1e-12);
        real_sum += r;
        imag_sum += i;
    }

    printf("\nFrame 5 Feature tensor:\n");
    printf("  Sum (mag): %.6f  (Python: 3.000679)\n", mag_sum);
    printf("  Sum (real): %.6f  (Python: 0.067720)\n", real_sum);
    printf("  Sum (imag): %.6f  (Python: -0.022347)\n", imag_sum);

    /* Now process this frame through the NN */
    /* Update input buffer first */
    memcpy(state->stft_input_buffer, audio + 5 * FRAME_SIZE, FRAME_SIZE * sizeof(gtcrn_float));

    /* Call the NN forward */
    gtcrn_process_frame_impl(model, spec_real, spec_imag, out_spec_real, out_spec_imag);

    /* Output stats */
    double out_real_sum = 0.0, out_imag_sum = 0.0, out_mag_sum = 0.0;
    for (int f = 0; f < N_FREQ; f++) {
        out_real_sum += out_spec_real[f];
        out_imag_sum += out_spec_imag[f];
        out_mag_sum += sqrt(out_spec_real[f] * out_spec_real[f] +
                           out_spec_imag[f] * out_spec_imag[f]);
    }

    printf("\nFrame 5 output:\n");
    printf("  Real sum: %.6f  (Python: 0.064489)\n", out_real_sum);
    printf("  Imag sum: %.6f  (Python: -0.004923)\n", out_imag_sum);
    printf("  Mag sum: %.6f  (Python: 0.203372)\n", out_mag_sum);

    free(output_frame);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
