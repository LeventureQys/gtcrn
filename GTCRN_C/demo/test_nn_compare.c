/// <file>test_nn_compare.c</file>
/// <summary>Compare C stream NN forward vs offline NN forward</summary>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gtcrn_model.h"
#include "wav_io.h"

#define N_FREQ 257
#define WIN_SIZE 512

/* Declare the internal forward function */
extern gtcrn_status_t gtcrn_process_frame_impl(gtcrn_t* model,
    const gtcrn_float* spec_real, const gtcrn_float* spec_imag,
    gtcrn_float* out_spec_real, gtcrn_float* out_spec_imag);

/* Declare the offline forward for a single frame */
extern gtcrn_status_t gtcrn_forward_single_frame(gtcrn_t* model,
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

    printf("=== NN Forward Comparison Test ===\n\n");

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

    /* Compute STFT for first few frames */
    gtcrn_float spec_real[N_FREQ];
    gtcrn_float spec_imag[N_FREQ];
    gtcrn_float out_stream_real[N_FREQ];
    gtcrn_float out_stream_imag[N_FREQ];

    /* Create window for manual STFT */
    gtcrn_float window[WIN_SIZE];
    for (int i = 0; i < WIN_SIZE; i++) {
        window[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * i / WIN_SIZE)));
    }

    /* Reset stream state */
    gtcrn_reset_state(model);

    printf("Testing frame 5 (after warmup):\n\n");

    /* Process frames 0-4 to warm up cache */
    gtcrn_float output_frame[256];
    for (int frame = 0; frame < 5; frame++) {
        gtcrn_process_frame(model, audio + frame * 256, output_frame);
    }

    /* Frame 5: Get the input spectrum from the model */
    /* First, do the STFT manually to get the same input as stream */
    gtcrn_float stft_window[WIN_SIZE];

    /* Stream frame 5 uses: [prev_frame = audio[4*256:5*256], curr_frame = audio[5*256:6*256]] */
    /* Which is audio[4*256 : 6*256] = audio[1024:1536] */
    memcpy(stft_window, audio + 4 * 256, 256 * sizeof(gtcrn_float));
    memcpy(stft_window + 256, audio + 5 * 256, 256 * sizeof(gtcrn_float));

    /* Apply window and compute FFT */
    gtcrn_float windowed[WIN_SIZE];
    for (int i = 0; i < WIN_SIZE; i++) {
        windowed[i] = stft_window[i] * window[i];
    }

    /* Simple DFT for verification */
    for (int k = 0; k < N_FREQ; k++) {
        gtcrn_float real_sum = 0.0f;
        gtcrn_float imag_sum = 0.0f;
        for (int n = 0; n < WIN_SIZE; n++) {
            gtcrn_float angle = -2.0f * 3.14159265358979323846f * k * n / WIN_SIZE;
            real_sum += windowed[n] * cosf(angle);
            imag_sum += windowed[n] * sinf(angle);
        }
        spec_real[k] = real_sum;
        spec_imag[k] = imag_sum;
    }

    /* Process with stream forward */
    gtcrn_process_frame_impl(model, spec_real, spec_imag, out_stream_real, out_stream_imag);

    printf("Input spectrum (frame 5):\n");
    double in_real_sum = 0, in_imag_sum = 0;
    for (int f = 0; f < N_FREQ; f++) {
        in_real_sum += spec_real[f];
        in_imag_sum += spec_imag[f];
    }
    printf("  Real sum: %.4f\n", in_real_sum);
    printf("  Imag sum: %.4f\n\n", in_imag_sum);

    printf("Stream output spectrum:\n");
    double out_real_sum = 0, out_imag_sum = 0, out_mag_sum = 0;
    for (int f = 0; f < N_FREQ; f++) {
        out_real_sum += out_stream_real[f];
        out_imag_sum += out_stream_imag[f];
        out_mag_sum += sqrtf(out_stream_real[f] * out_stream_real[f] +
                            out_stream_imag[f] * out_stream_imag[f]);
    }
    printf("  Real sum: %.4f\n", out_real_sum);
    printf("  Imag sum: %.4f\n", out_imag_sum);
    printf("  Mag sum: %.4f\n", out_mag_sum);

    free(audio);
    gtcrn_destroy(model);

    return 0;
}
