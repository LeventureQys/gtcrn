/// <file>test_stream_vs_complete.c</file>
/// <summary>Compare streaming vs complete processing for a single frame</summary>

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

/* External functions from gtcrn_stream.c */
extern gtcrn_status_t gtcrn_process_frame_impl(gtcrn_t* model,
                                                const float* spec_real,
                                                const float* spec_imag,
                                                float* out_real,
                                                float* out_imag);

/* External functions from gtcrn_forward.c */
extern gtcrn_status_t gtcrn_forward_complete(gtcrn_t* model,
                                              const float* spec_real,
                                              const float* spec_imag,
                                              float* out_real,
                                              float* out_imag,
                                              int n_frames);

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

    printf("=== Stream vs Complete Processing Test ===\n\n");

    /* Create models */
    gtcrn_t* stream_model = gtcrn_create();
    gtcrn_t* complete_model = gtcrn_create();

    if (!stream_model || !complete_model) {
        fprintf(stderr, "Error: Failed to create models\n");
        return 1;
    }

    /* Load weights */
    if (gtcrn_load_weights(stream_model, weights_path) != GTCRN_OK ||
        gtcrn_load_weights(complete_model, weights_path) != GTCRN_OK) {
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

    /* Warm up streaming model for 100 frames */
    printf("Warming up streaming model (100 frames)...\n");
    gtcrn_reset_state(stream_model);

    float* output_frame = (float*)malloc(FRAME_SIZE * sizeof(float));
    for (int i = 0; i < 100; i++) {
        gtcrn_process_frame(stream_model, audio + i * FRAME_SIZE, output_frame);
    }

    /* Now process frame 100 with streaming model */
    printf("Processing frame 100 with streaming model...\n");

    /* Get the spectrum input for frame 100 */
    float* stft_window = (float*)malloc(WIN_SIZE * sizeof(float));
    float* spec_real = (float*)malloc(N_FREQ * sizeof(float));
    float* spec_imag = (float*)malloc(N_FREQ * sizeof(float));
    float* stream_out_real = (float*)malloc(N_FREQ * sizeof(float));
    float* stream_out_imag = (float*)malloc(N_FREQ * sizeof(float));
    float* complete_out_real = (float*)malloc(N_FREQ * sizeof(float));
    float* complete_out_imag = (float*)malloc(N_FREQ * sizeof(float));

    /* Build STFT input for frame 100 (C style: [prev_frame, curr_frame]) */
    int frame_idx = 100;
    memcpy(stft_window, audio + (frame_idx - 1) * FRAME_SIZE, FRAME_SIZE * sizeof(float));
    memcpy(stft_window + FRAME_SIZE, audio + frame_idx * FRAME_SIZE, FRAME_SIZE * sizeof(float));

    /* STFT */
    gtcrn_stft_frame(stream_model->stft, stft_window, spec_real, spec_imag);

    printf("\n=== Input spectrum (frame 100) ===\n");
    double in_real_sum = 0, in_imag_sum = 0, in_mag_sum = 0;
    for (int i = 0; i < N_FREQ; i++) {
        in_real_sum += spec_real[i];
        in_imag_sum += spec_imag[i];
        in_mag_sum += sqrt(spec_real[i] * spec_real[i] + spec_imag[i] * spec_imag[i]);
    }
    printf("Real sum: %.4f\n", in_real_sum);
    printf("Imag sum: %.4f\n", in_imag_sum);
    printf("Mag sum:  %.4f\n", in_mag_sum);

    /* Process with streaming neural network */
    gtcrn_process_frame_impl(stream_model, spec_real, spec_imag,
                              stream_out_real, stream_out_imag);

    printf("\n=== Streaming output spectrum ===\n");
    double stream_real_sum = 0, stream_imag_sum = 0, stream_mag_sum = 0;
    for (int i = 0; i < N_FREQ; i++) {
        stream_real_sum += stream_out_real[i];
        stream_imag_sum += stream_out_imag[i];
        stream_mag_sum += sqrt(stream_out_real[i] * stream_out_real[i] +
                               stream_out_imag[i] * stream_out_imag[i]);
    }
    printf("Real sum: %.4f\n", stream_real_sum);
    printf("Imag sum: %.4f\n", stream_imag_sum);
    printf("Mag sum:  %.4f\n", stream_mag_sum);

    /* Now process the same input with complete model */
    /* Need to process enough frames to match streaming model's history */
    printf("\nProcessing with complete model (101 frames)...\n");

    int n_complete_frames = 101;
    float* complete_spec_real = (float*)malloc(n_complete_frames * N_FREQ * sizeof(float));
    float* complete_spec_imag = (float*)malloc(n_complete_frames * N_FREQ * sizeof(float));
    float* complete_all_out_real = (float*)malloc(n_complete_frames * N_FREQ * sizeof(float));
    float* complete_all_out_imag = (float*)malloc(n_complete_frames * N_FREQ * sizeof(float));

    /* STFT for all frames (complete style) */
    gtcrn_stft_forward(complete_model->stft, audio,
                       n_complete_frames * FRAME_SIZE + WIN_SIZE - FRAME_SIZE,
                       complete_spec_real, complete_spec_imag);

    /* Complete forward */
    gtcrn_forward_complete(complete_model, complete_spec_real, complete_spec_imag,
                           complete_all_out_real, complete_all_out_imag, n_complete_frames);

    /* Get frame 100's output from complete model */
    /* Due to 1-frame offset, frame 100 of streaming = frame 100 of complete (if same STFT) */
    memcpy(complete_out_real, complete_all_out_real + 100 * N_FREQ, N_FREQ * sizeof(float));
    memcpy(complete_out_imag, complete_all_out_imag + 100 * N_FREQ, N_FREQ * sizeof(float));

    printf("\n=== Complete output spectrum (frame 100) ===\n");
    double complete_real_sum = 0, complete_imag_sum = 0, complete_mag_sum = 0;
    for (int i = 0; i < N_FREQ; i++) {
        complete_real_sum += complete_out_real[i];
        complete_imag_sum += complete_out_imag[i];
        complete_mag_sum += sqrt(complete_out_real[i] * complete_out_real[i] +
                                 complete_out_imag[i] * complete_out_imag[i]);
    }
    printf("Real sum: %.4f\n", complete_real_sum);
    printf("Imag sum: %.4f\n", complete_imag_sum);
    printf("Mag sum:  %.4f\n", complete_mag_sum);

    /* Compare */
    printf("\n=== Comparison ===\n");
    printf("Streaming/Complete mag ratio: %.4f\n", stream_mag_sum / complete_mag_sum);

    /* Check correlation */
    double mean_s = stream_mag_sum / N_FREQ;
    double mean_c = complete_mag_sum / N_FREQ;
    double cov = 0, var_s = 0, var_c = 0;

    for (int i = 0; i < N_FREQ; i++) {
        double s_mag = sqrt(stream_out_real[i] * stream_out_real[i] +
                            stream_out_imag[i] * stream_out_imag[i]);
        double c_mag = sqrt(complete_out_real[i] * complete_out_real[i] +
                            complete_out_imag[i] * complete_out_imag[i]);
        cov += (s_mag - mean_s) * (c_mag - mean_c);
        var_s += (s_mag - mean_s) * (s_mag - mean_s);
        var_c += (c_mag - mean_c) * (c_mag - mean_c);
    }

    double corr = cov / sqrt(var_s * var_c + 1e-12);
    printf("Magnitude correlation: %.4f\n", corr);

    /* First 5 bins comparison */
    printf("\n=== First 5 bins comparison ===\n");
    printf("Bin | Stream Real | Complete Real | Stream Imag | Complete Imag\n");
    printf("----+-------------+---------------+-------------+--------------\n");
    for (int i = 0; i < 5; i++) {
        printf("%3d | %11.6f | %13.6f | %11.6f | %12.6f\n",
               i, stream_out_real[i], complete_out_real[i],
               stream_out_imag[i], complete_out_imag[i]);
    }

    /* Clean up */
    free(stft_window);
    free(spec_real);
    free(spec_imag);
    free(stream_out_real);
    free(stream_out_imag);
    free(complete_out_real);
    free(complete_out_imag);
    free(complete_spec_real);
    free(complete_spec_imag);
    free(complete_all_out_real);
    free(complete_all_out_imag);
    free(output_frame);
    free(audio);
    gtcrn_destroy(stream_model);
    gtcrn_destroy(complete_model);

    return 0;
}
