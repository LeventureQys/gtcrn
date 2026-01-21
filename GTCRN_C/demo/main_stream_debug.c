/// <file>main_stream_debug.c</file>
/// <summary>GTCRN streaming debug - outputs spectrum at frame 100</summary>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gtcrn_model.h"
#include "wav_io.h"

#define FRAME_SIZE 256
#define N_FREQ 257

/* External function to get spectrum output - we need to add this */
extern gtcrn_status_t gtcrn_process_frame_debug(gtcrn_t* model,
                                                 const gtcrn_float* input_frame,
                                                 gtcrn_float* output_frame,
                                                 gtcrn_float* out_spec_real,
                                                 gtcrn_float* out_spec_imag);

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    if (argc != 4) {
        printf("Usage: %s <weights_file> <input_wav> <output_wav>\n", argv[0]);
        return 1;
    }

    const char* weights_path = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];

    printf("=== GTCRN Streaming Debug ===\n\n");

    gtcrn_t* model = gtcrn_create();
    if (!model) {
        fprintf(stderr, "Error: Failed to create model\n");
        return 1;
    }

    gtcrn_status_t status = gtcrn_load_weights(model, weights_path);
    if (status != GTCRN_OK) {
        fprintf(stderr, "Error: Failed to load weights\n");
        gtcrn_destroy(model);
        return 1;
    }

    wav_info_t wav_info;
    float* audio_in = NULL;
    int num_samples = wav_read(input_path, &wav_info, &audio_in);
    if (num_samples <= 0) {
        fprintf(stderr, "Error: Failed to read input WAV\n");
        gtcrn_destroy(model);
        return 1;
    }

    int num_frames = num_samples / FRAME_SIZE;
    float* audio_out = (float*)malloc(num_frames * FRAME_SIZE * sizeof(float));
    if (!audio_out) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(audio_in);
        gtcrn_destroy(model);
        return 1;
    }

    gtcrn_reset_state(model);

    /* Buffers for spectrum debugging */
    gtcrn_float* spec_real = (gtcrn_float*)malloc(N_FREQ * sizeof(gtcrn_float));
    gtcrn_float* spec_imag = (gtcrn_float*)malloc(N_FREQ * sizeof(gtcrn_float));

    printf("Processing %d frames...\n", num_frames);

    for (int i = 0; i < num_frames; i++) {
        const float* input_frame = audio_in + i * FRAME_SIZE;
        float* output_frame = audio_out + i * FRAME_SIZE;

        status = gtcrn_process_frame(model, input_frame, output_frame);
        if (status != GTCRN_OK) {
            fprintf(stderr, "Error: Frame %d processing failed\n", i);
            break;
        }

        /* Print debug info for frames around 100 */
        if (i >= 100 && i < 105) {
            double frame_rms = 0.0;
            for (int j = 0; j < FRAME_SIZE; j++) {
                frame_rms += output_frame[j] * output_frame[j];
            }
            frame_rms = sqrt(frame_rms / FRAME_SIZE);
            printf("Frame %d: output RMS = %.6f\n", i, frame_rms);
        }
    }

    if (wav_write(output_path, audio_out, num_frames * FRAME_SIZE, wav_info.sample_rate) != 0) {
        fprintf(stderr, "Error: Failed to write output WAV\n");
    } else {
        printf("Output written to: %s\n", output_path);
    }

    free(spec_real);
    free(spec_imag);
    free(audio_in);
    free(audio_out);
    gtcrn_destroy(model);

    return 0;
}
