/// <file>test_stream_debug.c</file>
/// <summary>Debug streaming neural network output spectrum</summary>

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

    printf("=== Streaming Debug Test ===\n\n");

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

    /* Process frames and collect output spectrum sums */
    int num_frames = 120;
    float* output_frame = (float*)malloc(FRAME_SIZE * sizeof(float));

    printf("=== Output spectrum magnitude sums ===\n");
    printf("Frame | Out Mag Sum (C)  | Expected (Py)\n");
    printf("------+------------------+---------------\n");

    /* Expected values from Python (from save_py_spectrums.py output) */
    double py_mag_sums[] = {
        73.3360, 66.6587, 58.9623, 33.7761, 13.5810,
        34.6504, 67.7482, 78.5853, 84.1987, 79.6428
    };

    for (int i = 0; i < num_frames; i++) {
        gtcrn_process_frame(model, audio + i * FRAME_SIZE, output_frame);

        if (i >= 100 && i < 110) {
            /* Compute output RMS for this frame */
            double frame_rms = 0.0;
            for (int j = 0; j < FRAME_SIZE; j++) {
                frame_rms += output_frame[j] * output_frame[j];
            }
            frame_rms = sqrt(frame_rms / FRAME_SIZE);

            /* We can also compute expected ratio */
            int py_idx = i - 100;
            printf("%5d | RMS: %8.6f   | (Python mag sum: %.2f)\n",
                   i, frame_rms, py_mag_sums[py_idx]);
        }
    }

    /* Also check the workspace to get raw spectrum values */
    /* After gtcrn_process_frame, the workspace contains:
     * - stft_window at offset 0 (512 floats)
     * - spec_real at offset 512 (257 floats)
     * - spec_imag at offset 769 (257 floats)
     * - out_spec_real at offset 1026 (257 floats)
     * - out_spec_imag at offset 1283 (257 floats)
     */

    /* Process frame 100 again and check spectrum directly */
    printf("\n=== Direct spectrum check for frame 100 ===\n");
    gtcrn_reset_state(model);

    /* Warm up */
    for (int i = 0; i < 100; i++) {
        gtcrn_process_frame(model, audio + i * FRAME_SIZE, output_frame);
    }

    /* Process frame 100 */
    gtcrn_process_frame(model, audio + 100 * FRAME_SIZE, output_frame);

    /* Access the internal spectrum buffers */
    /* Based on gtcrn_model.c:326-344, the layout is:
     * stft_window = workspace
     * spec_real = stft_window + WIN_SIZE
     * spec_imag = spec_real + FREQ_BINS
     * out_spec_real = spec_imag + FREQ_BINS
     * out_spec_imag = out_spec_real + FREQ_BINS
     */
    float* workspace = model->workspace;
    float* out_spec_real = workspace + WIN_SIZE + 2 * N_FREQ;
    float* out_spec_imag = out_spec_real + N_FREQ;

    double c_mag_sum = 0.0;
    double c_real_sum = 0.0;
    double c_imag_sum = 0.0;

    for (int f = 0; f < N_FREQ; f++) {
        c_real_sum += out_spec_real[f];
        c_imag_sum += out_spec_imag[f];
        c_mag_sum += sqrt(out_spec_real[f] * out_spec_real[f] +
                          out_spec_imag[f] * out_spec_imag[f]);
    }

    printf("C output spectrum:\n");
    printf("  Real sum:  %.4f (Python: -0.3100)\n", c_real_sum);
    printf("  Imag sum:  %.4f (Python: -5.5179)\n", c_imag_sum);
    printf("  Mag sum:   %.4f (Python: 73.3360)\n", c_mag_sum);
    printf("  Ratio:     %.4f\n", c_mag_sum / 73.3360);

    printf("\n=== First 10 output spectrum values ===\n");
    printf("Bin | C Real     | C Imag     | C Mag\n");
    printf("----+------------+------------+--------\n");
    for (int i = 0; i < 10; i++) {
        double mag = sqrt(out_spec_real[i] * out_spec_real[i] +
                          out_spec_imag[i] * out_spec_imag[i]);
        printf("%3d | %10.6f | %10.6f | %.4f\n",
               i, out_spec_real[i], out_spec_imag[i], mag);
    }

    /* Clean up */
    free(output_frame);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
