/// <file>debug_stream_layers.c</file>
/// <summary>Debug streaming layer-by-layer outputs vs Python</summary>

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
#define FREQ_ERB 129
#define FREQ_65 65
#define FREQ_DOWN 33

#ifndef GTCRN_EPS
#define GTCRN_EPS 1e-12f
#endif

/* Helper to compute sum of array */
static double array_sum(const float* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

/* External declarations from gtcrn_stream.c */
extern void erb_bm_stream(const gtcrn_weights_t* w, const gtcrn_float* input,
                          gtcrn_float* output, int channels);
extern void sfe_stream(const gtcrn_float* input, gtcrn_float* output,
                       int channels, int freq);
extern void conv2d_stream_frame(const gtcrn_float* weight,
                                const gtcrn_float* bias,
                                gtcrn_float* cache,
                                const gtcrn_float* input,
                                gtcrn_float* output,
                                int in_ch, int out_ch,
                                int kernel_t, int kernel_f,
                                int stride_f, int pad_f,
                                int dilation_t,
                                int cache_t, int freq_in, int freq_out,
                                int groups);
extern void bn_stream(const gtcrn_float* gamma, const gtcrn_float* beta,
                      const gtcrn_float* mean, const gtcrn_float* var,
                      gtcrn_float* data, int channels, int freq);
extern void prelu_stream(const gtcrn_float* alpha, gtcrn_float* data,
                         int channels, int freq);
extern void channel_shuffle_stream(const gtcrn_float* x1, const gtcrn_float* x2,
                                   gtcrn_float* output, int half_ch, int freq);
extern void tra_gru_step(const gtcrn_float* gru_ih, const gtcrn_float* gru_hh,
                         const gtcrn_float* gru_bih, const gtcrn_float* gru_bhh,
                         const gtcrn_float* fc_weight, const gtcrn_float* fc_bias,
                         gtcrn_float* hidden, gtcrn_float* x, int channels, int freq,
                         gtcrn_float* workspace);

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

    printf("=== C Streaming Layer Debug ===\n\n");

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

    /* Process frames 0-4 to warm up state */
    printf("Processing frames 0-4 to warm up...\n");
    float* output_frame = (float*)malloc(FRAME_SIZE * sizeof(float));

    for (int frame = 0; frame < 5; frame++) {
        float* input_frame = audio + frame * FRAME_SIZE;
        gtcrn_process_frame(model, input_frame, output_frame);
    }

    /* Now manually trace through frame 5 */
    printf("\n=== Frame 5 Layer-by-Layer Debug ===\n");

    gtcrn_weights_t* w = model->weights;
    gtcrn_state_t* s = model->state;

    /* Allocate working buffers */
    float* feat = (float*)calloc(3 * N_FREQ, sizeof(float));
    float* erb_out = (float*)calloc(3 * FREQ_ERB, sizeof(float));
    float* sfe_out = (float*)calloc(9 * FREQ_ERB, sizeof(float));
    float* en_conv0_out = (float*)calloc(16 * FREQ_65, sizeof(float));
    float* en_conv1_out = (float*)calloc(16 * FREQ_DOWN, sizeof(float));
    float* buf4 = (float*)calloc(16 * FREQ_DOWN, sizeof(float));
    float* buf5 = (float*)calloc(16 * FREQ_DOWN, sizeof(float));
    float* scratch = (float*)calloc(64 * FREQ_DOWN, sizeof(float));

    /* Build STFT input for frame 5 */
    /* C frame 5 uses: [prev = audio[4*256:5*256], curr = audio[5*256:6*256]] */
    float* stft_window = (float*)calloc(WIN_SIZE, sizeof(float));
    memcpy(stft_window, s->stft_input_buffer, FRAME_SIZE * sizeof(float));
    memcpy(stft_window + FRAME_SIZE, audio + 5 * FRAME_SIZE, FRAME_SIZE * sizeof(float));

    /* Compute STFT */
    float* spec_real = (float*)calloc(N_FREQ, sizeof(float));
    float* spec_imag = (float*)calloc(N_FREQ, sizeof(float));
    gtcrn_stft_frame(model->stft, stft_window, spec_real, spec_imag);

    /* Step 1: Create feature tensor (3, 257) = [mag, real, imag] */
    for (int f = 0; f < N_FREQ; f++) {
        float r = spec_real[f];
        float i = spec_imag[f];
        float mag = sqrtf(r * r + i * i + GTCRN_EPS);
        feat[0 * N_FREQ + f] = mag;
        feat[1 * N_FREQ + f] = r;
        feat[2 * N_FREQ + f] = i;
    }

    printf("\nInput features (3, 257):\n");
    printf("  Mag sum: %.6f (Python: 3.000679)\n", array_sum(feat, N_FREQ));
    printf("  Real sum: %.6f (Python: 0.067720)\n", array_sum(feat + N_FREQ, N_FREQ));
    printf("  Imag sum: %.6f (Python: -0.022347)\n", array_sum(feat + 2 * N_FREQ, N_FREQ));

    /* Step 2: ERB compression (3, 257) -> (3, 129) */
    erb_bm_stream(w, feat, erb_out, 3);
    printf("\nAfter ERB compression (3, 129):\n");
    printf("  Sum: %.6f (Python: 3.046053)\n", array_sum(erb_out, 3 * FREQ_ERB));

    /* Step 3: SFE (3, 129) -> (9, 129) */
    sfe_stream(erb_out, sfe_out, 3, FREQ_ERB);
    printf("\nAfter SFE (9, 129):\n");
    printf("  Sum: %.6f (Python: 8.866683)\n", array_sum(sfe_out, 9 * FREQ_ERB));

    /* Step 4: EnConv0 (9, 129) -> (16, 65) */
    conv2d_stream_frame(w->en_conv0_weight, w->en_conv0_bias,
                        NULL, sfe_out,
                        en_conv0_out, 9, 16, 1, 5, 2, 2, 1, 1, 129, 65, 1);
    printf("\nAfter EnConv0 conv (16, 65):\n");
    printf("  Sum (before BN): %.6f\n", array_sum(en_conv0_out, 16 * FREQ_65));

    bn_stream(w->en_bn0_gamma, w->en_bn0_beta, w->en_bn0_mean, w->en_bn0_var,
              en_conv0_out, 16, 65);
    printf("  Sum (after BN): %.6f\n", array_sum(en_conv0_out, 16 * FREQ_65));

    prelu_stream(w->en_prelu0, en_conv0_out, 16, 65);
    printf("  Sum (after PReLU): %.6f (Python: 131.510941)\n", array_sum(en_conv0_out, 16 * FREQ_65));

    /* Step 5: EnConv1 (16, 65) -> (16, 33) */
    conv2d_stream_frame(w->en_conv1_weight, w->en_conv1_bias,
                        NULL, en_conv0_out,
                        en_conv1_out, 16, 16, 1, 5, 2, 2, 1, 1, 65, 33, 2);
    bn_stream(w->en_bn1_gamma, w->en_bn1_beta, w->en_bn1_mean, w->en_bn1_var,
              en_conv1_out, 16, 33);
    prelu_stream(w->en_prelu1, en_conv1_out, 16, 33);
    printf("\nAfter EnConv1 (16, 33):\n");
    printf("  Sum: %.6f (Python: 107.113319)\n", array_sum(en_conv1_out, 16 * FREQ_DOWN));

    /* Step 6: GTConvBlock 2 (dilation=1) */
    {
        int half_ch = 8;
        float* x1 = scratch;
        float* x2 = x1 + half_ch * FREQ_DOWN;
        float* x1_sfe = x2 + half_ch * FREQ_DOWN;
        float* h1 = x1_sfe + 24 * FREQ_DOWN;
        float* h1_out = h1 + 16 * FREQ_DOWN;

        /* Channel split */
        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < FREQ_DOWN; f++) {
                x1[c * FREQ_DOWN + f] = en_conv1_out[c * FREQ_DOWN + f];
                x2[c * FREQ_DOWN + f] = en_conv1_out[(c + half_ch) * FREQ_DOWN + f];
            }
        }

        /* SFE on x1 */
        sfe_stream(x1, x1_sfe, half_ch, FREQ_DOWN);

        /* Point conv 1 */
        conv2d_stream_frame(w->en_gt2_pc1_weight, w->en_gt2_pc1_bias,
                            NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, FREQ_DOWN, FREQ_DOWN, 1);
        bn_stream(w->en_gt2_bn1_gamma, w->en_gt2_bn1_beta, w->en_gt2_bn1_mean, w->en_gt2_bn1_var,
                  h1, 16, FREQ_DOWN);
        prelu_stream(w->en_gt2_prelu1, h1, 16, FREQ_DOWN);

        /* Depth conv with cache (dilation=1, cache_t=2) */
        conv2d_stream_frame(w->en_gt2_dc_weight, w->en_gt2_dc_bias,
                            s->en_conv_cache, h1,
                            buf4, 16, 16, 3, 3, 1, 1, 1, 2, FREQ_DOWN, FREQ_DOWN, 16);
        bn_stream(w->en_gt2_bn2_gamma, w->en_gt2_bn2_beta, w->en_gt2_bn2_mean, w->en_gt2_bn2_var,
                  buf4, 16, FREQ_DOWN);
        prelu_stream(w->en_gt2_prelu2, buf4, 16, FREQ_DOWN);

        /* Point conv 2 */
        conv2d_stream_frame(w->en_gt2_pc2_weight, w->en_gt2_pc2_bias,
                            NULL, buf4, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, FREQ_DOWN, FREQ_DOWN, 1);
        bn_stream(w->en_gt2_bn3_gamma, w->en_gt2_bn3_beta, w->en_gt2_bn3_mean, w->en_gt2_bn3_var,
                  h1_out, 8, FREQ_DOWN);

        /* TRA (modifies h1_out in-place) */
        tra_gru_step(w->en_gt2_tra_gru_ih, w->en_gt2_tra_gru_hh,
                     w->en_gt2_tra_gru_bih, w->en_gt2_tra_gru_bhh,
                     w->en_gt2_tra_fc_weight, w->en_gt2_tra_fc_bias,
                     s->en_tra_h2, h1_out, 8, FREQ_DOWN, scratch + 32 * FREQ_DOWN);

        /* Channel shuffle */
        channel_shuffle_stream(h1_out, x2, buf4, 8, FREQ_DOWN);
    }
    printf("\nAfter EnGT2 (16, 33):\n");
    printf("  Sum: %.6f (Python: 114.750290)\n", array_sum(buf4, 16 * FREQ_DOWN));

    /* Step 7: GTConvBlock 3 (dilation=2) */
    {
        int half_ch = 8;
        float* x1 = scratch;
        float* x2 = x1 + half_ch * FREQ_DOWN;
        float* x1_sfe = x2 + half_ch * FREQ_DOWN;
        float* h1 = x1_sfe + 24 * FREQ_DOWN;
        float* h1_out = h1 + 16 * FREQ_DOWN;

        /* Channel split */
        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < FREQ_DOWN; f++) {
                x1[c * FREQ_DOWN + f] = buf4[c * FREQ_DOWN + f];
                x2[c * FREQ_DOWN + f] = buf4[(c + half_ch) * FREQ_DOWN + f];
            }
        }

        /* SFE on x1 */
        sfe_stream(x1, x1_sfe, half_ch, FREQ_DOWN);

        /* Point conv 1 */
        conv2d_stream_frame(w->en_gt3_pc1_weight, w->en_gt3_pc1_bias,
                            NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, FREQ_DOWN, FREQ_DOWN, 1);
        bn_stream(w->en_gt3_bn1_gamma, w->en_gt3_bn1_beta, w->en_gt3_bn1_mean, w->en_gt3_bn1_var,
                  h1, 16, FREQ_DOWN);
        prelu_stream(w->en_gt3_prelu1, h1, 16, FREQ_DOWN);

        /* Depth conv with cache (dilation=2, cache_t=4) */
        conv2d_stream_frame(w->en_gt3_dc_weight, w->en_gt3_dc_bias,
                            s->en_conv_cache + 2 * 16 * FREQ_DOWN, h1,
                            buf4, 16, 16, 3, 3, 1, 1, 2, 4, FREQ_DOWN, FREQ_DOWN, 16);
        bn_stream(w->en_gt3_bn2_gamma, w->en_gt3_bn2_beta, w->en_gt3_bn2_mean, w->en_gt3_bn2_var,
                  buf4, 16, FREQ_DOWN);
        prelu_stream(w->en_gt3_prelu2, buf4, 16, FREQ_DOWN);

        /* Point conv 2 */
        conv2d_stream_frame(w->en_gt3_pc2_weight, w->en_gt3_pc2_bias,
                            NULL, buf4, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, FREQ_DOWN, FREQ_DOWN, 1);
        bn_stream(w->en_gt3_bn3_gamma, w->en_gt3_bn3_beta, w->en_gt3_bn3_mean, w->en_gt3_bn3_var,
                  h1_out, 8, FREQ_DOWN);

        /* TRA */
        tra_gru_step(w->en_gt3_tra_gru_ih, w->en_gt3_tra_gru_hh,
                     w->en_gt3_tra_gru_bih, w->en_gt3_tra_gru_bhh,
                     w->en_gt3_tra_fc_weight, w->en_gt3_tra_fc_bias,
                     s->en_tra_h3, h1_out, 8, FREQ_DOWN, scratch + 32 * FREQ_DOWN);

        /* Channel shuffle */
        channel_shuffle_stream(h1_out, x2, buf4, 8, FREQ_DOWN);
    }
    printf("\nAfter EnGT3 (16, 33):\n");
    printf("  Sum: %.6f (Python: 64.331329)\n", array_sum(buf4, 16 * FREQ_DOWN));

    /* Step 8: GTConvBlock 4 (dilation=5) */
    {
        int half_ch = 8;
        float* x1 = scratch;
        float* x2 = x1 + half_ch * FREQ_DOWN;
        float* x1_sfe = x2 + half_ch * FREQ_DOWN;
        float* h1 = x1_sfe + 24 * FREQ_DOWN;
        float* h1_out = h1 + 16 * FREQ_DOWN;

        /* Channel split */
        for (int c = 0; c < half_ch; c++) {
            for (int f = 0; f < FREQ_DOWN; f++) {
                x1[c * FREQ_DOWN + f] = buf4[c * FREQ_DOWN + f];
                x2[c * FREQ_DOWN + f] = buf4[(c + half_ch) * FREQ_DOWN + f];
            }
        }

        /* SFE on x1 */
        sfe_stream(x1, x1_sfe, half_ch, FREQ_DOWN);

        /* Point conv 1 */
        conv2d_stream_frame(w->en_gt4_pc1_weight, w->en_gt4_pc1_bias,
                            NULL, x1_sfe, h1, 24, 16, 1, 1, 1, 0, 1, 1, FREQ_DOWN, FREQ_DOWN, 1);
        bn_stream(w->en_gt4_bn1_gamma, w->en_gt4_bn1_beta, w->en_gt4_bn1_mean, w->en_gt4_bn1_var,
                  h1, 16, FREQ_DOWN);
        prelu_stream(w->en_gt4_prelu1, h1, 16, FREQ_DOWN);

        /* Depth conv with cache (dilation=5, cache_t=10) */
        conv2d_stream_frame(w->en_gt4_dc_weight, w->en_gt4_dc_bias,
                            s->en_conv_cache + 6 * 16 * FREQ_DOWN, h1,
                            buf4, 16, 16, 3, 3, 1, 1, 5, 10, FREQ_DOWN, FREQ_DOWN, 16);
        bn_stream(w->en_gt4_bn2_gamma, w->en_gt4_bn2_beta, w->en_gt4_bn2_mean, w->en_gt4_bn2_var,
                  buf4, 16, FREQ_DOWN);
        prelu_stream(w->en_gt4_prelu2, buf4, 16, FREQ_DOWN);

        /* Point conv 2 */
        conv2d_stream_frame(w->en_gt4_pc2_weight, w->en_gt4_pc2_bias,
                            NULL, buf4, h1_out, 16, 8, 1, 1, 1, 0, 1, 1, FREQ_DOWN, FREQ_DOWN, 1);
        bn_stream(w->en_gt4_bn3_gamma, w->en_gt4_bn3_beta, w->en_gt4_bn3_mean, w->en_gt4_bn3_var,
                  h1_out, 8, FREQ_DOWN);

        /* TRA */
        tra_gru_step(w->en_gt4_tra_gru_ih, w->en_gt4_tra_gru_hh,
                     w->en_gt4_tra_gru_bih, w->en_gt4_tra_gru_bhh,
                     w->en_gt4_tra_fc_weight, w->en_gt4_tra_fc_bias,
                     s->en_tra_h4, h1_out, 8, FREQ_DOWN, scratch + 32 * FREQ_DOWN);

        /* Channel shuffle */
        channel_shuffle_stream(h1_out, x2, buf5, 8, FREQ_DOWN);
    }
    printf("\nAfter EnGT4 (16, 33):\n");
    printf("  Sum: %.6f (Python: 35.517010)\n", array_sum(buf5, 16 * FREQ_DOWN));

    /* Cleanup */
    free(feat);
    free(erb_out);
    free(sfe_out);
    free(en_conv0_out);
    free(en_conv1_out);
    free(buf4);
    free(buf5);
    free(scratch);
    free(stft_window);
    free(spec_real);
    free(spec_imag);
    free(output_frame);
    free(audio);
    gtcrn_destroy(model);

    return 0;
}
