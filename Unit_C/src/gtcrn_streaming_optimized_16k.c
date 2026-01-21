/**
 * gtcrn_streaming_optimized_16k.c - 16kHz Version
 *
 * Optimized streaming implementation for 16kHz audio
 * Key changes from 48kHz version:
 * - FFT size: 1536 -> 512
 * - Hop length: 768 -> 256
 * - Freq bins: 769 -> 257
 * - Chunk size: 768 -> 256
 */

#include "gtcrn_streaming_16k.h"
#include "gtcrn_model.h"
#include "GRU.h"
#include "stream_conv.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// Forward declarations from gtcrn_streaming_impl.c
extern void dpgrnn_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* inter_cache,
    DPGRNN* dpgrnn
);

extern void gtconvblock_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,
    float* tra_cache,
    GTConvBlock* block,
    int kernel_h,
    int dilation_h
);

// ============================================================================
// FIXED: Encoder with proper skip connection management
// ============================================================================

/**
 * Process one frame through encoder with state caching
 * FIXED: Uses persistent skip buffers from GTCRNStreaming_16k
 */
static int encoder_forward_streaming_16k(
    const Tensor* input,        // (1, 9, 1, 385) - single frame
    Tensor* output,             // (1, 16, 1, 97)
    GTCRNStreaming_16k* stream, // FIXED: Pass stream to access skip_buffers
    Encoder* encoder
) {
    int B = input->shape.batch;
    int T = input->shape.height;

    // FIXED: Use persistent skip buffers from stream instead of local allocation
    // This ensures memory is valid when decoder accesses it
    Tensor layer1_out = {
        .data = stream->skip_buffers[0].data,
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    Tensor layer2_out = {
        .data = stream->skip_buffers[1].data,
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer3_out = {
        .data = stream->skip_buffers[2].data,
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer4_out = {
        .data = stream->skip_buffers[3].data,
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    // Layer 1: ConvBlock
    if (encoder->conv1) {
        convblock_forward(input, &layer1_out, encoder->conv1);
    } else {
        memset(layer1_out.data, 0, B * 16 * T * 193 * sizeof(float));
    }

    // Layer 2: ConvBlock
    if (encoder->conv2) {
        convblock_forward(&layer1_out, &layer2_out, encoder->conv2);
    } else {
        memset(layer2_out.data, 0, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 3: GTConvBlock (dilation=1)
    if (encoder->gtconv1 && stream->encoder_gtconv1_tra_cache) {
        gtconvblock_forward_stream(
            &layer2_out, &layer3_out,
            stream->encoder_conv1_cache ? stream->encoder_conv1_cache->buffer : NULL,
            stream->encoder_gtconv1_tra_cache->gru_hidden,
            encoder->gtconv1,
            3, 1
        );
    } else {
        memcpy(layer3_out.data, layer2_out.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 4: GTConvBlock (dilation=2)
    if (encoder->gtconv2 && stream->encoder_gtconv2_tra_cache) {
        gtconvblock_forward_stream(
            &layer3_out, &layer4_out,
            stream->encoder_conv2_cache ? stream->encoder_conv2_cache->buffer : NULL,
            stream->encoder_gtconv2_tra_cache->gru_hidden,
            encoder->gtconv2,
            3, 2
        );
    } else {
        memcpy(layer4_out.data, layer3_out.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 5: GTConvBlock (dilation=5) - output directly to encoder_out
    // Use skip_buffers[4] for encoder output
    Tensor encoder_out_tensor = {
        .data = stream->skip_buffers[4].data,
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    if (encoder->gtconv3 && stream->encoder_gtconv3_tra_cache) {
        gtconvblock_forward_stream(
            &layer4_out, &encoder_out_tensor,
            stream->encoder_conv2_cache ? stream->encoder_conv2_cache->buffer : NULL,
            stream->encoder_gtconv3_tra_cache->gru_hidden,
            encoder->gtconv3,
            3, 5
        );
    } else {
        memcpy(encoder_out_tensor.data, layer4_out.data, B * 16 * T * 97 * sizeof(float));
    }

    // Copy to output
    memcpy(output->data, encoder_out_tensor.data, B * 16 * T * 97 * sizeof(float));

    // FIXED: No need to free anything - using persistent buffers
    return 0;
}

// ============================================================================
// FIXED: DPGRNN with proper cache management
// ============================================================================

/**
 * Process one frame through DPGRNN with state caching
 * FIXED: Uses persistent cache buffer from DPGRNNCache_16k instead of static variable
 */
static int dpgrnn_forward_streaming_wrapper_16k(
    const Tensor* input,
    Tensor* output,
    DPGRNN* dpgrnn,
    DPGRNNCache_16k* cache
) {
    if (!cache || !cache->inter_gru_g1_cache || !cache->inter_gru_g2_cache) {
        fprintf(stderr, "Error: DPGRNN cache not initialized\n");
        return -1;
    }

    if (!cache->inter_cache_buffer) {
        fprintf(stderr, "Error: DPGRNN inter_cache_buffer not allocated\n");
        return -1;
    }

    // FIXED: Use persistent buffer from cache instead of static variable
    // This allows multiple streaming instances and is thread-safe per instance
    dpgrnn_forward_stream(input, output, cache->inter_cache_buffer, dpgrnn);

    return 0;
}

// ============================================================================
// FIXED: Decoder with proper skip connection access
// ============================================================================

/**
 * Process one frame through decoder with state caching
 * FIXED: Accesses skip connections from persistent buffers
 */
static int decoder_forward_streaming_16k(
    const Tensor* input,
    GTCRNStreaming_16k* stream,
    Tensor* output,
    Decoder* decoder
) {
    int B = input->shape.batch;
    int T = input->shape.height;

    // Allocate temporary buffers for intermediate layers
    Tensor layer1_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer1_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer2_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer2_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer3_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer3_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer4_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer4_out = {
        .data = (float*)malloc(B * 16 * T * 193 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    Tensor layer5_in = {
        .data = (float*)malloc(B * 16 * T * 193 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    // FIXED: Access skip connections from persistent buffers
    // skip_buffers[4] = encoder_out, skip_buffers[3] = layer4, etc.

    // Layer 1: GTConvBlock (dilation=5) + skip[4]
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer1_in.data[i] = input->data[i] + stream->skip_buffers[4].data[i];
    }

    if (decoder->gtconv1 && stream->decoder_gtconv1_tra_cache) {
        gtconvblock_forward_stream(
            &layer1_in, &layer1_out,
            stream->decoder_conv1_cache ? stream->decoder_conv1_cache->buffer : NULL,
            stream->decoder_gtconv1_tra_cache->gru_hidden,
            decoder->gtconv1,
            3, 5
        );
    } else {
        memcpy(layer1_out.data, layer1_in.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 2: GTConvBlock (dilation=2) + skip[3]
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer2_in.data[i] = layer1_out.data[i] + stream->skip_buffers[3].data[i];
    }

    if (decoder->gtconv2 && stream->decoder_gtconv2_tra_cache) {
        gtconvblock_forward_stream(
            &layer2_in, &layer2_out,
            stream->decoder_conv1_cache ? stream->decoder_conv1_cache->buffer : NULL,
            stream->decoder_gtconv2_tra_cache->gru_hidden,
            decoder->gtconv2,
            3, 2
        );
    } else {
        memcpy(layer2_out.data, layer2_in.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 3: GTConvBlock (dilation=1) + skip[2]
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer3_in.data[i] = layer2_out.data[i] + stream->skip_buffers[2].data[i];
    }

    if (decoder->gtconv3 && stream->decoder_gtconv3_tra_cache) {
        gtconvblock_forward_stream(
            &layer3_in, &layer3_out,
            stream->decoder_conv1_cache ? stream->decoder_conv1_cache->buffer : NULL,
            stream->decoder_gtconv3_tra_cache->gru_hidden,
            decoder->gtconv3,
            3, 1
        );
    } else {
        memcpy(layer3_out.data, layer3_in.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 4: ConvBlock (deconv) + skip[1]
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer4_in.data[i] = layer3_out.data[i] + stream->skip_buffers[1].data[i];
    }

    if (decoder->conv1) {
        convblock_forward(&layer4_in, &layer4_out, decoder->conv1);
    } else {
        memset(layer4_out.data, 0, B * 16 * T * 193 * sizeof(float));
    }

    // Layer 5: ConvBlock (deconv, tanh) + skip[0]
    for (int i = 0; i < B * 16 * T * 193; i++) {
        layer5_in.data[i] = layer4_out.data[i] + stream->skip_buffers[0].data[i];
    }

    if (decoder->conv2) {
        convblock_forward(&layer5_in, output, decoder->conv2);
    } else {
        memset(output->data, 0, B * 2 * T * 385 * sizeof(float));
    }

    // Clean up temporary buffers
    free(layer1_in.data);
    free(layer1_out.data);
    free(layer2_in.data);
    free(layer2_out.data);
    free(layer3_in.data);
    free(layer3_out.data);
    free(layer4_in.data);
    free(layer4_out.data);
    free(layer5_in.data);

    return 0;
}

// ============================================================================
// Optimized Streaming Interface for 16kHz
// ============================================================================

/**
 * Process one frame with full state caching (optimized version for 16kHz)
 * FIXED: Proper memory management for skip connections
 */
int gtcrn_streaming_16k_process_frame_optimized(
    GTCRNStreaming_16k* stream,
    const float* spec_real,
    const float* spec_imag,
    float* out_real,
    float* out_imag
) {
    if (!stream || !stream->model) return -1;

    clock_t start = clock();

    int freq_bins = stream->n_fft / 2 + 1;  // 257 for 16kHz
    int B = 1;
    int T = 1;

    // ========================================================================
    // 1. Input preprocessing
    // ========================================================================

    float* spec_mag = (float*)malloc(freq_bins * sizeof(float));
    for (int f = 0; f < freq_bins; f++) {
        float real = spec_real[f];
        float imag = spec_imag[f];
        spec_mag[f] = sqrtf(real * real + imag * imag + 1e-12f);
    }

    float* feat = (float*)malloc(B * 3 * T * freq_bins * sizeof(float));
    for (int f = 0; f < freq_bins; f++) {
        feat[0 * freq_bins + f] = spec_mag[f];
        feat[1 * freq_bins + f] = spec_real[f];
        feat[2 * freq_bins + f] = spec_imag[f];
    }

    // ========================================================================
    // 2. ERB compression
    // ========================================================================

    Tensor feat_tensor = {
        .data = feat,
        .shape = {.batch = B, .channels = 3, .height = T, .width = freq_bins}
    };

    Tensor erb_tensor = {
        .data = (float*)malloc(B * 3 * T * 385 * sizeof(float)),
        .shape = {.batch = B, .channels = 3, .height = T, .width = 385}
    };

    if (stream->model->erb) {
        erb_compress(&feat_tensor, &erb_tensor, stream->model->erb);
    } else {
        memcpy(erb_tensor.data, feat, B * 3 * T * 385 * sizeof(float));
    }

    // ========================================================================
    // 3. SFE
    // ========================================================================

    Tensor sfe_tensor = {
        .data = (float*)malloc(B * 9 * T * 385 * sizeof(float)),
        .shape = {.batch = B, .channels = 9, .height = T, .width = 385}
    };

    if (stream->model->sfe) {
        sfe_forward(&erb_tensor, &sfe_tensor, stream->model->sfe);
    }

    // ========================================================================
    // 4. Encoder (FIXED: uses persistent skip buffers)
    // ========================================================================

    Tensor encoder_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    if (stream->model->encoder) {
        encoder_forward_streaming_16k(&sfe_tensor, &encoder_out, stream, stream->model->encoder);
    }

    // ========================================================================
    // 5. DPGRNN (FIXED: uses persistent cache buffer)
    // ========================================================================

    Tensor dpgrnn1_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor dpgrnn2_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    if (stream->model->dpgrnn1 && stream->dpgrnn1_cache) {
        dpgrnn_forward_streaming_wrapper_16k(&encoder_out, &dpgrnn1_out,
                                        stream->model->dpgrnn1, stream->dpgrnn1_cache);
    } else {
        memcpy(dpgrnn1_out.data, encoder_out.data, B * 16 * T * 97 * sizeof(float));
    }

    if (stream->model->dpgrnn2 && stream->dpgrnn2_cache) {
        dpgrnn_forward_streaming_wrapper_16k(&dpgrnn1_out, &dpgrnn2_out,
                                        stream->model->dpgrnn2, stream->dpgrnn2_cache);
    } else {
        memcpy(dpgrnn2_out.data, dpgrnn1_out.data, B * 16 * T * 97 * sizeof(float));
    }

    // ========================================================================
    // 6. Decoder (FIXED: accesses persistent skip buffers)
    // ========================================================================

    Tensor decoder_out = {
        .data = (float*)malloc(B * 2 * T * 385 * sizeof(float)),
        .shape = {.batch = B, .channels = 2, .height = T, .width = 385}
    };

    if (stream->model->decoder) {
        decoder_forward_streaming_16k(&dpgrnn2_out, stream, &decoder_out, stream->model->decoder);
    }

    // ========================================================================
    // 7. ERB decompression
    // ========================================================================

    Tensor mask_tensor = {
        .data = (float*)malloc(B * 2 * T * freq_bins * sizeof(float)),
        .shape = {.batch = B, .channels = 2, .height = T, .width = freq_bins}
    };

    if (stream->model->erb) {
        erb_decompress(&decoder_out, &mask_tensor, stream->model->erb);
    }

    // ========================================================================
    // 8. Apply complex mask
    // ========================================================================

    for (int f = 0; f < freq_bins; f++) {
        float mask_r = mask_tensor.data[0 * freq_bins + f];
        float mask_i = mask_tensor.data[1 * freq_bins + f];
        float spec_r = spec_real[f];
        float spec_i = spec_imag[f];

        out_real[f] = spec_r * mask_r - spec_i * mask_i;
        out_imag[f] = spec_i * mask_r + spec_r * mask_i;
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    free(spec_mag);
    free(feat);
    free(erb_tensor.data);
    free(sfe_tensor.data);
    free(encoder_out.data);
    free(dpgrnn1_out.data);
    free(dpgrnn2_out.data);
    free(decoder_out.data);
    free(mask_tensor.data);

    // Update statistics
    clock_t end = clock();
    float latency_ms = (float)(end - start) / CLOCKS_PER_SEC * 1000;
    stream->avg_latency_ms = (stream->avg_latency_ms * stream->frames_processed + latency_ms) /
                             (stream->frames_processed + 1);
    stream->frames_processed++;

    return 0;
}

/**
 * Process audio chunk with optimized streaming for 16kHz
 */
int gtcrn_streaming_16k_process_chunk_optimized(
    GTCRNStreaming_16k* stream,
    const float* input,
    float* output
) {
    if (!stream || !input || !output) return -1;

    int chunk_size = stream->chunk_size;
    int hop_length = stream->hop_length;
    int n_fft = stream->n_fft;
    int freq_bins = n_fft / 2 + 1;

    // Initialize output
    memset(output, 0, chunk_size * sizeof(float));

    // Add input to STFT buffer
    for (int i = 0; i < chunk_size; i++) {
        stream->stft_input_buffer[stream->stft_buffer_pos++] = input[i];

        // Process when we have enough samples
        if (stream->stft_buffer_pos >= n_fft) {
            float* spec_real = (float*)malloc(freq_bins * sizeof(float));
            float* spec_imag = (float*)malloc(freq_bins * sizeof(float));
            float* out_real = (float*)malloc(freq_bins * sizeof(float));
            float* out_imag = (float*)malloc(freq_bins * sizeof(float));

            // Perform STFT
            stft_16k_forward(stream->stft_input_buffer, n_fft, spec_real, spec_imag,
                        stream->stft_params);

            // Process frame through GTCRN
            gtcrn_streaming_16k_process_frame_optimized(stream, spec_real, spec_imag,
                                                   out_real, out_imag);

            // Perform iSTFT
            float* frame_audio = (float*)malloc(n_fft * sizeof(float));
            istft_16k_forward(out_real, out_imag, 1, frame_audio, stream->stft_params);

            // Overlap-add
            for (int j = 0; j < hop_length && (i - hop_length + j) >= 0 && (i - hop_length + j) < chunk_size; j++) {
                output[i - hop_length + j] += frame_audio[j];
            }

            // Shift buffer
            memmove(stream->stft_input_buffer,
                    stream->stft_input_buffer + hop_length,
                    (n_fft - hop_length) * sizeof(float));
            stream->stft_buffer_pos -= hop_length;

            free(spec_real);
            free(spec_imag);
            free(out_real);
            free(out_imag);
            free(frame_audio);
        }
    }

    return 0;
}
