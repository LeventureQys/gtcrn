#include "gtcrn_streaming_16k.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

// ============================================================================
// Cache Management Implementation for 16kHz
// ============================================================================

GRUCache_16k* gru_cache_16k_create(int hidden_size) {
    GRUCache_16k* cache = (GRUCache_16k*)malloc(sizeof(GRUCache_16k));
    if (!cache) return NULL;

    cache->hidden_size = hidden_size;
    cache->hidden_state = (float*)calloc(hidden_size, sizeof(float));

    return cache;
}

void gru_cache_16k_free(GRUCache_16k* cache) {
    if (cache) {
        free(cache->hidden_state);
        free(cache);
    }
}

void gru_cache_16k_reset(GRUCache_16k* cache) {
    if (cache) {
        memset(cache->hidden_state, 0, cache->hidden_size * sizeof(float));
    }
}

TRACache_16k* tra_cache_16k_create(int channels) {
    TRACache_16k* cache = (TRACache_16k*)malloc(sizeof(TRACache_16k));
    if (!cache) return NULL;

    cache->channels = channels;
    cache->gru_hidden = (float*)calloc(channels, sizeof(float));

    return cache;
}

void tra_cache_16k_free(TRACache_16k* cache) {
    if (cache) {
        free(cache->gru_hidden);
        free(cache);
    }
}

void tra_cache_16k_reset(TRACache_16k* cache) {
    if (cache) {
        memset(cache->gru_hidden, 0, cache->channels * sizeof(float));
    }
}

ConvCache_16k* conv_cache_16k_create(int channels, int cache_frames, int freq_bins) {
    ConvCache_16k* cache = (ConvCache_16k*)malloc(sizeof(ConvCache_16k));
    if (!cache) return NULL;

    cache->channels = channels;
    cache->cache_frames = cache_frames;
    cache->freq_bins = freq_bins;
    cache->write_pos = 0;
    cache->buffer = (float*)calloc(channels * cache_frames * freq_bins, sizeof(float));

    return cache;
}

void conv_cache_16k_free(ConvCache_16k* cache) {
    if (cache) {
        free(cache->buffer);
        free(cache);
    }
}

void conv_cache_16k_reset(ConvCache_16k* cache) {
    if (cache) {
        memset(cache->buffer, 0, cache->channels * cache->cache_frames * cache->freq_bins * sizeof(float));
        cache->write_pos = 0;
    }
}

DPGRNNCache_16k* dpgrnn_cache_16k_create(int hidden_size, int batch_size, int freq_bins) {
    DPGRNNCache_16k* cache = (DPGRNNCache_16k*)malloc(sizeof(DPGRNNCache_16k));
    if (!cache) return NULL;

    // Inter RNN needs state caching (unidirectional)
    cache->inter_gru_g1_cache = gru_cache_16k_create(hidden_size / 2);
    cache->inter_gru_g2_cache = gru_cache_16k_create(hidden_size / 2);

    // FIXED: Allocate persistent inter_cache buffer to avoid static variables
    cache->inter_cache_size = batch_size * freq_bins * hidden_size;
    cache->inter_cache_buffer = (float*)calloc(cache->inter_cache_size, sizeof(float));

    return cache;
}

void dpgrnn_cache_16k_free(DPGRNNCache_16k* cache) {
    if (cache) {
        gru_cache_16k_free(cache->inter_gru_g1_cache);
        gru_cache_16k_free(cache->inter_gru_g2_cache);
        // FIXED: Free the persistent inter_cache buffer
        free(cache->inter_cache_buffer);
        free(cache);
    }
}

void dpgrnn_cache_16k_reset(DPGRNNCache_16k* cache) {
    if (cache) {
        gru_cache_16k_reset(cache->inter_gru_g1_cache);
        gru_cache_16k_reset(cache->inter_gru_g2_cache);
        // FIXED: Reset the persistent inter_cache buffer
        if (cache->inter_cache_buffer) {
            memset(cache->inter_cache_buffer, 0, cache->inter_cache_size * sizeof(float));
        }
    }
}

// ============================================================================
// Streaming Processor Implementation for 16kHz
// ============================================================================

GTCRNStreaming_16k* gtcrn_streaming_16k_create(
    GTCRN* model,
    int sample_rate,
    int chunk_size
) {
    if (!model) {
        fprintf(stderr, "Error: Model is NULL\n");
        return NULL;
    }

    GTCRNStreaming_16k* stream = (GTCRNStreaming_16k*)malloc(sizeof(GTCRNStreaming_16k));
    if (!stream) return NULL;

    stream->model = model;
    stream->sample_rate = sample_rate;
    stream->n_fft = 512;        // Changed from 1536 to 512 for 16kHz
    stream->hop_length = 256;   // Changed from 768 to 256 for 16kHz
    stream->chunk_size = chunk_size > 0 ? chunk_size : stream->hop_length;

    // Create STFT processor for 16kHz
    stream->stft_params = stft_16k_create(stream->n_fft, stream->hop_length, sample_rate);

    // Allocate STFT buffers
    stream->stft_input_buffer = (float*)calloc(stream->n_fft, sizeof(float));
    stream->istft_overlap_buffer = (float*)calloc(stream->n_fft, sizeof(float));
    stream->stft_buffer_pos = 0;

    // Create encoder caches
    // Conv1: kernel=(1,5), dilation=(1,1) -> cache_frames = (5-1)*1 = 4
    stream->encoder_conv1_cache = conv_cache_16k_create(16, 4, 385);

    // Conv2: kernel=(1,5), dilation=(1,1) -> cache_frames = 4
    stream->encoder_conv2_cache = conv_cache_16k_create(16, 4, 193);

    // GTConv TRA caches
    stream->encoder_gtconv1_tra_cache = tra_cache_16k_create(8);  // 16/2 = 8
    stream->encoder_gtconv2_tra_cache = tra_cache_16k_create(8);
    stream->encoder_gtconv3_tra_cache = tra_cache_16k_create(8);

    // Create DPGRNN caches (FIXED: pass batch_size=1 and freq_bins=97 for streaming)
    stream->dpgrnn1_cache = dpgrnn_cache_16k_create(16, 1, 97);
    stream->dpgrnn2_cache = dpgrnn_cache_16k_create(16, 1, 97);

    // Create decoder caches
    stream->decoder_gtconv1_tra_cache = tra_cache_16k_create(8);
    stream->decoder_gtconv2_tra_cache = tra_cache_16k_create(8);
    stream->decoder_gtconv3_tra_cache = tra_cache_16k_create(8);
    stream->decoder_conv1_cache = conv_cache_16k_create(16, 4, 97);
    stream->decoder_conv2_cache = conv_cache_16k_create(2, 4, 193);

    // FIXED: Initialize skip connection buffers
    // Skip buffers for encoder layers: [layer1, layer2, layer3, layer4, encoder_out]
    int skip_sizes[5] = {
        1 * 16 * 1 * 193,  // layer1: (1, 16, 1, 193)
        1 * 16 * 1 * 97,   // layer2: (1, 16, 1, 97)
        1 * 16 * 1 * 97,   // layer3: (1, 16, 1, 97)
        1 * 16 * 1 * 97,   // layer4: (1, 16, 1, 97)
        1 * 16 * 1 * 97    // encoder_out: (1, 16, 1, 97)
    };

    for (int i = 0; i < 5; i++) {
        stream->skip_buffers[i].size = skip_sizes[i];
        stream->skip_buffers[i].data = (float*)calloc(skip_sizes[i], sizeof(float));
    }

    // Initialize statistics
    stream->frames_processed = 0;
    stream->avg_latency_ms = 0.0f;

    printf("GTCRN Streaming 16kHz created:\n");
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Chunk size: %d samples\n", stream->chunk_size);
    printf("  FFT size: %d\n", stream->n_fft);
    printf("  Hop length: %d\n", stream->hop_length);
    printf("  Latency: ~%.1f ms\n", (float)stream->n_fft / sample_rate * 1000);

    return stream;
}

void gtcrn_streaming_16k_free(GTCRNStreaming_16k* stream) {
    if (!stream) return;

    // Free STFT
    stft_16k_free(stream->stft_params);
    free(stream->stft_input_buffer);
    free(stream->istft_overlap_buffer);

    // Free encoder caches
    conv_cache_16k_free(stream->encoder_conv1_cache);
    conv_cache_16k_free(stream->encoder_conv2_cache);
    tra_cache_16k_free(stream->encoder_gtconv1_tra_cache);
    tra_cache_16k_free(stream->encoder_gtconv2_tra_cache);
    tra_cache_16k_free(stream->encoder_gtconv3_tra_cache);

    // Free DPGRNN caches
    dpgrnn_cache_16k_free(stream->dpgrnn1_cache);
    dpgrnn_cache_16k_free(stream->dpgrnn2_cache);

    // Free decoder caches
    tra_cache_16k_free(stream->decoder_gtconv1_tra_cache);
    tra_cache_16k_free(stream->decoder_gtconv2_tra_cache);
    tra_cache_16k_free(stream->decoder_gtconv3_tra_cache);
    conv_cache_16k_free(stream->decoder_conv1_cache);
    conv_cache_16k_free(stream->decoder_conv2_cache);

    // FIXED: Free skip connection buffers
    for (int i = 0; i < 5; i++) {
        free(stream->skip_buffers[i].data);
    }

    free(stream);
}

void gtcrn_streaming_16k_reset(GTCRNStreaming_16k* stream) {
    if (!stream) return;

    // Reset STFT buffers
    memset(stream->stft_input_buffer, 0, stream->n_fft * sizeof(float));
    memset(stream->istft_overlap_buffer, 0, stream->n_fft * sizeof(float));
    stream->stft_buffer_pos = 0;

    // Reset encoder caches
    conv_cache_16k_reset(stream->encoder_conv1_cache);
    conv_cache_16k_reset(stream->encoder_conv2_cache);
    tra_cache_16k_reset(stream->encoder_gtconv1_tra_cache);
    tra_cache_16k_reset(stream->encoder_gtconv2_tra_cache);
    tra_cache_16k_reset(stream->encoder_gtconv3_tra_cache);

    // Reset DPGRNN caches
    dpgrnn_cache_16k_reset(stream->dpgrnn1_cache);
    dpgrnn_cache_16k_reset(stream->dpgrnn2_cache);

    // Reset decoder caches
    tra_cache_16k_reset(stream->decoder_gtconv1_tra_cache);
    tra_cache_16k_reset(stream->decoder_gtconv2_tra_cache);
    tra_cache_16k_reset(stream->decoder_gtconv3_tra_cache);
    conv_cache_16k_reset(stream->decoder_conv1_cache);
    conv_cache_16k_reset(stream->decoder_conv2_cache);

    // Reset statistics
    stream->frames_processed = 0;
    stream->avg_latency_ms = 0.0f;

    printf("GTCRN Streaming 16kHz reset\n");
}

int gtcrn_streaming_16k_process_frame(
    GTCRNStreaming_16k* stream,
    const float* spec_real,
    const float* spec_imag,
    float* out_real,
    float* out_imag
) {
    if (!stream || !stream->model) return -1;

    clock_t start = clock();

    // For now, use the batch processing version
    // In a complete implementation, this would use cached states
    // and process frame-by-frame

    int freq_bins = stream->n_fft / 2 + 1;  // 257 for 16kHz

    // Prepare input tensor (B=1, F=257, T=1, 2)
    float* spec_input = (float*)malloc(freq_bins * 2 * sizeof(float));
    for (int f = 0; f < freq_bins; f++) {
        spec_input[f * 2] = spec_real[f];
        spec_input[f * 2 + 1] = spec_imag[f];
    }

    // Allocate output
    float* spec_output = (float*)malloc(freq_bins * 2 * sizeof(float));

    // Run GTCRN forward pass
    gtcrn_forward(spec_input, spec_output, 1, freq_bins, 1, stream->model);

    // Extract output
    for (int f = 0; f < freq_bins; f++) {
        out_real[f] = spec_output[f * 2];
        out_imag[f] = spec_output[f * 2 + 1];
    }

    free(spec_input);
    free(spec_output);

    // Update statistics
    clock_t end = clock();
    float latency_ms = (float)(end - start) / CLOCKS_PER_SEC * 1000;
    stream->avg_latency_ms = (stream->avg_latency_ms * stream->frames_processed + latency_ms) /
                             (stream->frames_processed + 1);
    stream->frames_processed++;

    return 0;
}

int gtcrn_streaming_16k_process_chunk(
    GTCRNStreaming_16k* stream,
    const float* input,
    float* output
) {
    if (!stream || !input || !output) return -1;

    int chunk_size = stream->chunk_size;
    int hop_length = stream->hop_length;
    int n_fft = stream->n_fft;
    int freq_bins = n_fft / 2 + 1;

    // Add input to STFT buffer
    for (int i = 0; i < chunk_size; i++) {
        stream->stft_input_buffer[stream->stft_buffer_pos] = input[i];
        stream->stft_buffer_pos++;

        // Process when we have enough samples
        if (stream->stft_buffer_pos >= n_fft) {
            // Perform STFT
            float* spec_real = (float*)malloc(freq_bins * sizeof(float));
            float* spec_imag = (float*)malloc(freq_bins * sizeof(float));

            stft_16k_forward(stream->stft_input_buffer, n_fft, spec_real, spec_imag, stream->stft_params);

            // Process frame through GTCRN
            float* out_real = (float*)malloc(freq_bins * sizeof(float));
            float* out_imag = (float*)malloc(freq_bins * sizeof(float));

            gtcrn_streaming_16k_process_frame(stream, spec_real, spec_imag, out_real, out_imag);

            // Perform iSTFT
            float* frame_audio = (float*)malloc(n_fft * sizeof(float));
            istft_16k_forward(out_real, out_imag, 1, frame_audio, stream->stft_params);

            // Overlap-add
            for (int j = 0; j < hop_length && (i - chunk_size + j) < chunk_size; j++) {
                output[i - chunk_size + j] += frame_audio[j];
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

int gtcrn_streaming_16k_process_buffer(
    GTCRNStreaming_16k* stream,
    const float* input,
    float* output,
    int num_samples
) {
    if (!stream || !input || !output) return -1;

    int chunk_size = stream->chunk_size;
    int processed = 0;

    // Initialize output to zero
    memset(output, 0, num_samples * sizeof(float));

    // Process in chunks
    while (processed < num_samples) {
        int remaining = num_samples - processed;
        int current_chunk = remaining < chunk_size ? remaining : chunk_size;

        gtcrn_streaming_16k_process_chunk(
            stream,
            input + processed,
            output + processed
        );

        processed += current_chunk;
    }

    return processed;
}

float gtcrn_streaming_16k_get_latency_ms(GTCRNStreaming_16k* stream) {
    if (!stream) return 0.0f;

    // Algorithmic latency: STFT window size
    float algo_latency = (float)stream->n_fft / stream->sample_rate * 1000;

    // Processing latency: average inference time
    float proc_latency = stream->avg_latency_ms;

    return algo_latency + proc_latency;
}

void gtcrn_streaming_16k_get_stats(
    GTCRNStreaming_16k* stream,
    int* frames_processed,
    float* avg_latency_ms
) {
    if (!stream) return;

    if (frames_processed) *frames_processed = stream->frames_processed;
    if (avg_latency_ms) *avg_latency_ms = stream->avg_latency_ms;
}

int gtcrn_streaming_16k_flush(
    GTCRNStreaming_16k* stream,
    float* output,
    int* num_samples
) {
    if (!stream || !output || !num_samples) return -1;

    // Process remaining samples in buffer
    int remaining = stream->stft_buffer_pos;
    if (remaining > 0) {
        // Pad with zeros
        memset(stream->stft_input_buffer + remaining, 0,
               (stream->n_fft - remaining) * sizeof(float));

        // Process final frame
        gtcrn_streaming_16k_process_chunk(stream, stream->stft_input_buffer, output);

        *num_samples = remaining;
        stream->stft_buffer_pos = 0;

        return 0;
    }

    *num_samples = 0;
    return 0;
}
