#ifndef GTCRN_STREAMING_16K_H
#define GTCRN_STREAMING_16K_H

#include "gtcrn_model.h"
#include "stft_16k.h"

/**
 * GTCRN Streaming Processor for Real-Time Audio Enhancement - 16kHz Version
 *
 * This module provides real-time streaming inference for GTCRN model at 16kHz.
 * It manages all state caches and buffers needed for low-latency processing.
 *
 * Features:
 * - Frame-by-frame processing
 * - State caching for GRU, TRA, and convolution layers
 * - Overlap-add for STFT/iSTFT
 * - Configurable latency (default: ~32ms @ 16kHz)
 */

// ============================================================================
// State Cache Structures
// ============================================================================

/**
 * GRU state cache for streaming
 */
typedef struct {
    float* hidden_state;    // Hidden state (hidden_size,)
    int hidden_size;
} GRUCache_16k;

/**
 * TRA (Temporal Recurrent Attention) state cache
 */
typedef struct {
    float* gru_hidden;      // GRU hidden state (1, B, C)
    int channels;
} TRACache_16k;

/**
 * Convolution state cache for causal convolution
 */
typedef struct {
    float* buffer;          // Cached frames (C, cache_frames, F)
    int channels;
    int cache_frames;
    int freq_bins;
    int write_pos;          // Circular buffer write position
} ConvCache_16k;

/**
 * DPGRNN state cache
 */
typedef struct {
    // Intra RNN caches (bidirectional, so no state needed between frames)
    // Inter RNN caches (unidirectional, needs state)
    GRUCache_16k* inter_gru_g1_cache;
    GRUCache_16k* inter_gru_g2_cache;

    // FIXED: Add persistent inter_cache buffer to avoid static variables
    float* inter_cache_buffer;  // Persistent buffer for inter-RNN state (B*F*hidden_size)
    int inter_cache_size;       // Size of the buffer
} DPGRNNCache_16k;

/**
 * Skip connection buffers for encoder/decoder
 * FIXED: Manage skip connection memory properly
 */
typedef struct {
    float* data;
    int size;
} SkipBuffer_16k;

/**
 * Complete GTCRN streaming state for 16kHz
 */
typedef struct {
    // Model reference
    GTCRN* model;

    // STFT/iSTFT processors
    STFTParams_16k* stft_params;
    float* stft_input_buffer;
    float* istft_overlap_buffer;
    int stft_buffer_pos;

    // Encoder caches
    ConvCache_16k* encoder_conv1_cache;
    ConvCache_16k* encoder_conv2_cache;
    TRACache_16k* encoder_gtconv1_tra_cache;
    TRACache_16k* encoder_gtconv2_tra_cache;
    TRACache_16k* encoder_gtconv3_tra_cache;

    // DPGRNN caches
    DPGRNNCache_16k* dpgrnn1_cache;
    DPGRNNCache_16k* dpgrnn2_cache;

    // Decoder caches
    TRACache_16k* decoder_gtconv1_tra_cache;
    TRACache_16k* decoder_gtconv2_tra_cache;
    TRACache_16k* decoder_gtconv3_tra_cache;
    ConvCache_16k* decoder_conv1_cache;
    ConvCache_16k* decoder_conv2_cache;

    // FIXED: Skip connection buffers (persistent across encoder/decoder calls)
    SkipBuffer_16k skip_buffers[5];

    // Configuration
    int hop_length;         // STFT hop length (256 for 16kHz)
    int n_fft;              // STFT FFT size (512 for 16kHz)
    int sample_rate;        // Sample rate (16000)
    int chunk_size;         // Input chunk size (default: hop_length)

    // Statistics
    int frames_processed;
    float avg_latency_ms;
} GTCRNStreaming_16k;

// ============================================================================
// Cache Management Functions
// ============================================================================

/**
 * Create GRU cache
 */
GRUCache_16k* gru_cache_16k_create(int hidden_size);

/**
 * Free GRU cache
 */
void gru_cache_16k_free(GRUCache_16k* cache);

/**
 * Reset GRU cache to zero
 */
void gru_cache_16k_reset(GRUCache_16k* cache);

/**
 * Create TRA cache
 */
TRACache_16k* tra_cache_16k_create(int channels);

/**
 * Free TRA cache
 */
void tra_cache_16k_free(TRACache_16k* cache);

/**
 * Reset TRA cache
 */
void tra_cache_16k_reset(TRACache_16k* cache);

/**
 * Create convolution cache
 */
ConvCache_16k* conv_cache_16k_create(int channels, int cache_frames, int freq_bins);

/**
 * Free convolution cache
 */
void conv_cache_16k_free(ConvCache_16k* cache);

/**
 * Reset convolution cache
 */
void conv_cache_16k_reset(ConvCache_16k* cache);

/**
 * Create DPGRNN cache
 */
DPGRNNCache_16k* dpgrnn_cache_16k_create(int hidden_size, int batch_size, int freq_bins);

/**
 * Free DPGRNN cache
 */
void dpgrnn_cache_16k_free(DPGRNNCache_16k* cache);

/**
 * Reset DPGRNN cache
 */
void dpgrnn_cache_16k_reset(DPGRNNCache_16k* cache);

// ============================================================================
// Streaming Processor Functions
// ============================================================================

/**
 * Create GTCRN streaming processor for 16kHz
 *
 * @param model         GTCRN model (must be initialized with weights)
 * @param sample_rate   Sample rate (default: 16000)
 * @param chunk_size    Input chunk size in samples (default: 256)
 * @return              Streaming processor instance
 */
GTCRNStreaming_16k* gtcrn_streaming_16k_create(
    GTCRN* model,
    int sample_rate,
    int chunk_size
);

/**
 * Free streaming processor
 */
void gtcrn_streaming_16k_free(GTCRNStreaming_16k* stream);

/**
 * Reset all state caches
 * Call this when starting a new audio stream
 */
void gtcrn_streaming_16k_reset(GTCRNStreaming_16k* stream);

/**
 * Process one chunk of audio
 *
 * @param stream        Streaming processor
 * @param input         Input audio chunk (chunk_size samples)
 * @param output        Output audio chunk (chunk_size samples)
 * @return              0 on success, -1 on error
 *
 * Note: This function processes audio in chunks of size chunk_size.
 * For real-time processing, call this function repeatedly with new audio data.
 */
int gtcrn_streaming_16k_process_chunk(
    GTCRNStreaming_16k* stream,
    const float* input,
    float* output
);

/**
 * Process audio buffer (multiple chunks)
 *
 * @param stream        Streaming processor
 * @param input         Input audio buffer
 * @param output        Output audio buffer
 * @param num_samples   Number of samples to process
 * @return              Number of samples processed
 */
int gtcrn_streaming_16k_process_buffer(
    GTCRNStreaming_16k* stream,
    const float* input,
    float* output,
    int num_samples
);

/**
 * Get current latency in milliseconds
 */
float gtcrn_streaming_16k_get_latency_ms(GTCRNStreaming_16k* stream);

/**
 * Get processing statistics
 */
void gtcrn_streaming_16k_get_stats(
    GTCRNStreaming_16k* stream,
    int* frames_processed,
    float* avg_latency_ms
);

// ============================================================================
// Advanced Streaming Functions
// ============================================================================

/**
 * Process one STFT frame (for advanced users)
 *
 * @param stream        Streaming processor
 * @param spec_real     Input spectrum real part (freq_bins,)
 * @param spec_imag     Input spectrum imaginary part (freq_bins,)
 * @param out_real      Output spectrum real part (freq_bins,)
 * @param out_imag      Output spectrum imaginary part (freq_bins,)
 * @return              0 on success, -1 on error
 */
int gtcrn_streaming_16k_process_frame(
    GTCRNStreaming_16k* stream,
    const float* spec_real,
    const float* spec_imag,
    float* out_real,
    float* out_imag
);

/**
 * Flush remaining audio from internal buffers
 * Call this at the end of a stream
 */
int gtcrn_streaming_16k_flush(
    GTCRNStreaming_16k* stream,
    float* output,
    int* num_samples
);

// ============================================================================
// Optimized Streaming Functions (from gtcrn_streaming_optimized_16k.c)
// ============================================================================

/**
 * Process one frame with full state caching (optimized version)
 *
 * This is the optimized version that uses:
 * - gtconvblock_forward_stream() for GTConvBlocks
 * - dpgrnn_forward_stream() for DPGRNN
 * - Proper state caching for all components
 *
 * @param stream        Streaming processor
 * @param spec_real     Input spectrum real part (freq_bins,)
 * @param spec_imag     Input spectrum imaginary part (freq_bins,)
 * @param out_real      Output spectrum real part (freq_bins,)
 * @param out_imag      Output spectrum imaginary part (freq_bins,)
 * @return              0 on success, -1 on error
 */
int gtcrn_streaming_16k_process_frame_optimized(
    GTCRNStreaming_16k* stream,
    const float* spec_real,
    const float* spec_imag,
    float* out_real,
    float* out_imag
);

/**
 * Process audio chunk with optimized streaming
 *
 * Uses gtcrn_streaming_16k_process_frame_optimized() internally
 * with proper state caching for real-time processing
 *
 * @param stream        Streaming processor
 * @param input         Input audio chunk (chunk_size samples)
 * @param output        Output audio chunk (chunk_size samples)
 * @return              0 on success, -1 on error
 */
int gtcrn_streaming_16k_process_chunk_optimized(
    GTCRNStreaming_16k* stream,
    const float* input,
    float* output
);

#endif // GTCRN_STREAMING_16K_H
