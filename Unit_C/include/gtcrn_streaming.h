#ifndef GTCRN_STREAMING_H
#define GTCRN_STREAMING_H

#include "gtcrn_model.h"
#include "stft.h"

/**
 * GTCRN Streaming Processor for Real-Time Audio Enhancement
 *
 * This module provides real-time streaming inference for GTCRN model.
 * It manages all state caches and buffers needed for low-latency processing.
 *
 * Features:
 * - Frame-by-frame processing
 * - State caching for GRU, TRA, and convolution layers
 * - Overlap-add for STFT/iSTFT
 * - Configurable latency (default: ~32ms @ 48kHz)
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
} GRUCache;

/**
 * TRA (Temporal Recurrent Attention) state cache
 */
typedef struct {
    float* gru_hidden;      // GRU hidden state (1, B, C)
    int channels;
} TRACache;

/**
 * Convolution state cache for causal convolution
 */
typedef struct {
    float* buffer;          // Cached frames (C, cache_frames, F)
    int channels;
    int cache_frames;
    int freq_bins;
    int write_pos;          // Circular buffer write position
} ConvCache;

/**
 * DPGRNN state cache
 */
typedef struct {
    // Intra RNN caches (bidirectional, so no state needed between frames)
    // Inter RNN caches (unidirectional, needs state)
    GRUCache* inter_gru_g1_cache;
    GRUCache* inter_gru_g2_cache;

    // FIXED: Add persistent inter_cache buffer to avoid static variables
    float* inter_cache_buffer;  // Persistent buffer for inter-RNN state (B*F*hidden_size)
    int inter_cache_size;       // Size of the buffer
} DPGRNNCache;

/**
 * Skip connection buffers for encoder/decoder
 * FIXED: Manage skip connection memory properly
 */
typedef struct {
    float* data;
    int size;
} SkipBuffer;

/**
 * Complete GTCRN streaming state
 */
typedef struct {
    // Model reference
    GTCRN* model;

    // STFT/iSTFT processors
    STFTParams* stft_params;
    float* stft_input_buffer;
    float* istft_overlap_buffer;
    int stft_buffer_pos;

    // Encoder caches
    ConvCache* encoder_conv1_cache;
    ConvCache* encoder_conv2_cache;
    TRACache* encoder_gtconv1_tra_cache;
    TRACache* encoder_gtconv2_tra_cache;
    TRACache* encoder_gtconv3_tra_cache;

    // DPGRNN caches
    DPGRNNCache* dpgrnn1_cache;
    DPGRNNCache* dpgrnn2_cache;

    // Decoder caches
    TRACache* decoder_gtconv1_tra_cache;
    TRACache* decoder_gtconv2_tra_cache;
    TRACache* decoder_gtconv3_tra_cache;
    ConvCache* decoder_conv1_cache;
    ConvCache* decoder_conv2_cache;

    // FIXED: Skip connection buffers (persistent across encoder/decoder calls)
    SkipBuffer skip_buffers[5];

    // Configuration
    int hop_length;         // STFT hop length (768 for 48kHz)
    int n_fft;              // STFT FFT size (1536 for 48kHz)
    int sample_rate;        // Sample rate (48000)
    int chunk_size;         // Input chunk size (default: hop_length)

    // Statistics
    int frames_processed;
    float avg_latency_ms;
} GTCRNStreaming;

// ============================================================================
// Cache Management Functions
// ============================================================================

/**
 * Create GRU cache
 */
GRUCache* gru_cache_create(int hidden_size);

/**
 * Free GRU cache
 */
void gru_cache_free(GRUCache* cache);

/**
 * Reset GRU cache to zero
 */
void gru_cache_reset(GRUCache* cache);

/**
 * Create TRA cache
 */
TRACache* tra_cache_create(int channels);

/**
 * Free TRA cache
 */
void tra_cache_free(TRACache* cache);

/**
 * Reset TRA cache
 */
void tra_cache_reset(TRACache* cache);

/**
 * Create convolution cache
 */
ConvCache* conv_cache_create(int channels, int cache_frames, int freq_bins);

/**
 * Free convolution cache
 */
void conv_cache_free(ConvCache* cache);

/**
 * Reset convolution cache
 */
void conv_cache_reset(ConvCache* cache);

/**
 * Create DPGRNN cache
 */
DPGRNNCache* dpgrnn_cache_create(int hidden_size, int batch_size, int freq_bins);

/**
 * Free DPGRNN cache
 */
void dpgrnn_cache_free(DPGRNNCache* cache);

/**
 * Reset DPGRNN cache
 */
void dpgrnn_cache_reset(DPGRNNCache* cache);

// ============================================================================
// Streaming Processor Functions
// ============================================================================

/**
 * Create GTCRN streaming processor
 *
 * @param model         GTCRN model (must be initialized with weights)
 * @param sample_rate   Sample rate (default: 48000)
 * @param chunk_size    Input chunk size in samples (default: 768)
 * @return              Streaming processor instance
 */
GTCRNStreaming* gtcrn_streaming_create(
    GTCRN* model,
    int sample_rate,
    int chunk_size
);

/**
 * Free streaming processor
 */
void gtcrn_streaming_free(GTCRNStreaming* stream);

/**
 * Reset all state caches
 * Call this when starting a new audio stream
 */
void gtcrn_streaming_reset(GTCRNStreaming* stream);

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
int gtcrn_streaming_process_chunk(
    GTCRNStreaming* stream,
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
int gtcrn_streaming_process_buffer(
    GTCRNStreaming* stream,
    const float* input,
    float* output,
    int num_samples
);

/**
 * Get current latency in milliseconds
 */
float gtcrn_streaming_get_latency_ms(GTCRNStreaming* stream);

/**
 * Get processing statistics
 */
void gtcrn_streaming_get_stats(
    GTCRNStreaming* stream,
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
int gtcrn_streaming_process_frame(
    GTCRNStreaming* stream,
    const float* spec_real,
    const float* spec_imag,
    float* out_real,
    float* out_imag
);

/**
 * Flush remaining audio from internal buffers
 * Call this at the end of a stream
 */
int gtcrn_streaming_flush(
    GTCRNStreaming* stream,
    float* output,
    int* num_samples
);

// ============================================================================
// Optimized Streaming Functions (from gtcrn_streaming_optimized.c)
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
int gtcrn_streaming_process_frame_optimized(
    GTCRNStreaming* stream,
    const float* spec_real,
    const float* spec_imag,
    float* out_real,
    float* out_imag
);

/**
 * Process audio chunk with optimized streaming
 *
 * Uses gtcrn_streaming_process_frame_optimized() internally
 * with proper state caching for real-time processing
 *
 * @param stream        Streaming processor
 * @param input         Input audio chunk (chunk_size samples)
 * @param output        Output audio chunk (chunk_size samples)
 * @return              0 on success, -1 on error
 */
int gtcrn_streaming_process_chunk_optimized(
    GTCRNStreaming* stream,
    const float* input,
    float* output
);

#endif // GTCRN_STREAMING_H
