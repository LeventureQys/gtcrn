/**
 * gtconvblock_forward.c - Complete GTConvBlock forward implementation
 *
 * Implements the full forward pass for GTConvBlock including:
 * - Channel split
 * - SFE (Subband Feature Extraction)
 * - Point Conv1 + BN + PReLU
 * - Temporal Padding
 * - Depth Conv + BN + PReLU
 * - Point Conv2 + BN
 * - TRA (Temporal Recurrent Attention)
 * - Channel Shuffle
 */

#include "gtcrn_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Channel split: (B, C, T, F) -> x1(B, C/2, T, F), x2(B, C/2, T, F)
 */
static void channel_split(
    const float* input,
    float* x1,
    float* x2,
    int B, int C, int T, int F
) {
    int C_half = C / 2;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                for (int c = 0; c < C_half; c++) {
                    int in_idx1 = b * (C * T * F) + c * (T * F) + t * F + f;
                    int in_idx2 = b * (C * T * F) + (c + C_half) * (T * F) + t * F + f;
                    int out_idx = b * (C_half * T * F) + c * (T * F) + t * F + f;

                    x1[out_idx] = input[in_idx1];
                    x2[out_idx] = input[in_idx2];
                }
            }
        }
    }
}

/**
 * Channel shuffle: x1(B, C/2, T, F), x2(B, C/2, T, F) -> (B, C, T, F)
 * Interleaves two channel groups
 */
static void channel_shuffle(
    const float* x1,
    const float* x2,
    float* output,
    int B, int C_half, int T, int F
) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C_half; c++) {
            for (int t = 0; t < T; t++) {
                for (int f = 0; f < F; f++) {
                    // x1 goes to even channels
                    int out_idx1 = b * (2 * C_half * T * F) + (2 * c) * (T * F) + t * F + f;
                    int in_idx1 = b * (C_half * T * F) + c * (T * F) + t * F + f;
                    output[out_idx1] = x1[in_idx1];

                    // x2 goes to odd channels
                    int out_idx2 = b * (2 * C_half * T * F) + (2 * c + 1) * (T * F) + t * F + f;
                    int in_idx2 = b * (C_half * T * F) + c * (T * F) + t * F + f;
                    output[out_idx2] = x2[in_idx2];
                }
            }
        }
    }
}

/**
 * Temporal padding: pad zeros at the beginning of time dimension
 * Used for causal convolution
 */
static void temporal_pad(
    const float* input,
    float* output,
    int B, int C, int T, int F,
    int pad_size
) {
    int T_padded = T + pad_size;

    // Initialize output to zero
    memset(output, 0, B * C * T_padded * F * sizeof(float));

    // Copy input data to padded position
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < T; t++) {
                for (int f = 0; f < F; f++) {
                    int in_idx = b * (C * T * F) + c * (T * F) + t * F + f;
                    int out_idx = b * (C * T_padded * F) + c * (T_padded * F) + (t + pad_size) * F + f;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

/**
 * Remove temporal padding: extract valid time frames
 */
static void temporal_unpad(
    const float* input,
    float* output,
    int B, int C, int T_padded, int F,
    int pad_size,
    int T_out
) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < T_out; t++) {
                for (int f = 0; f < F; f++) {
                    int in_idx = b * (C * T_padded * F) + c * (T_padded * F) + (t + pad_size) * F + f;
                    int out_idx = b * (C * T_out * F) + c * (T_out * F) + t * F + f;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// ============================================================================
// Complete GTConvBlock Forward Pass
// ============================================================================

/**
 * Complete GTConvBlock forward pass
 *
 * Input: (B, C, T, F)
 * Output: (B, C, T, F)
 *
 * Steps:
 * 1. Channel split: (B, C, T, F) -> x1(B, C/2, T, F), x2(B, C/2, T, F)
 * 2. SFE on x1: (B, C/2, T, F) -> (B, C/2*3, T, F)
 * 3. Point Conv1 + BN + PReLU: (B, C/2*3, T, F) -> (B, hidden, T, F)
 * 4. Temporal Padding: (B, hidden, T, F) -> (B, hidden, T+pad, F)
 * 5. Depth Conv + BN + PReLU: (B, hidden, T+pad, F) -> (B, hidden, T', F)
 * 6. Temporal Unpad: (B, hidden, T', F) -> (B, hidden, T, F)
 * 7. Point Conv2 + BN: (B, hidden, T, F) -> (B, C/2, T, F)
 * 8. TRA: (B, C/2, T, F) -> (B, C/2, T, F)
 * 9. Channel Shuffle: x1(B, C/2, T, F), x2(B, C/2, T, F) -> (B, C, T, F)
 */
void gtconvblock_forward_complete(
    const Tensor* input,
    Tensor* output,
    GTConvBlock* block
) {
    int B = input->shape.batch;
    int C = input->shape.channels;
    int T = input->shape.height;
    int F = input->shape.width;
    int C_half = C / 2;

    // Allocate working buffers
    float* x1 = (float*)malloc(B * C_half * T * F * sizeof(float));
    float* x2 = (float*)malloc(B * C_half * T * F * sizeof(float));
    float* sfe_out = (float*)malloc(B * C_half * 3 * T * F * sizeof(float));

    // 1. Channel split
    channel_split(input->data, x1, x2, B, C, T, F);

    // 2. SFE on x1
    Tensor x1_tensor = {
        .data = x1,
        .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
    };
    Tensor sfe_tensor = {
        .data = sfe_out,
        .shape = {.batch = B, .channels = C_half * 3, .height = T, .width = F}
    };
    sfe_forward(&x1_tensor, &sfe_tensor, block->sfe);

    // Get hidden_channels from fused_conv_bn
    int hidden_channels = block->point_conv1.out_channels;

    // 3. Point Conv1 + BN + PReLU
    float* h1 = (float*)malloc(B * hidden_channels * T * F * sizeof(float));
    Tensor h1_tensor = {
        .data = h1,
        .shape = {.batch = B, .channels = hidden_channels, .height = T, .width = F}
    };

    fused_conv_bn_forward(&sfe_tensor, &h1_tensor, &block->point_conv1);

    if (block->point_prelu1) {
        prelu_forward_v2(&h1_tensor, block->point_prelu1);
    }

    // 4. Temporal Padding (for causal convolution)
    // Calculate padding size based on dilation and kernel size
    int kernel_t = block->depth_conv.kernel_h;  // Assuming kernel_h is time dimension
    int dilation_t = block->depth_conv.dilation_h;
    int pad_size = (kernel_t - 1) * dilation_t;

    float* h1_padded = (float*)malloc(B * hidden_channels * (T + pad_size) * F * sizeof(float));
    temporal_pad(h1, h1_padded, B, hidden_channels, T, F, pad_size);

    // 5. Depth Conv + BN + PReLU
    Tensor h1_padded_tensor = {
        .data = h1_padded,
        .shape = {.batch = B, .channels = hidden_channels, .height = T + pad_size, .width = F}
    };

    // Calculate output size after depth conv
    int T_after_conv = calculate_output_size(
        T + pad_size,
        kernel_t,
        block->depth_conv.stride_h,
        0,  // No additional padding
        dilation_t
    );

    float* h2 = (float*)malloc(B * hidden_channels * T_after_conv * F * sizeof(float));
    Tensor h2_tensor = {
        .data = h2,
        .shape = {.batch = B, .channels = hidden_channels, .height = T_after_conv, .width = F}
    };

    fused_conv_bn_forward(&h1_padded_tensor, &h2_tensor, &block->depth_conv);

    if (block->depth_prelu) {
        prelu_forward_v2(&h2_tensor, block->depth_prelu);
    }

    // 6. Temporal Unpad (if needed)
    float* h2_unpadded = h2;  // If T_after_conv == T, no unpadding needed
    if (T_after_conv != T) {
        h2_unpadded = (float*)malloc(B * hidden_channels * T * F * sizeof(float));
        temporal_unpad(h2, h2_unpadded, B, hidden_channels, T_after_conv, F, 0, T);
    }

    // 7. Point Conv2 + BN
    float* h3 = (float*)malloc(B * C_half * T * F * sizeof(float));
    Tensor h2_unpadded_tensor = {
        .data = h2_unpadded,
        .shape = {.batch = B, .channels = hidden_channels, .height = T, .width = F}
    };
    Tensor h3_tensor = {
        .data = h3,
        .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
    };

    fused_conv_bn_forward(&h2_unpadded_tensor, &h3_tensor, &block->point_conv2);

    // 8. TRA (if enabled)
    if (block->tra && block->use_tra) {
        Tensor tra_out_tensor = {
            .data = h3,  // In-place operation
            .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
        };
        tra_forward(&h3_tensor, &tra_out_tensor, block->tra);
    }

    // 9. Channel Shuffle
    channel_shuffle(h3, x2, output->data, B, C_half, T, F);

    // Free working buffers
    free(x1);
    free(x2);
    free(sfe_out);
    free(h1);
    free(h1_padded);
    free(h2);
    if (h2_unpadded != h2) free(h2_unpadded);
    free(h3);
}

// ============================================================================
// Streaming GTConvBlock Forward (with state caching)
// ============================================================================

/**
 * Streaming GTConvBlock forward pass with state caching
 * For real-time processing
 */
typedef struct {
    float* conv_cache;      // Cache for temporal convolution
    float* tra_cache;       // Cache for TRA hidden state
    int cache_size;
} GTConvBlockCache;

/**
 * Create cache for streaming GTConvBlock
 */
GTConvBlockCache* gtconvblock_cache_create(int hidden_channels, int cache_frames, int freq_bins) {
    GTConvBlockCache* cache = (GTConvBlockCache*)malloc(sizeof(GTConvBlockCache));
    if (!cache) return NULL;

    cache->cache_size = cache_frames;
    cache->conv_cache = (float*)calloc(hidden_channels * cache_frames * freq_bins, sizeof(float));
    cache->tra_cache = (float*)calloc(hidden_channels, sizeof(float));

    return cache;
}

/**
 * Free GTConvBlock cache
 */
void gtconvblock_cache_free(GTConvBlockCache* cache) {
    if (cache) {
        free(cache->conv_cache);
        free(cache->tra_cache);
        free(cache);
    }
}

/**
 * Streaming GTConvBlock forward with caching
 * Processes one frame at a time for real-time inference
 */
void gtconvblock_forward_streaming(
    const Tensor* input,
    Tensor* output,
    GTConvBlock* block,
    GTConvBlockCache* cache
) {
    // Similar to gtconvblock_forward_complete, but:
    // 1. Use cache->conv_cache for temporal padding
    // 2. Use cache->tra_cache for TRA hidden state
    // 3. Update cache after processing

    // Implementation details depend on specific streaming requirements
    // This is a placeholder for the streaming version
    gtconvblock_forward_complete(input, output, block);
}
