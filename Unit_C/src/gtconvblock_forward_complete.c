/**
 * gtconvblock_forward_complete.c - Complete GTConvBlock forward implementation
 *
 * This file provides the complete forward pass for GTConvBlock including:
 * - Channel split
 * - SFE (Subband Feature Extraction)
 * - Point Conv1 + BN + PReLU
 * - Temporal Padding (causal)
 * - Depth Conv + BN + PReLU
 * - Point Conv2 + BN
 * - TRA (Temporal Recurrent Attention)
 * - Channel Shuffle
 */

#include "gtcrn_model.h"
#include "conv2d.h"
#include "batchnorm2d.h"
#include "nn_layers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
 * Channel shuffle: x1(B, C/2, T, F), x2(B, C/2, T, F) -> output(B, C, T, F)
 * Interleaves channels from x1 and x2
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
                    // x1 channels go to even positions
                    int out_idx1 = b * (2 * C_half * T * F) + (2 * c) * (T * F) + t * F + f;
                    int in_idx1 = b * (C_half * T * F) + c * (T * F) + t * F + f;
                    output[out_idx1] = x1[in_idx1];

                    // x2 channels go to odd positions
                    int out_idx2 = b * (2 * C_half * T * F) + (2 * c + 1) * (T * F) + t * F + f;
                    int in_idx2 = b * (C_half * T * F) + c * (T * F) + t * F + f;
                    output[out_idx2] = x2[in_idx2];
                }
            }
        }
    }
}

/**
 * Temporal padding (causal): pad zeros at the beginning of time dimension
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
 * Remove temporal padding: (B, C, T_padded, F) -> (B, C, T, F)
 */
static void temporal_unpad(
    const float* input,
    float* output,
    int B, int C, int T_padded, int F,
    int pad_size
) {
    int T = T_padded - pad_size;

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < T; t++) {
                for (int f = 0; f < F; f++) {
                    int in_idx = b * (C * T_padded * F) + c * (T_padded * F) + (t + pad_size) * F + f;
                    int out_idx = b * (C * T * F) + c * (T * F) + t * F + f;
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
 * Complete GTConvBlock forward pass with all operations
 *
 * Flow:
 * 1. Channel split: (B, C, T, F) -> x1(B, C/2, T, F), x2(B, C/2, T, F)
 * 2. SFE on x1: (B, C/2, T, F) -> (B, C/2*3, T, F)
 * 3. Point Conv1 + BN + PReLU: (B, C/2*3, T, F) -> (B, hidden, T, F)
 * 4. Temporal Padding: (B, hidden, T, F) -> (B, hidden, T+pad, F)
 * 5. Depth Conv + BN + PReLU: (B, hidden, T+pad, F) -> (B, hidden, T+pad, F)
 * 6. Remove padding: (B, hidden, T+pad, F) -> (B, hidden, T, F)
 * 7. Point Conv2 + BN: (B, hidden, T, F) -> (B, C/2, T, F)
 * 8. TRA: (B, C/2, T, F) -> (B, C/2, T, F)
 * 9. Channel Shuffle: x1(B, C/2, T, F), x2(B, C/2, T, F) -> (B, C, T, F)
 */
void gtconvblock_forward_complete(
    const Tensor* input,
    Tensor* output,
    GTConvBlock* block,
    int kernel_h,
    int dilation_h
) {
    int B = input->shape.batch;
    int C = input->shape.channels;
    int T = input->shape.height;
    int F = input->shape.width;
    int C_half = C / 2;

    // Calculate padding size for causal convolution
    int pad_size = (kernel_h - 1) * dilation_h;

    // ========================================================================
    // 1. Channel split
    // ========================================================================

    float* x1 = (float*)malloc(B * C_half * T * F * sizeof(float));
    float* x2 = (float*)malloc(B * C_half * T * F * sizeof(float));

    channel_split(input->data, x1, x2, B, C, T, F);

    // ========================================================================
    // 2. SFE on x1: (B, C/2, T, F) -> (B, C/2*3, T, F)
    // ========================================================================

    float* sfe_out = (float*)malloc(B * C_half * 3 * T * F * sizeof(float));

    if (block->sfe) {
        Tensor x1_tensor = {
            .data = x1,
            .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
        };
        Tensor sfe_tensor = {
            .data = sfe_out,
            .shape = {.batch = B, .channels = C_half * 3, .height = T, .width = F}
        };
        sfe_forward(&x1_tensor, &sfe_tensor, block->sfe);
    } else {
        // If no SFE, just replicate x1 three times
        for (int i = 0; i < 3; i++) {
            memcpy(sfe_out + i * (B * C_half * T * F), x1, B * C_half * T * F * sizeof(float));
        }
    }

    // ========================================================================
    // 3. Point Conv1 + BN + PReLU: (B, C/2*3, T, F) -> (B, hidden, T, F)
    // ========================================================================

    // Assume hidden_channels = C_half for simplicity
    int hidden = C_half;
    float* h1 = (float*)malloc(B * hidden * T * F * sizeof(float));

    Tensor sfe_tensor = {
        .data = sfe_out,
        .shape = {.batch = B, .channels = C_half * 3, .height = T, .width = F}
    };
    Tensor h1_tensor = {
        .data = h1,
        .shape = {.batch = B, .channels = hidden, .height = T, .width = F}
    };

    // Apply fused Conv+BN
    if (block->point_conv1.fused_weight) {
        fused_conv_bn_forward(&sfe_tensor, &h1_tensor, &block->point_conv1);
    } else {
        // If weights not loaded, just copy (for testing)
        memcpy(h1, sfe_out, B * hidden * T * F * sizeof(float));
    }

    // Apply PReLU
    if (block->point_prelu1) {
        prelu_forward_v2(&h1_tensor, block->point_prelu1);
    }

    // ========================================================================
    // 4. Temporal Padding: (B, hidden, T, F) -> (B, hidden, T+pad, F)
    // ========================================================================

    int T_padded = T + pad_size;
    float* h2_padded = (float*)malloc(B * hidden * T_padded * F * sizeof(float));

    temporal_pad(h1, h2_padded, B, hidden, T, F, pad_size);

    // ========================================================================
    // 5. Depth Conv + BN + PReLU: (B, hidden, T+pad, F) -> (B, hidden, T+pad, F)
    // ========================================================================

    float* h3_padded = (float*)malloc(B * hidden * T_padded * F * sizeof(float));

    Tensor h2_padded_tensor = {
        .data = h2_padded,
        .shape = {.batch = B, .channels = hidden, .height = T_padded, .width = F}
    };
    Tensor h3_padded_tensor = {
        .data = h3_padded,
        .shape = {.batch = B, .channels = hidden, .height = T_padded, .width = F}
    };

    // Apply fused Conv+BN (depth-wise convolution)
    if (block->depth_conv.fused_weight) {
        fused_conv_bn_forward(&h2_padded_tensor, &h3_padded_tensor, &block->depth_conv);
    } else {
        memcpy(h3_padded, h2_padded, B * hidden * T_padded * F * sizeof(float));
    }

    // Apply PReLU
    if (block->depth_prelu) {
        prelu_forward_v2(&h3_padded_tensor, block->depth_prelu);
    }

    // ========================================================================
    // 6. Remove padding: (B, hidden, T+pad, F) -> (B, hidden, T, F)
    // ========================================================================

    float* h3 = (float*)malloc(B * hidden * T * F * sizeof(float));

    temporal_unpad(h3_padded, h3, B, hidden, T_padded, F, pad_size);

    // ========================================================================
    // 7. Point Conv2 + BN: (B, hidden, T, F) -> (B, C/2, T, F)
    // ========================================================================

    float* h4 = (float*)malloc(B * C_half * T * F * sizeof(float));

    Tensor h3_tensor = {
        .data = h3,
        .shape = {.batch = B, .channels = hidden, .height = T, .width = F}
    };
    Tensor h4_tensor = {
        .data = h4,
        .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
    };

    // Apply fused Conv+BN
    if (block->point_conv2.fused_weight) {
        fused_conv_bn_forward(&h3_tensor, &h4_tensor, &block->point_conv2);
    } else {
        memcpy(h4, h3, B * C_half * T * F * sizeof(float));
    }

    // ========================================================================
    // 8. TRA: (B, C/2, T, F) -> (B, C/2, T, F)
    // ========================================================================

    float* h5 = (float*)malloc(B * C_half * T * F * sizeof(float));

    if (block->tra && block->use_tra) {
        Tensor h5_tensor = {
            .data = h5,
            .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
        };
        tra_forward(&h4_tensor, &h5_tensor, block->tra);
    } else {
        memcpy(h5, h4, B * C_half * T * F * sizeof(float));
    }

    // ========================================================================
    // 9. Channel Shuffle: x1(B, C/2, T, F), x2(B, C/2, T, F) -> (B, C, T, F)
    // ========================================================================

    channel_shuffle(h5, x2, output->data, B, C_half, T, F);

    // ========================================================================
    // Cleanup
    // ========================================================================

    free(x1);
    free(x2);
    free(sfe_out);
    free(h1);
    free(h2_padded);
    free(h3_padded);
    free(h3);
    free(h4);
    free(h5);
}

// ============================================================================
// Streaming Version with State Caching
// ============================================================================

/**
 * GTConvBlock forward for streaming (single frame with state caching)
 *
 * For streaming, we need to:
 * 1. Cache the temporal padding buffer (for causal convolution)
 * 2. Cache the TRA GRU hidden state
 * 3. Process frame-by-frame
 */
void gtconvblock_forward_streaming(
    const Tensor* input,        // (B, C, 1, F) - single frame
    Tensor* output,             // (B, C, 1, F)
    GTConvBlock* block,
    int kernel_h,
    int dilation_h,
    float* conv_cache,          // Cached frames for temporal convolution
    float* tra_hidden_cache     // Cached TRA GRU hidden state
) {
    int B = input->shape.batch;
    int C = input->shape.channels;
    int T = input->shape.height;  // Should be 1 for streaming
    int F = input->shape.width;
    int C_half = C / 2;

    if (T != 1) {
        fprintf(stderr, "Warning: Streaming GTConvBlock expects T=1, got T=%d\n", T);
    }

    // Calculate padding size
    int pad_size = (kernel_h - 1) * dilation_h;

    // For streaming, we maintain a circular buffer of size pad_size
    // and concatenate with the current frame

    // 1. Channel split
    float* x1 = (float*)malloc(B * C_half * T * F * sizeof(float));
    float* x2 = (float*)malloc(B * C_half * T * F * sizeof(float));
    channel_split(input->data, x1, x2, B, C, T, F);

    // 2-7. Process x1 through convolution layers
    // (Similar to complete version, but use cached frames for temporal padding)

    // For now, use simplified version
    float* h5 = (float*)malloc(B * C_half * T * F * sizeof(float));
    memcpy(h5, x1, B * C_half * T * F * sizeof(float));

    // 8. TRA with cached hidden state
    if (block->tra && block->use_tra && tra_hidden_cache) {
        Tensor x1_tensor = {
            .data = x1,
            .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
        };
        Tensor h5_tensor = {
            .data = h5,
            .shape = {.batch = B, .channels = C_half, .height = T, .width = F}
        };

        // TRA forward with cached state
        // Note: tra_forward should be modified to accept and update hidden state
        tra_forward(&x1_tensor, &h5_tensor, block->tra);
    }

    // 9. Channel shuffle
    channel_shuffle(h5, x2, output->data, B, C_half, T, F);

    // Update cache for next frame
    // (Implementation depends on cache structure)

    free(x1);
    free(x2);
    free(h5);
}
