/**
 * GRU_bidirectional_complete.c - Complete Bidirectional Grouped GRU Implementation
 *
 * This file provides a complete implementation of bidirectional grouped GRU (GRNN)
 * as used in GTCRN's DPGRNN module.
 *
 * Key features:
 * - Proper bidirectional processing with separate forward/backward weights
 * - Grouped GRU for efficiency (splits channels into groups)
 * - State caching for streaming inference
 */

#include "GRU.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Complete Bidirectional GRNN Implementation
// ============================================================================

/**
 * Bidirectional Grouped GRU forward pass (complete version)
 *
 * This function implements the Intra-RNN in GTCRN's DPGRNN:
 * - Input: (B*T, F, C) where F=97 (frequency bins), C=16 (channels)
 * - Output: (B*T, F, C) (same size due to bidirectional)
 * - Processes frequency dimension with bidirectional GRU
 * - Uses 2 groups for efficiency
 *
 * @param input             Input sequence (seq_len, input_size)
 * @param output            Output sequence (seq_len, input_size)
 * @param h_init_fwd_g1     Initial hidden state for forward group 1
 * @param h_init_fwd_g2     Initial hidden state for forward group 2
 * @param h_init_bwd_g1     Initial hidden state for backward group 1
 * @param h_init_bwd_g2     Initial hidden state for backward group 2
 * @param weights_fwd_g1    Forward weights for group 1
 * @param weights_fwd_g2    Forward weights for group 2
 * @param weights_bwd_g1    Backward weights for group 1
 * @param weights_bwd_g2    Backward weights for group 2
 * @param seq_len           Sequence length (97 for GTCRN)
 * @param temp              Temporary buffer (4 * max_hidden_size)
 */
void grnn_bidirectional_forward_complete(
    const float* input,
    float* output,
    const float* h_init_fwd_g1,
    const float* h_init_fwd_g2,
    const float* h_init_bwd_g1,
    const float* h_init_bwd_g2,
    const GRUWeights* weights_fwd_g1,
    const GRUWeights* weights_fwd_g2,
    const GRUWeights* weights_bwd_g1,
    const GRUWeights* weights_bwd_g2,
    int seq_len,
    float* temp
) {
    // Get dimensions
    int input_size_g1 = weights_fwd_g1->input_size;
    int input_size_g2 = weights_fwd_g2->input_size;
    int hidden_size_g1 = weights_fwd_g1->hidden_size;
    int hidden_size_g2 = weights_fwd_g2->hidden_size;
    int input_size = input_size_g1 + input_size_g2;

    // ========================================================================
    // Step 1: Split input into two groups
    // ========================================================================

    float* input_g1 = (float*)malloc(seq_len * input_size_g1 * sizeof(float));
    float* input_g2 = (float*)malloc(seq_len * input_size_g2 * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        // Group 1: first half of channels
        memcpy(input_g1 + t * input_size_g1,
               input + t * input_size,
               input_size_g1 * sizeof(float));

        // Group 2: second half of channels
        memcpy(input_g2 + t * input_size_g2,
               input + t * input_size + input_size_g1,
               input_size_g2 * sizeof(float));
    }

    // ========================================================================
    // Step 2: Forward pass for both groups
    // ========================================================================

    float* output_fwd_g1 = (float*)malloc(seq_len * hidden_size_g1 * sizeof(float));
    float* output_fwd_g2 = (float*)malloc(seq_len * hidden_size_g2 * sizeof(float));

    gru_forward(input_g1, output_fwd_g1, h_init_fwd_g1, weights_fwd_g1, seq_len, temp);
    gru_forward(input_g2, output_fwd_g2, h_init_fwd_g2, weights_fwd_g2, seq_len, temp);

    // ========================================================================
    // Step 3: Backward pass for both groups
    // ========================================================================

    // Reverse input for backward pass
    float* input_g1_rev = (float*)malloc(seq_len * input_size_g1 * sizeof(float));
    float* input_g2_rev = (float*)malloc(seq_len * input_size_g2 * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        memcpy(input_g1_rev + t * input_size_g1,
               input_g1 + (seq_len - 1 - t) * input_size_g1,
               input_size_g1 * sizeof(float));
        memcpy(input_g2_rev + t * input_size_g2,
               input_g2 + (seq_len - 1 - t) * input_size_g2,
               input_size_g2 * sizeof(float));
    }

    // Run backward GRU
    float* output_bwd_g1 = (float*)malloc(seq_len * hidden_size_g1 * sizeof(float));
    float* output_bwd_g2 = (float*)malloc(seq_len * hidden_size_g2 * sizeof(float));

    gru_forward(input_g1_rev, output_bwd_g1, h_init_bwd_g1, weights_bwd_g1, seq_len, temp);
    gru_forward(input_g2_rev, output_bwd_g2, h_init_bwd_g2, weights_bwd_g2, seq_len, temp);

    // Reverse backward outputs
    float* output_bwd_g1_rev = (float*)malloc(seq_len * hidden_size_g1 * sizeof(float));
    float* output_bwd_g2_rev = (float*)malloc(seq_len * hidden_size_g2 * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        memcpy(output_bwd_g1_rev + t * hidden_size_g1,
               output_bwd_g1 + (seq_len - 1 - t) * hidden_size_g1,
               hidden_size_g1 * sizeof(float));
        memcpy(output_bwd_g2_rev + t * hidden_size_g2,
               output_bwd_g2 + (seq_len - 1 - t) * hidden_size_g2,
               hidden_size_g2 * sizeof(float));
    }

    // ========================================================================
    // Step 4: Concatenate forward and backward outputs
    // ========================================================================

    // Output format: [fwd_g1, bwd_g1, fwd_g2, bwd_g2] for each timestep
    // This gives us the same total size as input (bidirectional doubles, but we split)

    for (int t = 0; t < seq_len; t++) {
        int out_idx = t * input_size;

        // Group 1: forward + backward
        memcpy(output + out_idx,
               output_fwd_g1 + t * hidden_size_g1,
               hidden_size_g1 * sizeof(float));
        memcpy(output + out_idx + hidden_size_g1,
               output_bwd_g1_rev + t * hidden_size_g1,
               hidden_size_g1 * sizeof(float));

        // Group 2: forward + backward
        memcpy(output + out_idx + 2 * hidden_size_g1,
               output_fwd_g2 + t * hidden_size_g2,
               hidden_size_g2 * sizeof(float));
        memcpy(output + out_idx + 2 * hidden_size_g1 + hidden_size_g2,
               output_bwd_g2_rev + t * hidden_size_g2,
               hidden_size_g2 * sizeof(float));
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    free(input_g1);
    free(input_g2);
    free(input_g1_rev);
    free(input_g2_rev);
    free(output_fwd_g1);
    free(output_fwd_g2);
    free(output_bwd_g1);
    free(output_bwd_g2);
    free(output_bwd_g1_rev);
    free(output_bwd_g2_rev);
}

// ============================================================================
// Unidirectional Grouped GRU (for Inter-RNN)
// ============================================================================

/**
 * Unidirectional Grouped GRU forward pass with state caching
 *
 * This function implements the Inter-RNN in GTCRN's DPGRNN:
 * - Input: (B*F, T, C) where F=97, T=time, C=16
 * - Output: (B*F, T, C)
 * - Processes time dimension with unidirectional (causal) GRU
 * - Uses 2 groups for efficiency
 * - Supports state caching for streaming
 *
 * @param input             Input sequence (seq_len, input_size)
 * @param output            Output sequence (seq_len, input_size)
 * @param h_prev_g1         Previous hidden state for group 1 (for streaming)
 * @param h_prev_g2         Previous hidden state for group 2 (for streaming)
 * @param h_next_g1         Next hidden state for group 1 (output for streaming)
 * @param h_next_g2         Next hidden state for group 2 (output for streaming)
 * @param weights_g1        Weights for group 1
 * @param weights_g2        Weights for group 2
 * @param seq_len           Sequence length
 * @param temp              Temporary buffer
 */
void grnn_unidirectional_forward_with_state(
    const float* input,
    float* output,
    const float* h_prev_g1,
    const float* h_prev_g2,
    float* h_next_g1,
    float* h_next_g2,
    const GRUWeights* weights_g1,
    const GRUWeights* weights_g2,
    int seq_len,
    float* temp
) {
    // Get dimensions
    int input_size_g1 = weights_g1->input_size;
    int input_size_g2 = weights_g2->input_size;
    int hidden_size_g1 = weights_g1->hidden_size;
    int hidden_size_g2 = weights_g2->hidden_size;
    int input_size = input_size_g1 + input_size_g2;

    // ========================================================================
    // Step 1: Split input into two groups
    // ========================================================================

    float* input_g1 = (float*)malloc(seq_len * input_size_g1 * sizeof(float));
    float* input_g2 = (float*)malloc(seq_len * input_size_g2 * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        memcpy(input_g1 + t * input_size_g1,
               input + t * input_size,
               input_size_g1 * sizeof(float));
        memcpy(input_g2 + t * input_size_g2,
               input + t * input_size + input_size_g1,
               input_size_g2 * sizeof(float));
    }

    // ========================================================================
    // Step 2: Process each group with GRU (using cached states)
    // ========================================================================

    float* output_g1 = (float*)malloc(seq_len * hidden_size_g1 * sizeof(float));
    float* output_g2 = (float*)malloc(seq_len * hidden_size_g2 * sizeof(float));

    // Process group 1
    gru_forward(input_g1, output_g1, h_prev_g1, weights_g1, seq_len, temp);

    // Process group 2
    gru_forward(input_g2, output_g2, h_prev_g2, weights_g2, seq_len, temp);

    // ========================================================================
    // Step 3: Save final hidden states for next iteration (streaming)
    // ========================================================================

    if (h_next_g1) {
        memcpy(h_next_g1,
               output_g1 + (seq_len - 1) * hidden_size_g1,
               hidden_size_g1 * sizeof(float));
    }

    if (h_next_g2) {
        memcpy(h_next_g2,
               output_g2 + (seq_len - 1) * hidden_size_g2,
               hidden_size_g2 * sizeof(float));
    }

    // ========================================================================
    // Step 4: Concatenate outputs
    // ========================================================================

    for (int t = 0; t < seq_len; t++) {
        memcpy(output + t * input_size,
               output_g1 + t * hidden_size_g1,
               hidden_size_g1 * sizeof(float));
        memcpy(output + t * input_size + hidden_size_g1,
               output_g2 + t * hidden_size_g2,
               hidden_size_g2 * sizeof(float));
    }

    // ========================================================================
    // Cleanup
    // ========================================================================

    free(input_g1);
    free(input_g2);
    free(output_g1);
    free(output_g2);
}

// ============================================================================
// Helper Functions for DPGRNN
// ============================================================================

/**
 * Create bidirectional GRNN weights structure
 */
typedef struct {
    GRUWeights* fwd_g1;
    GRUWeights* fwd_g2;
    GRUWeights* bwd_g1;
    GRUWeights* bwd_g2;
} BiGRNNWeights;

BiGRNNWeights* bigrnn_weights_create(int input_size, int hidden_size) {
    BiGRNNWeights* weights = (BiGRNNWeights*)malloc(sizeof(BiGRNNWeights));

    int input_size_per_group = input_size / 2;
    int hidden_size_per_group = hidden_size / 4;  // Bidirectional doubles output

    weights->fwd_g1 = gru_weights_create(input_size_per_group, hidden_size_per_group);
    weights->fwd_g2 = gru_weights_create(input_size_per_group, hidden_size_per_group);
    weights->bwd_g1 = gru_weights_create(input_size_per_group, hidden_size_per_group);
    weights->bwd_g2 = gru_weights_create(input_size_per_group, hidden_size_per_group);

    return weights;
}

void bigrnn_weights_free(BiGRNNWeights* weights) {
    if (weights) {
        gru_weights_free(weights->fwd_g1);
        gru_weights_free(weights->fwd_g2);
        gru_weights_free(weights->bwd_g1);
        gru_weights_free(weights->bwd_g2);
        free(weights);
    }
}

/**
 * Create unidirectional GRNN weights structure
 */
typedef struct {
    GRUWeights* g1;
    GRUWeights* g2;
} UniGRNNWeights;

UniGRNNWeights* unigrnn_weights_create(int input_size, int hidden_size) {
    UniGRNNWeights* weights = (UniGRNNWeights*)malloc(sizeof(UniGRNNWeights));

    int input_size_per_group = input_size / 2;
    int hidden_size_per_group = hidden_size / 2;

    weights->g1 = gru_weights_create(input_size_per_group, hidden_size_per_group);
    weights->g2 = gru_weights_create(input_size_per_group, hidden_size_per_group);

    return weights;
}

void unigrnn_weights_free(UniGRNNWeights* weights) {
    if (weights) {
        gru_weights_free(weights->g1);
        gru_weights_free(weights->g2);
        free(weights);
    }
}

// ============================================================================
// Test/Example Usage
// ============================================================================

#ifdef TEST_BIGRNN
int main() {
    printf("Testing Bidirectional Grouped GRU\n");
    printf("==================================\n\n");

    // GTCRN Intra-RNN parameters
    int seq_len = 97;      // Frequency bins
    int input_size = 16;   // Channels
    int hidden_size = 16;  // Output size (same as input for bidirectional grouped)

    // Create weights
    BiGRNNWeights* weights = bigrnn_weights_create(input_size, hidden_size);

    // Allocate buffers
    float* input = (float*)malloc(seq_len * input_size * sizeof(float));
    float* output = (float*)malloc(seq_len * input_size * sizeof(float));
    float* temp = (float*)malloc(4 * hidden_size * sizeof(float));

    // Initialize input with random data
    for (int i = 0; i < seq_len * input_size; i++) {
        input[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    // Run bidirectional GRNN
    grnn_bidirectional_forward_complete(
        input, output,
        NULL, NULL, NULL, NULL,  // No initial hidden states
        weights->fwd_g1, weights->fwd_g2,
        weights->bwd_g1, weights->bwd_g2,
        seq_len, temp
    );

    // Print results
    printf("Input shape: (%d, %d)\n", seq_len, input_size);
    printf("Output shape: (%d, %d)\n", seq_len, input_size);
    printf("\nFirst timestep output:\n");
    for (int i = 0; i < input_size; i++) {
        printf("  %.4f", output[i]);
    }
    printf("\n\n");

    // Cleanup
    free(input);
    free(output);
    free(temp);
    bigrnn_weights_free(weights);

    printf("✓ Test completed successfully\n");

    return 0;
}
#endif
