/**
 * GRU.h - Header file for GRU implementation
 *
 * Provides GRU (Gated Recurrent Unit) functionality for GTCRN network
 */

#ifndef GRU_H
#define GRU_H

#include <stddef.h>  // For NULL

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Data Structures
 * ============================================================================ */

/**
 * GRU weights structure
 * Contains all learnable parameters for a GRU layer
 */
typedef struct {
    int input_size;      // Input dimension
    int hidden_size;     // Hidden state dimension

    // Update gate weights
    float *W_z;          // (hidden_size, input_size)
    float *U_z;          // (hidden_size, hidden_size)
    float *b_z;          // (hidden_size,)

    // Reset gate weights
    float *W_r;          // (hidden_size, input_size)
    float *U_r;          // (hidden_size, hidden_size)
    float *b_r;          // (hidden_size,)

    // Candidate hidden state weights
    float *W_h;          // (hidden_size, input_size)
    float *U_h;          // (hidden_size, hidden_size)
    float *b_h;          // (hidden_size,)
} GRUWeights;

/* ============================================================================
 * Core GRU Functions
 * ============================================================================ */

/**
 * Single GRU cell forward pass
 *
 * Computes one timestep of GRU:
 *   z_t = σ(W_z * x_t + U_z * h_{t-1} + b_z)
 *   r_t = σ(W_r * x_t + U_r * h_{t-1} + b_r)
 *   h_tilde = tanh(W_h * x_t + U_h * (r_t ⊙ h_{t-1}) + b_h)
 *   h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h_tilde
 *
 * @param x         Input vector (input_size,)
 * @param h_prev    Previous hidden state (hidden_size,)
 * @param h_new     Output hidden state (hidden_size,)
 * @param weights   GRU weights structure
 * @param temp      Temporary buffer (4 * hidden_size,)
 */
void gru_cell_forward(
    const float *x,
    const float *h_prev,
    float *h_new,
    const GRUWeights *weights,
    float *temp
);

/**
 * GRU sequence forward pass (unidirectional)
 *
 * Processes a sequence through GRU layer
 *
 * @param input      Input sequence (seq_len, input_size)
 * @param output     Output sequence (seq_len, hidden_size)
 * @param h_init     Initial hidden state (hidden_size,), NULL for zeros
 * @param weights    GRU weights
 * @param seq_len    Sequence length
 * @param temp       Temporary buffer (4 * hidden_size,)
 */
void gru_forward(
    const float *input,
    float *output,
    const float *h_init,
    const GRUWeights *weights,
    int seq_len,
    float *temp
);

/**
 * Bidirectional GRU forward pass
 *
 * Processes sequence in both forward and backward directions
 * Output is concatenation of forward and backward hidden states
 *
 * @param input         Input sequence (seq_len, input_size)
 * @param output        Output sequence (seq_len, 2 * hidden_size)
 * @param h_init_fwd    Initial hidden state for forward pass
 * @param h_init_bwd    Initial hidden state for backward pass
 * @param weights_fwd   Forward GRU weights
 * @param weights_bwd   Backward GRU weights
 * @param seq_len       Sequence length
 * @param temp          Temporary buffer (4 * hidden_size,)
 */
void gru_bidirectional_forward(
    const float *input,
    float *output,
    const float *h_init_fwd,
    const float *h_init_bwd,
    const GRUWeights *weights_fwd,
    const GRUWeights *weights_bwd,
    int seq_len,
    float *temp
);

/* ============================================================================
 * Grouped GRU (GRNN) Functions
 * Used in GTCRN for efficiency
 * ============================================================================ */

/**
 * Grouped GRU forward pass
 *
 * Splits input into 2 groups and processes independently
 * This reduces parameters by ~50% compared to standard GRU
 *
 * Used in GTCRN's DPGRNN module:
 * - Intra-RNN: Bidirectional GRNN across frequency dimension
 * - Inter-RNN: Unidirectional GRNN across time dimension
 *
 * @param input         Input sequence (seq_len, input_size)
 * @param output        Output sequence (seq_len, hidden_size)
 * @param h_init        Initial hidden state (hidden_size,)
 * @param weights_g1    GRU weights for group 1
 * @param weights_g2    GRU weights for group 2
 * @param seq_len       Sequence length
 * @param bidirectional Whether to use bidirectional GRU
 * @param temp          Temporary buffer (4 * hidden_size,)
 */
void grnn_forward(
    const float *input,
    float *output,
    const float *h_init,
    const GRUWeights *weights_g1,
    const GRUWeights *weights_g2,
    int seq_len,
    int bidirectional,
    float *temp
);

/* ============================================================================
 * Weight Management Functions
 * ============================================================================ */

/**
 * Create and allocate GRU weights structure
 *
 * @param input_size    Input dimension
 * @param hidden_size   Hidden state dimension
 * @return Pointer to allocated GRUWeights structure
 */
GRUWeights* gru_weights_create(int input_size, int hidden_size);

/**
 * Free GRU weights structure
 *
 * @param weights Pointer to GRUWeights structure
 */
void gru_weights_free(GRUWeights *weights);

/**
 * Load GRU weights from binary file
 *
 * File format (all float32, row-major):
 *   W_z: (hidden_size, input_size)
 *   U_z: (hidden_size, hidden_size)
 *   b_z: (hidden_size,)
 *   W_r: (hidden_size, input_size)
 *   U_r: (hidden_size, hidden_size)
 *   b_r: (hidden_size,)
 *   W_h: (hidden_size, input_size)
 *   U_h: (hidden_size, hidden_size)
 *   b_h: (hidden_size,)
 *
 * @param weights   GRUWeights structure to load into
 * @param filename  Path to binary weight file
 * @return 0 on success, -1 on error
 */
int gru_weights_load(GRUWeights *weights, const char *filename);

/* ============================================================================
 * GTCRN-Specific Helper Functions
 * ============================================================================ */

/**
 * GTCRN Intra-RNN forward pass
 *
 * Bidirectional GRNN across frequency dimension
 * Input shape: (B*T, F, C) where F=97, C=16
 * Output shape: (B*T, F, C)
 *
 * @param input         Input (batch*time, freq, channels)
 * @param output        Output (batch*time, freq, channels)
 * @param weights_g1    Group 1 forward weights
 * @param weights_g2    Group 2 forward weights
 * @param weights_g1_bwd Group 1 backward weights
 * @param weights_g2_bwd Group 2 backward weights
 * @param batch_time    Batch size * time steps
 * @param freq_bins     Number of frequency bins (97 for 48kHz)
 * @param channels      Number of channels (16)
 * @param temp          Temporary buffer
 */
static inline void gtcrn_intra_rnn(
    const float *input,
    float *output,
    const GRUWeights *weights_g1,
    const GRUWeights *weights_g2,
    const GRUWeights *weights_g1_bwd,
    const GRUWeights *weights_g2_bwd,
    int batch_time,
    int freq_bins,
    int channels,
    float *temp
) {
    // Process each (batch, time) sample independently
    for (int bt = 0; bt < batch_time; bt++) {
        const float *input_bt = input + bt * freq_bins * channels;
        float *output_bt = output + bt * freq_bins * channels;

        // Bidirectional GRNN across frequency dimension
        grnn_forward(
            input_bt,
            output_bt,
            NULL,  // No initial hidden state
            weights_g1,
            weights_g2,
            freq_bins,  // Sequence length = frequency bins
            1,          // Bidirectional
            temp
        );
    }
}

/**
 * GTCRN Inter-RNN forward pass
 *
 * Unidirectional GRNN across time dimension
 * Input shape: (B*F, T, C) where F=97, T=time, C=16
 * Output shape: (B*F, T, C)
 *
 * @param input         Input (batch*freq, time, channels)
 * @param output        Output (batch*freq, time, channels)
 * @param h_init        Initial hidden state (batch*freq, channels)
 * @param weights_g1    Group 1 weights
 * @param weights_g2    Group 2 weights
 * @param batch_freq    Batch size * frequency bins
 * @param time_steps    Number of time steps
 * @param channels      Number of channels (16)
 * @param temp          Temporary buffer
 */
static inline void gtcrn_inter_rnn(
    const float *input,
    float *output,
    const float *h_init,
    const GRUWeights *weights_g1,
    const GRUWeights *weights_g2,
    int batch_freq,
    int time_steps,
    int channels,
    float *temp
) {
    // Process each (batch, freq) sample independently
    for (int bf = 0; bf < batch_freq; bf++) {
        const float *input_bf = input + bf * time_steps * channels;
        float *output_bf = output + bf * time_steps * channels;
        const float *h_init_bf = h_init ? h_init + bf * channels : NULL;

        // Unidirectional GRNN across time dimension
        grnn_forward(
            input_bf,
            output_bf,
            h_init_bf,
            weights_g1,
            weights_g2,
            time_steps,  // Sequence length = time steps
            0,           // Unidirectional (causal)
            temp
        );
    }
}

#ifdef __cplusplus
}
#endif

#endif /* GRU_H */
