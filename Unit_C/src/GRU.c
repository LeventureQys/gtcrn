/**
 * GRU.c - GRU (Gated Recurrent Unit) Implementation for GTCRN
 *
 * This file implements:
 * 1. Standard GRU cell
 * 2. Grouped GRU (GRNN) - splits input into groups for efficiency
 * 3. Bidirectional GRU
 *
 * GRU Equations:
 *   z_t = sigmoid(W_z * x_t + U_z * h_{t-1} + b_z)  // Update gate
 *   r_t = sigmoid(W_r * x_t + U_r * h_{t-1} + b_r)  // Reset gate
 *   h_tilde = tanh(W_h * x_t + U_h * (r_t  h_{t-1}) + b_h)  // Candidate
 *   h_t = (1 - z_t)  h_{t-1} + z_t  h_tilde  // New hidden state
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "GRU.h"

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

/**
 * Sigmoid activation: Ã(x) = 1 / (1 + e^(-x))
 * Fast approximation using tanh
 */
static inline float sigmoid_approx(float x) {
    // Ã(x) = 0.5 + 0.5 * tanh(0.5 * x)
    if (x >= 8.0f) return 1.0f;
    if (x <= -8.0f) return 0.0f;

    // Use tanh approximation
    float x_half = 0.5f * x;
    float tanh_val = tanhf(x_half);
    return 0.5f + 0.5f * tanh_val;
}

/**
 * Tanh activation: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 */
static inline float tanh_approx(float x) {
    if (x >= 8.0f) return 1.0f;
    if (x <= -8.0f) return -1.0f;
    return tanhf(x);
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

/**
 * Matrix-vector multiplication: y = A * x + b
 * A: (out_size, in_size) row-major
 * x: (in_size,)
 * b: (out_size,)
 * y: (out_size,)
 */
static void matvec_add_bias(
    const float *A,
    const float *x,
    const float *b,
    float *y,
    int out_size,
    int in_size
) {
    for (int i = 0; i < out_size; i++) {
        float sum = b ? b[i] : 0.0f;
        for (int j = 0; j < in_size; j++) {
            sum += A[i * in_size + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * Element-wise multiplication: z = x  y
 */
static void elementwise_mul(
    const float *x,
    const float *y,
    float *z,
    int size
) {
    for (int i = 0; i < size; i++) {
        z[i] = x[i] * y[i];
    }
}

/**
 * Element-wise operation: z = (1 - x)  y + x  w
 * Used in GRU: h_t = (1 - z_t)  h_{t-1} + z_t  h_tilde
 */
static void gru_combine(
    const float *z,      // Update gate
    const float *h_prev, // Previous hidden state
    const float *h_tilde,// Candidate hidden state
    float *h_new,        // New hidden state
    int size
) {
    for (int i = 0; i < size; i++) {
        h_new[i] = (1.0f - z[i]) * h_prev[i] + z[i] * h_tilde[i];
    }
}

/* ============================================================================
 * GRU Cell Implementation
 * ============================================================================ */

/**
 * Single GRU cell forward pass
 *
 * @param x         Input vector (input_size,)
 * @param h_prev    Previous hidden state (hidden_size,)
 * @param h_new     Output hidden state (hidden_size,)
 * @param weights   GRU weights structure
 * @param temp      Temporary buffer (3 * hidden_size,)
 */
void gru_cell_forward(
    const float *x,
    const float *h_prev,
    float *h_new,
    const GRUWeights *weights,
    float *temp
) {
    int input_size = weights->input_size;
    int hidden_size = weights->hidden_size;

    // Allocate temporary buffers
    float *z = temp;                          // Update gate
    float *r = temp + hidden_size;            // Reset gate
    float *h_tilde = temp + 2 * hidden_size;  // Candidate hidden state
    float *temp_buf = temp + 3 * hidden_size; // Extra temp space

    // 1. Compute update gate: z = Ã(W_z * x + U_z * h_prev + b_z)
    matvec_add_bias(weights->W_z, x, weights->b_z, z, hidden_size, input_size);
    matvec_add_bias(weights->U_z, h_prev, NULL, temp_buf, hidden_size, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        z[i] = sigmoid_approx(z[i] + temp_buf[i]);
    }

    // 2. Compute reset gate: r = Ã(W_r * x + U_r * h_prev + b_r)
    matvec_add_bias(weights->W_r, x, weights->b_r, r, hidden_size, input_size);
    matvec_add_bias(weights->U_r, h_prev, NULL, temp_buf, hidden_size, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        r[i] = sigmoid_approx(r[i] + temp_buf[i]);
    }

    // 3. Compute candidate: h_tilde = tanh(W_h * x + U_h * (r  h_prev) + b_h)
    // First compute r  h_prev
    elementwise_mul(r, h_prev, temp_buf, hidden_size);

    // Then compute h_tilde
    matvec_add_bias(weights->W_h, x, weights->b_h, h_tilde, hidden_size, input_size);
    float *temp_buf2 = temp_buf + hidden_size;
    matvec_add_bias(weights->U_h, temp_buf, NULL, temp_buf2, hidden_size, hidden_size);
    for (int i = 0; i < hidden_size; i++) {
        h_tilde[i] = tanh_approx(h_tilde[i] + temp_buf2[i]);
    }

    // 4. Compute new hidden state: h_new = (1 - z)  h_prev + z  h_tilde
    gru_combine(z, h_prev, h_tilde, h_new, hidden_size);
}

/**
 * GRU sequence forward pass (unidirectional)
 *
 * @param input      Input sequence (seq_len, input_size)
 * @param output     Output sequence (seq_len, hidden_size)
 * @param h_init     Initial hidden state (hidden_size,), can be NULL
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
) {
    // Validate parameters
    if (!weights) {
        fprintf(stderr, "Error: gru_forward - weights is NULL\n");
        return;
    }
    if (!input) {
        fprintf(stderr, "Error: gru_forward - input is NULL\n");
        return;
    }
    if (!output) {
        fprintf(stderr, "Error: gru_forward - output is NULL\n");
        return;
    }
    if (!temp) {
        fprintf(stderr, "Error: gru_forward - temp is NULL\n");
        return;
    }
    if (seq_len <= 0) {
        fprintf(stderr, "Error: gru_forward - invalid seq_len: %d\n", seq_len);
        return;
    }

    int input_size = weights->input_size;
    int hidden_size = weights->hidden_size;

    // Validate weight dimensions
    if (input_size <= 0 || hidden_size <= 0) {
        fprintf(stderr, "Error: gru_forward - invalid weight dimensions: input_size=%d, hidden_size=%d\n",
                input_size, hidden_size);
        return;
    }

    // Allocate hidden state buffer
    float *h_prev = (float *)malloc(hidden_size * sizeof(float));
    float *h_new = (float *)malloc(hidden_size * sizeof(float));

    if (!h_prev || !h_new) {
        fprintf(stderr, "Error: gru_forward - failed to allocate hidden state buffers\n");
        free(h_prev);
        free(h_new);
        return;
    }

    // Initialize hidden state
    if (h_init) {
        memcpy(h_prev, h_init, hidden_size * sizeof(float));
    } else {
        memset(h_prev, 0, hidden_size * sizeof(float));
    }

    // Process sequence
    for (int t = 0; t < seq_len; t++) {
        const float *x_t = input + t * input_size;
        float *y_t = output + t * hidden_size;

        // Forward through GRU cell
        gru_cell_forward(x_t, h_prev, h_new, weights, temp);

        // Copy output
        memcpy(y_t, h_new, hidden_size * sizeof(float));

        // Update hidden state
        memcpy(h_prev, h_new, hidden_size * sizeof(float));
    }

    free(h_prev);
    free(h_new);
}

/**
 * Bidirectional GRU forward pass
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
) {
    int input_size = weights_fwd->input_size;
    int hidden_size = weights_fwd->hidden_size;

    // Allocate buffers for forward and backward outputs
    float *output_fwd = (float *)malloc(seq_len * hidden_size * sizeof(float));
    float *output_bwd = (float *)malloc(seq_len * hidden_size * sizeof(float));

    // Forward pass
    gru_forward(input, output_fwd, h_init_fwd, weights_fwd, seq_len, temp);

    // Backward pass - reverse input
    float *input_rev = (float *)malloc(seq_len * input_size * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        memcpy(input_rev + t * input_size,
               input + (seq_len - 1 - t) * input_size,
               input_size * sizeof(float));
    }

    gru_forward(input_rev, output_bwd, h_init_bwd, weights_bwd, seq_len, temp);

    // Reverse backward output
    float *output_bwd_rev = (float *)malloc(seq_len * hidden_size * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        memcpy(output_bwd_rev + t * hidden_size,
               output_bwd + (seq_len - 1 - t) * hidden_size,
               hidden_size * sizeof(float));
    }

    // Concatenate forward and backward outputs
    for (int t = 0; t < seq_len; t++) {
        memcpy(output + t * 2 * hidden_size,
               output_fwd + t * hidden_size,
               hidden_size * sizeof(float));
        memcpy(output + t * 2 * hidden_size + hidden_size,
               output_bwd_rev + t * hidden_size,
               hidden_size * sizeof(float));
    }

    free(output_fwd);
    free(output_bwd);
    free(input_rev);
    free(output_bwd_rev);
}

/* ============================================================================
 * Grouped GRU (GRNN) Implementation
 * Used in GTCRN for efficiency
 * ============================================================================ */

/**
 * Grouped GRU forward pass
 * Splits input into 2 groups and processes independently
 *
 * @param input         Input sequence (seq_len, input_size)
 * @param output        Output sequence (seq_len, hidden_size)
 * @param h_init        Initial hidden state (hidden_size,)
 * @param weights_g1    GRU weights for group 1
 * @param weights_g2    GRU weights for group 2
 * @param seq_len       Sequence length
 * @param bidirectional Whether to use bidirectional GRU
 * @param temp          Temporary buffer
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
) {
    int input_size = weights_g1->input_size + weights_g2->input_size;
    int hidden_size = weights_g1->hidden_size + weights_g2->hidden_size;
    int input_size_g1 = weights_g1->input_size;
    int input_size_g2 = weights_g2->input_size;
    int hidden_size_g1 = weights_g1->hidden_size;
    int hidden_size_g2 = weights_g2->hidden_size;

    // Split input into two groups
    float *input_g1 = (float *)malloc(seq_len * input_size_g1 * sizeof(float));
    float *input_g2 = (float *)malloc(seq_len * input_size_g2 * sizeof(float));

    for (int t = 0; t < seq_len; t++) {
        memcpy(input_g1 + t * input_size_g1,
               input + t * input_size,
               input_size_g1 * sizeof(float));
        memcpy(input_g2 + t * input_size_g2,
               input + t * input_size + input_size_g1,
               input_size_g2 * sizeof(float));
    }

    // Split initial hidden state
    const float *h_init_g1 = h_init;
    const float *h_init_g2 = h_init ? h_init + hidden_size_g1 : NULL;

    // Allocate output buffers for each group
    float *output_g1 = (float *)malloc(seq_len * hidden_size_g1 * sizeof(float));
    float *output_g2 = (float *)malloc(seq_len * hidden_size_g2 * sizeof(float));

    // Process each group independently
    if (bidirectional) {
        // For bidirectional, output size doubles
        int out_size_g1 = hidden_size_g1 * 2;
        int out_size_g2 = hidden_size_g2 * 2;

        free(output_g1);
        free(output_g2);
        output_g1 = (float *)malloc(seq_len * out_size_g1 * sizeof(float));
        output_g2 = (float *)malloc(seq_len * out_size_g2 * sizeof(float));

        // Note: For bidirectional, need separate forward/backward weights
        // This is simplified - in practice you'd pass 4 weight structures
        gru_forward(input_g1, output_g1, h_init_g1, weights_g1, seq_len, temp);
        gru_forward(input_g2, output_g2, h_init_g2, weights_g2, seq_len, temp);

        // Concatenate outputs
        for (int t = 0; t < seq_len; t++) {
            memcpy(output + t * (out_size_g1 + out_size_g2),
                   output_g1 + t * out_size_g1,
                   out_size_g1 * sizeof(float));
            memcpy(output + t * (out_size_g1 + out_size_g2) + out_size_g1,
                   output_g2 + t * out_size_g2,
                   out_size_g2 * sizeof(float));
        }
    } else {
        gru_forward(input_g1, output_g1, h_init_g1, weights_g1, seq_len, temp);
        gru_forward(input_g2, output_g2, h_init_g2, weights_g2, seq_len, temp);

        // Concatenate outputs
        for (int t = 0; t < seq_len; t++) {
            memcpy(output + t * hidden_size,
                   output_g1 + t * hidden_size_g1,
                   hidden_size_g1 * sizeof(float));
            memcpy(output + t * hidden_size + hidden_size_g1,
                   output_g2 + t * hidden_size_g2,
                   hidden_size_g2 * sizeof(float));
        }
    }

    free(input_g1);
    free(input_g2);
    free(output_g1);
    free(output_g2);
}

/* ============================================================================
 * Weight Loading Functions
 * ============================================================================ */

/**
 * Initialize GRU weights structure
 */
GRUWeights* gru_weights_create(int input_size, int hidden_size) {
    GRUWeights *weights = (GRUWeights *)malloc(sizeof(GRUWeights));
    if (!weights) {
        fprintf(stderr, "Error: Failed to allocate GRUWeights structure\n");
        return NULL;
    }

    weights->input_size = input_size;
    weights->hidden_size = hidden_size;

    // Allocate weight matrices
    weights->W_z = (float *)malloc(hidden_size * input_size * sizeof(float));
    weights->U_z = (float *)malloc(hidden_size * hidden_size * sizeof(float));
    weights->b_z = (float *)malloc(hidden_size * sizeof(float));

    weights->W_r = (float *)malloc(hidden_size * input_size * sizeof(float));
    weights->U_r = (float *)malloc(hidden_size * hidden_size * sizeof(float));
    weights->b_r = (float *)malloc(hidden_size * sizeof(float));

    weights->W_h = (float *)malloc(hidden_size * input_size * sizeof(float));
    weights->U_h = (float *)malloc(hidden_size * hidden_size * sizeof(float));
    weights->b_h = (float *)malloc(hidden_size * sizeof(float));

    // Check if any allocation failed
    if (!weights->W_z || !weights->U_z || !weights->b_z ||
        !weights->W_r || !weights->U_r || !weights->b_r ||
        !weights->W_h || !weights->U_h || !weights->b_h) {
        fprintf(stderr, "Error: Failed to allocate GRU weight matrices\n");
        gru_weights_free(weights);
        return NULL;
    }

    // Initialize weights to zero
    memset(weights->W_z, 0, hidden_size * input_size * sizeof(float));
    memset(weights->U_z, 0, hidden_size * hidden_size * sizeof(float));
    memset(weights->b_z, 0, hidden_size * sizeof(float));
    memset(weights->W_r, 0, hidden_size * input_size * sizeof(float));
    memset(weights->U_r, 0, hidden_size * hidden_size * sizeof(float));
    memset(weights->b_r, 0, hidden_size * sizeof(float));
    memset(weights->W_h, 0, hidden_size * input_size * sizeof(float));
    memset(weights->U_h, 0, hidden_size * hidden_size * sizeof(float));
    memset(weights->b_h, 0, hidden_size * sizeof(float));

    return weights;
}

/**
 * Free GRU weights structure
 */
void gru_weights_free(GRUWeights *weights) {
    if (weights) {
        free(weights->W_z);
        free(weights->U_z);
        free(weights->b_z);
        free(weights->W_r);
        free(weights->U_r);
        free(weights->b_r);
        free(weights->W_h);
        free(weights->U_h);
        free(weights->b_h);
        free(weights);
    }
}

/**
 * Load GRU weights from binary file
 * Format: W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h (all float32)
 */
int gru_weights_load(GRUWeights *weights, const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }

    int input_size = weights->input_size;
    int hidden_size = weights->hidden_size;

    // Read weights in order
    fread(weights->W_z, sizeof(float), hidden_size * input_size, fp);
    fread(weights->U_z, sizeof(float), hidden_size * hidden_size, fp);
    fread(weights->b_z, sizeof(float), hidden_size, fp);

    fread(weights->W_r, sizeof(float), hidden_size * input_size, fp);
    fread(weights->U_r, sizeof(float), hidden_size * hidden_size, fp);
    fread(weights->b_r, sizeof(float), hidden_size, fp);

    fread(weights->W_h, sizeof(float), hidden_size * input_size, fp);
    fread(weights->U_h, sizeof(float), hidden_size * hidden_size, fp);
    fread(weights->b_h, sizeof(float), hidden_size, fp);

    fclose(fp);
    return 0;
}

/* ============================================================================
 * Example Usage for GTCRN
 * ============================================================================ */

#ifdef GRU_TEST_MAIN
int main() {
    // Example: GTCRN Intra-RNN (Bidirectional GRNN)
    // Input: (B*T, 97, 16) where 97 is frequency bins, 16 is channels
    // Output: (B*T, 97, 16)

    int seq_len = 97;      // Frequency bins
    int input_size = 16;   // Channels
    int hidden_size = 8;   // Hidden size (16/2 for bidirectional)

    // Create weights for grouped GRU (2 groups)
    GRUWeights *weights_g1 = gru_weights_create(input_size / 2, hidden_size / 2);
    GRUWeights *weights_g2 = gru_weights_create(input_size / 2, hidden_size / 2);

    // Load weights from file (you need to export from PyTorch)
    // gru_weights_load(weights_g1, "intra_rnn_g1.bin");
    // gru_weights_load(weights_g2, "intra_rnn_g2.bin");

    // Allocate input/output buffers
    float *input = (float *)malloc(seq_len * input_size * sizeof(float));
    float *output = (float *)malloc(seq_len * hidden_size * 2 * sizeof(float)); // *2 for bidirectional
    float *temp = (float *)malloc(4 * hidden_size * sizeof(float));

    // Initialize input (example)
    for (int i = 0; i < seq_len * input_size; i++) {
        input[i] = (float)rand() / RAND_MAX;
    }

    // Run GRNN forward pass
    grnn_forward(input, output, NULL, weights_g1, weights_g2, seq_len, 1, temp);

    // Print first few outputs
    printf("Output (first 5 timesteps):\n");
    for (int t = 0; t < 5; t++) {
        printf("t=%d: ", t);
        for (int i = 0; i < hidden_size * 2; i++) {
            printf("%.4f ", output[t * hidden_size * 2 + i]);
        }
        printf("\n");
    }

    // Cleanup
    free(input);
    free(output);
    free(temp);
    gru_weights_free(weights_g1);
    gru_weights_free(weights_g2);

    return 0;
}
#endif
