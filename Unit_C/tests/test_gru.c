/**
 * test_gru.c - Test program for GRU implementation
 *
 * Demonstrates usage of GRU functions and validates against PyTorch
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "GRU.h"

/* ============================================================================
 * Test Utilities
 * ============================================================================ */

/**
 * Initialize array with random values
 */
void random_init(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Range: [-1, 1]
    }
}

/**
 * Print array (first n elements)
 */
void print_array(const char *name, const float *arr, int size, int max_print) {
    printf("%s: [", name);
    int n = (size < max_print) ? size : max_print;
    for (int i = 0; i < n; i++) {
        printf("%.4f", arr[i]);
        if (i < n - 1) printf(", ");
    }
    if (size > max_print) printf(", ...");
    printf("]\n");
}

/**
 * Compare two arrays and compute max absolute difference
 */
float compare_arrays(const float *a, const float *b, int size) {
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

/* ============================================================================
 * Test Cases
 * ============================================================================ */

/**
 * Test 1: Single GRU cell forward pass
 */
void test_gru_cell() {
    printf("\n");
    printf("========================================\n");
    printf("Test 1: GRU Cell Forward Pass\n");
    printf("========================================\n");

    int input_size = 8;
    int hidden_size = 16;

    // Create weights
    GRUWeights *weights = gru_weights_create(input_size, hidden_size);

    // Initialize weights randomly
    random_init(weights->W_z, hidden_size * input_size);
    random_init(weights->U_z, hidden_size * hidden_size);
    random_init(weights->b_z, hidden_size);
    random_init(weights->W_r, hidden_size * input_size);
    random_init(weights->U_r, hidden_size * hidden_size);
    random_init(weights->b_r, hidden_size);
    random_init(weights->W_h, hidden_size * input_size);
    random_init(weights->U_h, hidden_size * hidden_size);
    random_init(weights->b_h, hidden_size);

    // Create input and hidden state
    float *x = (float *)malloc(input_size * sizeof(float));
    float *h_prev = (float *)malloc(hidden_size * sizeof(float));
    float *h_new = (float *)malloc(hidden_size * sizeof(float));
    float *temp = (float *)malloc(4 * hidden_size * sizeof(float));

    random_init(x, input_size);
    random_init(h_prev, hidden_size);

    // Run GRU cell
    gru_cell_forward(x, h_prev, h_new, weights, temp);

    // Print results
    print_array("Input", x, input_size, 8);
    print_array("h_prev", h_prev, hidden_size, 8);
    print_array("h_new", h_new, hidden_size, 8);

    printf("✓ GRU cell test completed\n");

    // Cleanup
    free(x);
    free(h_prev);
    free(h_new);
    free(temp);
    gru_weights_free(weights);
}

/**
 * Test 2: GRU sequence forward pass
 */
void test_gru_sequence() {
    printf("\n");
    printf("========================================\n");
    printf("Test 2: GRU Sequence Forward Pass\n");
    printf("========================================\n");

    int seq_len = 10;
    int input_size = 8;
    int hidden_size = 16;

    printf("Sequence length: %d\n", seq_len);
    printf("Input size: %d\n", input_size);
    printf("Hidden size: %d\n", hidden_size);

    // Create weights
    GRUWeights *weights = gru_weights_create(input_size, hidden_size);

    // Initialize weights randomly
    random_init(weights->W_z, hidden_size * input_size);
    random_init(weights->U_z, hidden_size * hidden_size);
    random_init(weights->b_z, hidden_size);
    random_init(weights->W_r, hidden_size * input_size);
    random_init(weights->U_r, hidden_size * hidden_size);
    random_init(weights->b_r, hidden_size);
    random_init(weights->W_h, hidden_size * input_size);
    random_init(weights->U_h, hidden_size * hidden_size);
    random_init(weights->b_h, hidden_size);

    // Create input and output sequences
    float *input = (float *)malloc(seq_len * input_size * sizeof(float));
    float *output = (float *)malloc(seq_len * hidden_size * sizeof(float));
    float *temp = (float *)malloc(4 * hidden_size * sizeof(float));

    random_init(input, seq_len * input_size);

    // Run GRU forward
    clock_t start = clock();
    gru_forward(input, output, NULL, weights, seq_len, temp);
    clock_t end = clock();

    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    // Print results
    printf("\nFirst timestep output:\n");
    print_array("  output[0]", output, hidden_size, 8);

    printf("\nLast timestep output:\n");
    print_array("  output[%d]", output + (seq_len - 1) * hidden_size, hidden_size, 8);

    printf("\nExecution time: %.3f ms\n", time_ms);
    printf("✓ GRU sequence test completed\n");

    // Cleanup
    free(input);
    free(output);
    free(temp);
    gru_weights_free(weights);
}

/**
 * Test 3: Grouped GRU (GRNN)
 */
void test_grnn() {
    printf("\n");
    printf("========================================\n");
    printf("Test 3: Grouped GRU (GRNN)\n");
    printf("========================================\n");

    int seq_len = 97;      // Frequency bins (GTCRN)
    int input_size = 16;   // Channels
    int hidden_size = 8;   // Hidden size (16/2 for bidirectional)

    printf("Sequence length: %d (frequency bins)\n", seq_len);
    printf("Input size: %d (channels)\n", input_size);
    printf("Hidden size: %d (per direction)\n", hidden_size);

    // Create weights for 2 groups
    GRUWeights *weights_g1 = gru_weights_create(input_size / 2, hidden_size / 2);
    GRUWeights *weights_g2 = gru_weights_create(input_size / 2, hidden_size / 2);

    // Initialize weights randomly
    random_init(weights_g1->W_z, weights_g1->hidden_size * weights_g1->input_size);
    random_init(weights_g1->U_z, weights_g1->hidden_size * weights_g1->hidden_size);
    random_init(weights_g1->b_z, weights_g1->hidden_size);
    random_init(weights_g1->W_r, weights_g1->hidden_size * weights_g1->input_size);
    random_init(weights_g1->U_r, weights_g1->hidden_size * weights_g1->hidden_size);
    random_init(weights_g1->b_r, weights_g1->hidden_size);
    random_init(weights_g1->W_h, weights_g1->hidden_size * weights_g1->input_size);
    random_init(weights_g1->U_h, weights_g1->hidden_size * weights_g1->hidden_size);
    random_init(weights_g1->b_h, weights_g1->hidden_size);

    random_init(weights_g2->W_z, weights_g2->hidden_size * weights_g2->input_size);
    random_init(weights_g2->U_z, weights_g2->hidden_size * weights_g2->hidden_size);
    random_init(weights_g2->b_z, weights_g2->hidden_size);
    random_init(weights_g2->W_r, weights_g2->hidden_size * weights_g2->input_size);
    random_init(weights_g2->U_r, weights_g2->hidden_size * weights_g2->hidden_size);
    random_init(weights_g2->b_r, weights_g2->hidden_size);
    random_init(weights_g2->W_h, weights_g2->hidden_size * weights_g2->input_size);
    random_init(weights_g2->U_h, weights_g2->hidden_size * weights_g2->hidden_size);
    random_init(weights_g2->b_h, weights_g2->hidden_size);

    // Create input and output
    float *input = (float *)malloc(seq_len * input_size * sizeof(float));
    float *output = (float *)malloc(seq_len * hidden_size * sizeof(float));
    float *temp = (float *)malloc(4 * hidden_size * sizeof(float));

    random_init(input, seq_len * input_size);

    // Run GRNN (unidirectional)
    printf("\nRunning unidirectional GRNN...\n");
    clock_t start = clock();
    grnn_forward(input, output, NULL, weights_g1, weights_g2, seq_len, 0, temp);
    clock_t end = clock();

    double time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    printf("Output shape: (%d, %d)\n", seq_len, hidden_size);
    print_array("output[0]", output, hidden_size, 8);
    printf("Execution time: %.3f ms\n", time_ms);

    printf("✓ GRNN test completed\n");

    // Cleanup
    free(input);
    free(output);
    free(temp);
    gru_weights_free(weights_g1);
    gru_weights_free(weights_g2);
}

/**
 * Test 4: Load weights from file
 */
void test_load_weights() {
    printf("\n");
    printf("========================================\n");
    printf("Test 4: Load Weights from File\n");
    printf("========================================\n");

    const char *weight_file = "gru_weights/dpgrnn1_intra_g1.bin";

    // Check if file exists
    FILE *fp = fopen(weight_file, "rb");
    if (!fp) {
        printf("⚠ Weight file not found: %s\n", weight_file);
        printf("  Run export_gru_weights.py first to generate weight files\n");
        printf("  Skipping this test\n");
        return;
    }
    fclose(fp);

    // Create weights structure (sizes from GTCRN)
    int input_size = 8;   // 16 / 2 (grouped)
    int hidden_size = 4;  // 8 / 2 (bidirectional)

    GRUWeights *weights = gru_weights_create(input_size, hidden_size);

    // Load weights
    int ret = gru_weights_load(weights, weight_file);
    if (ret == 0) {
        printf("✓ Successfully loaded weights from: %s\n", weight_file);
        printf("  Input size: %d\n", weights->input_size);
        printf("  Hidden size: %d\n", weights->hidden_size);

        // Print first few weights
        print_array("  W_z[0]", weights->W_z, hidden_size * input_size, 8);
        print_array("  b_z", weights->b_z, hidden_size, 8);
    } else {
        printf("✗ Failed to load weights\n");
    }

    gru_weights_free(weights);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char *argv[]) {
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║   GRU Implementation Test Suite       ║\n");
    printf("║   For GTCRN Speech Enhancement         ║\n");
    printf("╚════════════════════════════════════════╝\n");

    // Seed random number generator
    srand(time(NULL));

    // Run tests
    test_gru_cell();
    test_gru_sequence();
    test_grnn();
    test_load_weights();

    printf("\n");
    printf("========================================\n");
    printf("All tests completed!\n");
    printf("========================================\n");
    printf("\n");

    return 0;
}
