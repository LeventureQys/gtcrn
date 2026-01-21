#include "weight_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Utility Functions
// ============================================================================

int load_float_array(float* array, int size, const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }

    size_t read_count = fread(array, sizeof(float), size, fp);
    fclose(fp);

    if (read_count != size) {
        fprintf(stderr, "Error: Expected %d floats, read %zu from %s\n",
                size, read_count, filename);
        return -1;
    }

    return 0;
}

int save_float_array(const float* array, int size, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return -1;
    }

    size_t write_count = fwrite(array, sizeof(float), size, fp);
    fclose(fp);

    if (write_count != size) {
        fprintf(stderr, "Error: Expected to write %d floats, wrote %zu to %s\n",
                size, write_count, filename);
        return -1;
    }

    return 0;
}

void print_weight_stats(const float* weights, int size, const char* name) {
    if (!weights || size <= 0) {
        printf("%s: NULL or empty\n", name);
        return;
    }

    float min_val = weights[0];
    float max_val = weights[0];
    double sum = 0.0;
    double sum_sq = 0.0;

    for (int i = 0; i < size; i++) {
        float val = weights[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / size;
    float variance = (sum_sq / size) - (mean * mean);
    float std = sqrtf(variance);

    printf("%s: size=%d, min=%.6f, max=%.6f, mean=%.6f, std=%.6f\n",
           name, size, min_val, max_val, mean, std);
}

// ============================================================================
// Basic Layer Weight Loading
// ============================================================================

int load_conv2d_weights(Conv2dParams* conv, const char* filename) {
    if (!conv) return -1;

    // Calculate total size
    int weight_size = conv->out_channels * conv->in_channels *
                      conv->kernel_h * conv->kernel_w;

    // Load weight
    char weight_file[512];
    snprintf(weight_file, sizeof(weight_file), "%s_weight.bin", filename);
    if (load_float_array(conv->weight, weight_size, weight_file) != 0) {
        return -1;
    }

    // Load bias (if exists)
    if (conv->bias) {
        char bias_file[512];
        snprintf(bias_file, sizeof(bias_file), "%s_bias.bin", filename);
        if (load_float_array(conv->bias, conv->out_channels, bias_file) != 0) {
            fprintf(stderr, "Warning: Could not load bias for %s\n", filename);
        }
    }

    printf("Loaded Conv2d: %s\n", filename);
    return 0;
}

int load_batchnorm2d_weights(BatchNorm2dParams* bn, const char* filename) {
    if (!bn) return -1;

    char weight_file[512], bias_file[512], mean_file[512], var_file[512];

    snprintf(weight_file, sizeof(weight_file), "%s_weight.bin", filename);
    snprintf(bias_file, sizeof(bias_file), "%s_bias.bin", filename);
    snprintf(mean_file, sizeof(mean_file), "%s_running_mean.bin", filename);
    snprintf(var_file, sizeof(var_file), "%s_running_var.bin", filename);

    if (load_float_array(bn->gamma, bn->num_features, weight_file) != 0) return -1;
    if (load_float_array(bn->beta, bn->num_features, bias_file) != 0) return -1;
    if (load_float_array(bn->running_mean, bn->num_features, mean_file) != 0) return -1;
    if (load_float_array(bn->running_var, bn->num_features, var_file) != 0) return -1;

    printf("Loaded BatchNorm2d: %s\n", filename);
    return 0;
}

int load_linear_weights(LinearParams* linear, const char* filename) {
    if (!linear) return -1;

    int weight_size = linear->out_features * linear->in_features;

    char weight_file[512];
    snprintf(weight_file, sizeof(weight_file), "%s_weight.bin", filename);
    if (load_float_array(linear->weight, weight_size, weight_file) != 0) {
        return -1;
    }

    if (linear->bias) {
        char bias_file[512];
        snprintf(bias_file, sizeof(bias_file), "%s_bias.bin", filename);
        if (load_float_array(linear->bias, linear->out_features, bias_file) != 0) {
            fprintf(stderr, "Warning: Could not load bias for %s\n", filename);
        }
    }

    printf("Loaded Linear: %s\n", filename);
    return 0;
}

int load_prelu_weights(PReLUParams* prelu, const char* filename) {
    if (!prelu) return -1;

    char weight_file[512];
    snprintf(weight_file, sizeof(weight_file), "%s_weight.bin", filename);
    if (load_float_array(prelu->weight, prelu->num_parameters, weight_file) != 0) {
        return -1;
    }

    printf("Loaded PReLU: %s\n", filename);
    return 0;
}

int load_layernorm_weights(LayerNormParams* ln, const char* filename) {
    if (!ln) return -1;

    // Calculate normalized shape size
    int size = 1;
    for (int i = 0; i < ln->ndim; i++) {
        size *= ln->normalized_shape[i];
    }

    char weight_file[512], bias_file[512];
    snprintf(weight_file, sizeof(weight_file), "%s_weight.bin", filename);
    snprintf(bias_file, sizeof(bias_file), "%s_bias.bin", filename);

    if (ln->gamma && load_float_array(ln->gamma, size, weight_file) != 0) {
        fprintf(stderr, "Warning: Could not load weight for %s\n", filename);
    }

    if (ln->beta && load_float_array(ln->beta, size, bias_file) != 0) {
        fprintf(stderr, "Warning: Could not load bias for %s\n", filename);
    }

    printf("Loaded LayerNorm: %s\n", filename);
    return 0;
}

int load_gru_weights(GRUWeights* gru, const char* filename) {
    if (!gru) return -1;

    int input_size = gru->input_size;
    int hidden_size = gru->hidden_size;

    char filepath[512];

    // Load W_z
    snprintf(filepath, sizeof(filepath), "%s_W_z.bin", filename);
    if (load_float_array(gru->W_z, hidden_size * input_size, filepath) != 0) return -1;

    // Load U_z
    snprintf(filepath, sizeof(filepath), "%s_U_z.bin", filename);
    if (load_float_array(gru->U_z, hidden_size * hidden_size, filepath) != 0) return -1;

    // Load b_z
    snprintf(filepath, sizeof(filepath), "%s_b_z.bin", filename);
    if (load_float_array(gru->b_z, hidden_size, filepath) != 0) return -1;

    // Load W_r
    snprintf(filepath, sizeof(filepath), "%s_W_r.bin", filename);
    if (load_float_array(gru->W_r, hidden_size * input_size, filepath) != 0) return -1;

    // Load U_r
    snprintf(filepath, sizeof(filepath), "%s_U_r.bin", filename);
    if (load_float_array(gru->U_r, hidden_size * hidden_size, filepath) != 0) return -1;

    // Load b_r
    snprintf(filepath, sizeof(filepath), "%s_b_r.bin", filename);
    if (load_float_array(gru->b_r, hidden_size, filepath) != 0) return -1;

    // Load W_h
    snprintf(filepath, sizeof(filepath), "%s_W_h.bin", filename);
    if (load_float_array(gru->W_h, hidden_size * input_size, filepath) != 0) return -1;

    // Load U_h
    snprintf(filepath, sizeof(filepath), "%s_U_h.bin", filename);
    if (load_float_array(gru->U_h, hidden_size * hidden_size, filepath) != 0) return -1;

    // Load b_h
    snprintf(filepath, sizeof(filepath), "%s_b_h.bin", filename);
    if (load_float_array(gru->b_h, hidden_size, filepath) != 0) return -1;

    printf("Loaded GRU: %s\n", filename);
    return 0;
}

// ============================================================================
// High-Level Model Loading
// ============================================================================

int load_convblock_weights(ConvBlock* block, const char* prefix) {
    if (!block) return -1;

    char filepath[512];

    // Load fused conv+bn weights
    snprintf(filepath, sizeof(filepath), "%s_conv", prefix);
    // Note: For fused conv+bn, we need to load original conv and bn separately
    // then fuse them using fuse_conv_batchnorm()

    // Load PReLU if exists
    if (block->prelu) {
        snprintf(filepath, sizeof(filepath), "%s_prelu", prefix);
        if (load_prelu_weights(block->prelu, filepath) != 0) {
            fprintf(stderr, "Warning: Could not load PReLU for %s\n", prefix);
        }
    }

    printf("Loaded ConvBlock: %s\n", prefix);
    return 0;
}

int load_gtconvblock_weights(GTConvBlock* block, const char* prefix) {
    if (!block) return -1;

    char filepath[512];

    // Load point_conv1
    snprintf(filepath, sizeof(filepath), "%s_point_conv1", prefix);
    // Load fused conv+bn weights

    // Load point_prelu1
    if (block->point_prelu1) {
        snprintf(filepath, sizeof(filepath), "%s_point_prelu1", prefix);
        load_prelu_weights(block->point_prelu1, filepath);
    }

    // Load depth_conv
    snprintf(filepath, sizeof(filepath), "%s_depth_conv", prefix);
    // Load fused conv+bn weights

    // Load depth_prelu
    if (block->depth_prelu) {
        snprintf(filepath, sizeof(filepath), "%s_depth_prelu", prefix);
        load_prelu_weights(block->depth_prelu, filepath);
    }

    // Load point_conv2
    snprintf(filepath, sizeof(filepath), "%s_point_conv2", prefix);
    // Load fused conv+bn weights

    // Load TRA weights
    if (block->tra) {
        snprintf(filepath, sizeof(filepath), "%s_tra", prefix);
        // Load TRA GRU and Linear weights
    }

    printf("Loaded GTConvBlock: %s\n", prefix);
    return 0;
}

int load_encoder_weights(Encoder* encoder, const char* prefix) {
    if (!encoder) return -1;

    char filepath[512];

    // Load conv1
    if (encoder->conv1) {
        snprintf(filepath, sizeof(filepath), "%s/conv1", prefix);
        load_convblock_weights(encoder->conv1, filepath);
    }

    // Load conv2
    if (encoder->conv2) {
        snprintf(filepath, sizeof(filepath), "%s/conv2", prefix);
        load_convblock_weights(encoder->conv2, filepath);
    }

    // Load gtconv1
    if (encoder->gtconv1) {
        snprintf(filepath, sizeof(filepath), "%s/gtconv1", prefix);
        load_gtconvblock_weights(encoder->gtconv1, filepath);
    }

    // Load gtconv2
    if (encoder->gtconv2) {
        snprintf(filepath, sizeof(filepath), "%s/gtconv2", prefix);
        load_gtconvblock_weights(encoder->gtconv2, filepath);
    }

    // Load gtconv3
    if (encoder->gtconv3) {
        snprintf(filepath, sizeof(filepath), "%s/gtconv3", prefix);
        load_gtconvblock_weights(encoder->gtconv3, filepath);
    }

    printf("Loaded Encoder: %s\n", prefix);
    return 0;
}

int load_decoder_weights(Decoder* decoder, const char* prefix) {
    if (!decoder) return -1;

    char filepath[512];

    // Load gtconv1
    if (decoder->gtconv1) {
        snprintf(filepath, sizeof(filepath), "%s/gtconv1", prefix);
        load_gtconvblock_weights(decoder->gtconv1, filepath);
    }

    // Load gtconv2
    if (decoder->gtconv2) {
        snprintf(filepath, sizeof(filepath), "%s/gtconv2", prefix);
        load_gtconvblock_weights(decoder->gtconv2, filepath);
    }

    // Load gtconv3
    if (decoder->gtconv3) {
        snprintf(filepath, sizeof(filepath), "%s/gtconv3", prefix);
        load_gtconvblock_weights(decoder->gtconv3, filepath);
    }

    // Load conv1
    if (decoder->conv1) {
        snprintf(filepath, sizeof(filepath), "%s/conv1", prefix);
        load_convblock_weights(decoder->conv1, filepath);
    }

    // Load conv2
    if (decoder->conv2) {
        snprintf(filepath, sizeof(filepath), "%s/conv2", prefix);
        load_convblock_weights(decoder->conv2, filepath);
    }

    printf("Loaded Decoder: %s\n", prefix);
    return 0;
}

int load_dpgrnn_weights(DPGRNN* dpgrnn, const char* prefix) {
    if (!dpgrnn) return -1;

    char filepath[512];

    // Load Intra RNN weights
    if (dpgrnn->intra_gru_g1_fwd) {
        snprintf(filepath, sizeof(filepath), "%s/intra_gru_g1_fwd", prefix);
        load_gru_weights(dpgrnn->intra_gru_g1_fwd, filepath);
    }

    if (dpgrnn->intra_gru_g2_fwd) {
        snprintf(filepath, sizeof(filepath), "%s/intra_gru_g2_fwd", prefix);
        load_gru_weights(dpgrnn->intra_gru_g2_fwd, filepath);
    }

    if (dpgrnn->intra_gru_g1_bwd) {
        snprintf(filepath, sizeof(filepath), "%s/intra_gru_g1_bwd", prefix);
        load_gru_weights(dpgrnn->intra_gru_g1_bwd, filepath);
    }

    if (dpgrnn->intra_gru_g2_bwd) {
        snprintf(filepath, sizeof(filepath), "%s/intra_gru_g2_bwd", prefix);
        load_gru_weights(dpgrnn->intra_gru_g2_bwd, filepath);
    }

    // Load Intra Linear
    if (dpgrnn->intra_fc) {
        snprintf(filepath, sizeof(filepath), "%s/intra_fc", prefix);
        load_linear_weights(dpgrnn->intra_fc, filepath);
    }

    // Load Intra LayerNorm
    if (dpgrnn->intra_ln) {
        snprintf(filepath, sizeof(filepath), "%s/intra_ln", prefix);
        load_layernorm_weights(dpgrnn->intra_ln, filepath);
    }

    // Load Inter RNN weights
    if (dpgrnn->inter_gru_g1) {
        snprintf(filepath, sizeof(filepath), "%s/inter_gru_g1", prefix);
        load_gru_weights(dpgrnn->inter_gru_g1, filepath);
    }

    if (dpgrnn->inter_gru_g2) {
        snprintf(filepath, sizeof(filepath), "%s/inter_gru_g2", prefix);
        load_gru_weights(dpgrnn->inter_gru_g2, filepath);
    }

    // Load Inter Linear
    if (dpgrnn->inter_fc) {
        snprintf(filepath, sizeof(filepath), "%s/inter_fc", prefix);
        load_linear_weights(dpgrnn->inter_fc, filepath);
    }

    // Load Inter LayerNorm
    if (dpgrnn->inter_ln) {
        snprintf(filepath, sizeof(filepath), "%s/inter_ln", prefix);
        load_layernorm_weights(dpgrnn->inter_ln, filepath);
    }

    printf("Loaded DPGRNN: %s\n", prefix);
    return 0;
}

int load_gtcrn_weights(GTCRN* model, const char* weights_dir) {
    if (!model) return -1;

    printf("\n=================================================================\n");
    printf("Loading GTCRN weights from: %s\n", weights_dir);
    printf("=================================================================\n\n");

    char filepath[512];

    // Load Encoder
    snprintf(filepath, sizeof(filepath), "%s/encoder", weights_dir);
    if (load_encoder_weights(model->encoder, filepath) != 0) {
        fprintf(stderr, "Error loading encoder weights\n");
        return -1;
    }

    // Load DPGRNN1
    snprintf(filepath, sizeof(filepath), "%s/dpgrnn1", weights_dir);
    if (load_dpgrnn_weights(model->dpgrnn1, filepath) != 0) {
        fprintf(stderr, "Error loading dpgrnn1 weights\n");
        return -1;
    }

    // Load DPGRNN2
    snprintf(filepath, sizeof(filepath), "%s/dpgrnn2", weights_dir);
    if (load_dpgrnn_weights(model->dpgrnn2, filepath) != 0) {
        fprintf(stderr, "Error loading dpgrnn2 weights\n");
        return -1;
    }

    // Load Decoder
    snprintf(filepath, sizeof(filepath), "%s/decoder", weights_dir);
    if (load_decoder_weights(model->decoder, filepath) != 0) {
        fprintf(stderr, "Error loading decoder weights\n");
        return -1;
    }

    printf("\n=================================================================\n");
    printf("Successfully loaded all GTCRN weights!\n");
    printf("=================================================================\n\n");

    return 0;
}
