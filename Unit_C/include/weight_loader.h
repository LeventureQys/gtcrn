#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include "gtcrn_model.h"
#include "conv2d.h"
#include "batchnorm2d.h"
#include "nn_layers.h"
#include "GRU.h"

/**
 * Weight Loader for GTCRN Model
 *
 * Supports loading weights from:
 * 1. Binary format (.bin) - raw float32 arrays
 * 2. NumPy format (.npy) - NumPy saved arrays
 * 3. PyTorch format (.pth) - via Python export script
 */

// ============================================================================
// Binary Format Loading
// ============================================================================

/**
 * Load Conv2d weights from binary file
 * Format: weight[out_ch, in_ch, kH, kW], bias[out_ch] (all float32)
 */
int load_conv2d_weights(Conv2dParams* conv, const char* filename);

/**
 * Load BatchNorm2d weights from binary file
 * Format: weight[num_features], bias[num_features],
 *         running_mean[num_features], running_var[num_features] (all float32)
 */
int load_batchnorm2d_weights(BatchNorm2dParams* bn, const char* filename);

/**
 * Load Linear layer weights from binary file
 * Format: weight[out_features, in_features], bias[out_features] (all float32)
 */
int load_linear_weights(LinearParams* linear, const char* filename);

/**
 * Load PReLU weights from binary file
 * Format: weight[num_parameters] (all float32)
 */
int load_prelu_weights(PReLUParams* prelu, const char* filename);

/**
 * Load LayerNorm weights from binary file
 * Format: weight[normalized_shape], bias[normalized_shape] (all float32)
 */
int load_layernorm_weights(LayerNormParams* ln, const char* filename);

/**
 * Load GRU weights from binary file
 * Format: W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h (all float32)
 */
int load_gru_weights(GRUWeights* gru, const char* filename);

// ============================================================================
// High-Level Model Loading
// ============================================================================

/**
 * Load ConvBlock weights
 * Loads: conv, bn, prelu weights
 */
int load_convblock_weights(ConvBlock* block, const char* prefix);

/**
 * Load GTConvBlock weights
 * Loads: sfe, point_conv1, depth_conv, point_conv2, tra weights
 */
int load_gtconvblock_weights(GTConvBlock* block, const char* prefix);

/**
 * Load Encoder weights
 * Loads: conv1, conv2, gtconv1, gtconv2, gtconv3 weights
 */
int load_encoder_weights(Encoder* encoder, const char* prefix);

/**
 * Load Decoder weights
 * Loads: gtconv1, gtconv2, gtconv3, conv1, conv2 weights
 */
int load_decoder_weights(Decoder* decoder, const char* prefix);

/**
 * Load DPGRNN weights
 * Loads: intra_gru, inter_gru, linear, layernorm weights
 */
int load_dpgrnn_weights(DPGRNN* dpgrnn, const char* prefix);

/**
 * Load complete GTCRN model weights
 * Loads all submodules from a directory containing weight files
 *
 * @param model         GTCRN model structure
 * @param weights_dir   Directory containing weight files
 * @return              0 on success, -1 on error
 */
int load_gtcrn_weights(GTCRN* model, const char* weights_dir);

// ============================================================================
// PyTorch Export Helper
// ============================================================================

/**
 * Export PyTorch model weights to binary format
 * This should be called from Python using the provided export script
 *
 * Python example:
 * ```python
 * import torch
 * import numpy as np
 *
 * model = GTCRN()
 * model.load_state_dict(torch.load('model.pth'))
 *
 * # Export each layer
 * for name, param in model.named_parameters():
 *     np_array = param.detach().cpu().numpy()
 *     np_array.astype(np.float32).tofile(f'weights/{name}.bin')
 * ```
 */

// ============================================================================
// Weight File Format Documentation
// ============================================================================

/*
 * Weight Directory Structure:
 *
 * weights/
 * ├── encoder/
 * │   ├── conv1_weight.bin
 * │   ├── conv1_bias.bin
 * │   ├── bn1_weight.bin
 * │   ├── bn1_bias.bin
 * │   ├── bn1_running_mean.bin
 * │   ├── bn1_running_var.bin
 * │   ├── gtconv1_point_conv1_weight.bin
 * │   ├── ...
 * ├── dpgrnn1/
 * │   ├── intra_gru_g1_W_z.bin
 * │   ├── intra_gru_g1_U_z.bin
 * │   ├── intra_gru_g1_b_z.bin
 * │   ├── ...
 * ├── dpgrnn2/
 * │   ├── ...
 * └── decoder/
 *     ├── ...
 *
 * Each .bin file contains raw float32 data in row-major order
 */

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Load float array from binary file
 */
int load_float_array(float* array, int size, const char* filename);

/**
 * Save float array to binary file (for testing)
 */
int save_float_array(const float* array, int size, const char* filename);

/**
 * Print weight statistics (for debugging)
 */
void print_weight_stats(const float* weights, int size, const char* name);

#endif // WEIGHT_LOADER_H
