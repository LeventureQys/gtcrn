#ifndef BATCHNORM2D_H
#define BATCHNORM2D_H

#include "conv2d.h"

/*
 * BatchNorm2d Implementation
 *
 * Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
 *
 * During inference (GTCRN use case):
 * - Uses pre-computed running_mean and running_var
 * - No gradient computation needed
 */

typedef struct {
    int num_features;
    float* gamma;           // Scale parameter (num_features)
    float* beta;            // Shift parameter (num_features)
    float* running_mean;    // Running mean (num_features)
    float* running_var;     // Running variance (num_features)
    float eps;              // Small constant for numerical stability
} BatchNorm2dParams;

/*
 * Standard BatchNorm2d forward pass
 * Input/Output: (B, C, H, W)
 */
void batchnorm2d_forward(
    Tensor* input,
    const BatchNorm2dParams* params
);

/*
 * Fused Conv2d + BatchNorm2d
 *
 * During inference, BatchNorm can be fused into Conv2d:
 *
 * Original:
 *   y = conv(x)
 *   z = bn(y) = gamma * (y - mean) / sqrt(var + eps) + beta
 *
 * Fused:
 *   z = conv_fused(x)
 *   where:
 *     weight_fused = weight * gamma / sqrt(var + eps)
 *     bias_fused = (bias - mean) * gamma / sqrt(var + eps) + beta
 *
 * Benefits:
 * - Eliminates one pass over the data
 * - Reduces memory bandwidth
 * - Faster inference (1.5-2x speedup)
 */

typedef struct {
    Conv2dParams conv_params;
    float* fused_weight;    // Pre-computed fused weights
    float* fused_bias;      // Pre-computed fused bias
    int is_fused;           // Flag indicating if fusion is done
} FusedConvBN;

/*
 * Fuse BatchNorm parameters into Conv2d weights
 * This should be done once during model loading
 */
void fuse_conv_batchnorm(
    FusedConvBN* fused,
    const Conv2dParams* conv_params,
    const BatchNorm2dParams* bn_params
);

/*
 * Forward pass with fused Conv+BN
 * Much faster than separate operations
 */
void fused_conv_bn_forward(
    const Tensor* input,
    Tensor* output,
    FusedConvBN* fused
);

/*
 * Free fused parameters
 */
void fused_conv_bn_free(FusedConvBN* fused);

/*
 * Create BatchNorm2d parameters from arrays
 */
BatchNorm2dParams* batchnorm2d_create(
    int num_features,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float eps
);

/*
 * Free BatchNorm2d parameters
 */
void batchnorm2d_free(BatchNorm2dParams* params);

#endif
