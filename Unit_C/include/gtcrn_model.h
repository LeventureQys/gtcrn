#ifndef GTCRN_MODEL_H
#define GTCRN_MODEL_H

#include "conv2d.h"
#include "batchnorm2d.h"
#include "nn_layers.h"
#include "layernorm.h"
#include "gtcrn_modules.h"  // ERB, SFE, TRA

/*
 * GTCRN 完整模型
 *
 * 网络结构:
 *   Input: (B, F, T, 2) - 复数频谱
 *   1. ERB 压缩: 769 bins -> 385 bins
 *   2. SFE: 提取子带特征
 *   3. Encoder: 5 层卷积 (385 -> 97 bins)
 *   4. DPGRNN: 2 层双路径 RNN
 *   5. Decoder: 5 层反卷积 (97 -> 385 bins)
 *   6. ERB 恢复: 385 bins -> 769 bins
 *   7. Mask: 复数掩码
 *   Output: (B, F, T, 2) - 增强后的复数频谱
 */

// ============================================================================
// ConvBlock - Conv + BN + Activation
// ============================================================================

typedef struct {
    FusedConvBN fused_conv_bn;  // 融合的 Conv + BN
    PReLUParams* prelu;         // PReLU 激活
    int use_tanh;               // 是否使用 Tanh（最后一层）
} ConvBlock;

ConvBlock* convblock_create(
    const Conv2dParams* conv_params,
    const BatchNorm2dParams* bn_params,
    const PReLUParams* prelu_params,
    int use_tanh
);

void convblock_forward(
    const Tensor* input,
    Tensor* output,
    ConvBlock* block
);

void convblock_free(ConvBlock* block);


// ============================================================================
// GTConvBlock - Group Temporal Convolution Block
// ============================================================================

typedef struct {
    SFEParams* sfe;                 // SFE 模块

    FusedConvBN point_conv1;        // Point Conv1 + BN
    PReLUParams* point_prelu1;

    FusedConvBN depth_conv;         // Depth Conv + BN
    PReLUParams* depth_prelu;

    FusedConvBN point_conv2;        // Point Conv2 + BN

    TRAParams* tra;                 // TRA 模块
    int use_tra;                    // 是否使用 TRA
} GTConvBlock;

GTConvBlock* gtconvblock_create(
    int in_channels,
    int hidden_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int use_deconv
);

void gtconvblock_forward(
    const Tensor* input,
    Tensor* output,
    GTConvBlock* block
);

/*
 * GTConvBlock 流式前向传播
 *
 * conv_cache: (B, C, cache_size, F) - 深度卷积的缓存
 * tra_cache: (1, B, C) - TRA模块的GRU隐藏状态缓存
 */
void gtconvblock_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,
    float* tra_cache,
    GTConvBlock* block,
    int kernel_h,
    int dilation_h
);

void gtconvblock_free(GTConvBlock* block);


// ============================================================================
// Encoder
// ============================================================================

typedef struct {
    ConvBlock* conv1;       // (9, 16, (1,5), stride=(1,2))
    ConvBlock* conv2;       // (16, 16, (1,5), stride=(1,2), groups=2)
    GTConvBlock* gtconv1;   // dilation=(1,1)
    GTConvBlock* gtconv2;   // dilation=(2,1)
    GTConvBlock* gtconv3;   // dilation=(5,1)
} Encoder;

Encoder* encoder_create();

void encoder_forward(
    const Tensor* input,
    Tensor* output,
    Tensor** skip_connections,  // 5 个跳跃连接
    Encoder* encoder
);

void encoder_free(Encoder* encoder);


// ============================================================================
// Decoder
// ============================================================================

typedef struct {
    GTConvBlock* gtconv1;   // dilation=(5,1), deconv
    GTConvBlock* gtconv2;   // dilation=(2,1), deconv
    GTConvBlock* gtconv3;   // dilation=(1,1), deconv
    ConvBlock* conv1;       // (16, 16, (1,5), stride=(1,2), groups=2, deconv)
    ConvBlock* conv2;       // (16, 2, (1,5), stride=(1,2), deconv, tanh)
} Decoder;

Decoder* decoder_create();

void decoder_forward(
    const Tensor* input,
    Tensor** skip_connections,  // 5 个跳跃连接
    Tensor* output,
    Decoder* decoder
);

void decoder_free(Decoder* decoder);


// ============================================================================
// DPGRNN - Dual-Path Grouped RNN
// ============================================================================

#include "GRU.h"

/*
 * DPGRNN 使用 Grouped GRU (GRNN) 实现双路径处理
 *
 * Intra-RNN: 双向 GRNN，处理频率维度
 *   - Input: (B*T, F, C) where F=97, C=16
 *   - Output: (B*T, F, C)
 *   - Bidirectional, hidden_size = C/2 = 8
 *
 * Inter-RNN: 单向 GRNN，处理时间维度
 *   - Input: (B*F, T, C) where F=97, C=16
 *   - Output: (B*F, T, C)
 *   - Unidirectional, hidden_size = C = 16
 */

typedef struct {
    int input_size;     // 16
    int width;          // 97 (frequency bins after encoder)
    int hidden_size;    // 16

    // Intra RNN (Bidirectional GRNN)
    GRUWeights* intra_gru_g1_fwd;   // Group 1 forward
    GRUWeights* intra_gru_g2_fwd;   // Group 2 forward
    GRUWeights* intra_gru_g1_bwd;   // Group 1 backward
    GRUWeights* intra_gru_g2_bwd;   // Group 2 backward
    LinearParams* intra_fc;
    LayerNormParams* intra_ln;

    // Inter RNN (Unidirectional GRNN)
    GRUWeights* inter_gru_g1;       // Group 1
    GRUWeights* inter_gru_g2;       // Group 2
    LinearParams* inter_fc;
    LayerNormParams* inter_ln;
} DPGRNN;

DPGRNN* dpgrnn_create(int input_size, int width, int hidden_size);

void dpgrnn_forward(
    const Tensor* input,
    Tensor* output,
    DPGRNN* dpgrnn
);

/*
 * DPGRNN 流式前向传播
 *
 * inter_cache: (1, B*F, hidden_size) - Inter-RNN的隐藏状态缓存
 */
void dpgrnn_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* inter_cache,
    DPGRNN* dpgrnn
);

void dpgrnn_free(DPGRNN* dpgrnn);


// ============================================================================
// GTCRN 完整模型
// ============================================================================

typedef struct {
    // ERB 模块
    ERBParams* erb;

    // SFE 模块
    SFEParams* sfe;

    // Encoder
    Encoder* encoder;

    // DPGRNN
    DPGRNN* dpgrnn1;
    DPGRNN* dpgrnn2;

    // Decoder
    Decoder* decoder;

    // 工作缓冲区
    Tensor* work_buffers[10];
    int num_buffers;
} GTCRN;

/*
 * 创建 GTCRN 模型
 */
GTCRN* gtcrn_create();

/*
 * GTCRN 前向传播（批处理模式）
 *
 * Input:  spec (B, F, T, 2) - 复数频谱，F=769 for 48kHz
 * Output: spec_enh (B, F, T, 2) - 增强后的复数频谱
 */
void gtcrn_forward(
    const float* spec_input,    // (B, F, T, 2)
    float* spec_output,         // (B, F, T, 2)
    int batch,
    int freq_bins,              // F=769
    int time_frames,            // T
    GTCRN* model
);

/*
 * GTCRN 流式前向传播（实时处理模式）
 *
 * Input:  spec (B, F, 1, 2) - 单帧复数频谱
 * Output: spec_enh (B, F, 1, 2) - 增强后的单帧复数频谱
 *
 * 状态缓存布局:
 * - conv_cache: 编码器和解码器的卷积缓存
 *   [encoder_conv_caches, decoder_conv_caches]
 *   每个GTConvBlock需要 (B, C, cache_size, F) 的缓存
 *
 * - tra_cache: 编码器和解码器的TRA缓存
 *   [encoder_tra_caches, decoder_tra_caches]
 *   每个TRA需要 (1, B, C) 的缓存
 *
 * - inter_cache: DPGRNN的Inter-RNN缓存
 *   [dpgrnn1_cache, dpgrnn2_cache]
 *   每个DPGRNN需要 (1, B*F, hidden_size) 的缓存
 */
void gtcrn_forward_stream(
    const float* spec_input,    // (B, F, 1, 2) - T=1
    float* spec_output,         // (B, F, 1, 2)
    float* conv_cache,          // 卷积缓存
    float* tra_cache,           // TRA缓存
    float* inter_cache,         // Inter-RNN缓存
    int batch,
    int freq_bins,
    GTCRN* model
);

/*
 * 释放 GTCRN 模型
 */
void gtcrn_free(GTCRN* model);


// ============================================================================
// 辅助函数
// ============================================================================

/*
 * 计算复数幅度
 */
void compute_magnitude(
    const float* real,
    const float* imag,
    float* magnitude,
    int size
);

/*
 * 应用复数掩码
 * s_real = spec_real * mask_real - spec_imag * mask_imag
 * s_imag = spec_imag * mask_real + spec_real * mask_imag
 */
void apply_complex_mask(
    const float* spec_real,
    const float* spec_imag,
    const float* mask_real,
    const float* mask_imag,
    float* output_real,
    float* output_imag,
    int size
);

/*
 * 打印模型信息
 */
void print_gtcrn_info(const GTCRN* model);

#endif
