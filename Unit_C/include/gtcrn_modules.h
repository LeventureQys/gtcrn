#ifndef GTCRN_MODULES_H
#define GTCRN_MODULES_H

#include "nn_layers.h"
#include "GRU.h"  // 使用完整的GRU实现

/*
 * GTCRN 特定模块的 C 实现
 * - ERB (Equivalent Rectangular Bandwidth)
 * - SFE (Subband Feature Extraction)
 * - TRA (Temporal Recurrent Attention)
 */

// ============================================================================
// ERB - Equivalent Rectangular Bandwidth
// ============================================================================

/*
 * ERB 模块用于频率压缩和恢复
 *
 * 从 gtcrn1.py lines 11-61:
 *   - 将 769 bins 压缩到 385 bins
 *   - 低频部分保持不变 (195 bins)
 *   - 高频部分使用 ERB 滤波器组压缩 (190 bins)
 *   - 总共: 195 + 190 = 385 bins
 */

typedef struct {
    int erb_subband_1;      // 低频保持不变的 bins 数量 (195)
    int erb_subband_2;      // ERB 压缩后的 bins 数量 (190)
    int nfft;               // FFT 大小 (1536)
    int nfreqs;             // 频率 bins 数量 (769 = nfft/2+1)

    // ERB 滤波器组
    float* erb_filters;     // (erb_subband_2, nfreqs-erb_subband_1)
    float* ierb_filters;    // 转置，用于恢复

    // Linear 层参数（实际上是矩阵乘法）
    LinearParams* erb_fc;   // 压缩
    LinearParams* ierb_fc;  // 恢复
} ERBParams;

/*
 * 创建 ERB 参数
 */
ERBParams* erb_create(
    int erb_subband_1,      // 195
    int erb_subband_2,      // 190
    int nfft,               // 1536
    int high_lim,           // 24000 Hz
    int fs                  // 48000 Hz
);

/*
 * ERB 压缩 (bm - bandwidth mapping)
 * Input:  (B, C, T, F) where F=769
 * Output: (B, C, T, F_erb) where F_erb=385
 */
void erb_compress(
    const Tensor* input,    // (B, C, T, 769)
    Tensor* output,         // (B, C, T, 385)
    const ERBParams* params
);

/*
 * ERB 恢复 (bs - bandwidth synthesis)
 * Input:  (B, C, T, F_erb) where F_erb=385
 * Output: (B, C, T, F) where F=769
 */
void erb_decompress(
    const Tensor* input,    // (B, C, T, 385)
    Tensor* output,         // (B, C, T, 769)
    const ERBParams* params
);

/*
 * 释放 ERB 参数
 */
void erb_free(ERBParams* params);


// ============================================================================
// SFE - Subband Feature Extraction
// ============================================================================

/*
 * SFE 模块使用 Unfold 提取子带特征
 *
 * 从 gtcrn1.py lines 64-74:
 *   - 使用 Unfold 在频率维度上提取邻域特征
 *   - kernel_size=3: 每个频率位置提取 3 个邻域值
 *   - 输入: (B, C, T, F)
 *   - 输出: (B, C*3, T, F)
 */

typedef struct {
    int kernel_size;        // 通常为 3
    UnfoldParams unfold_params;
} SFEParams;

/*
 * 创建 SFE 参数
 */
SFEParams* sfe_create(int kernel_size, int stride);

/*
 * SFE 前向传播
 * Input:  (B, C, T, F)
 * Output: (B, C*kernel_size, T, F)
 */
void sfe_forward(
    const Tensor* input,
    Tensor* output,
    const SFEParams* params
);

/*
 * 释放 SFE 参数
 */
void sfe_free(SFEParams* params);


// ============================================================================
// TRA - Temporal Recurrent Attention
// ============================================================================

/*
 * TRA 模块使用 GRU 生成时间注意力权重
 *
 * 从 gtcrn1.py lines 77-93:
 *   1. 计算能量: zt = mean(x^2, dim=-1)  # (B,C,T)
 *   2. GRU: at = GRU(zt)  # (B,C,T) -> (B,C*2,T)
 *   3. Linear: at = Linear(at)  # (B,C*2,T) -> (B,C,T)
 *   4. Sigmoid: at = Sigmoid(at)  # (B,C,T)
 *   5. 应用注意力: output = x * at[..., None]  # (B,C,T,F)
 */

// GRU 参数（简化版本）
typedef struct {
    int input_size;
    int hidden_size;
    int num_layers;
    int bidirectional;

    // GRU 权重
    float* weight_ih;       // input-hidden
    float* weight_hh;       // hidden-hidden
    float* bias_ih;
    float* bias_hh;
} GRUParams;

typedef struct {
    int channels;

    // GRU 层 - 使用GRU.h中的完整实现
    GRUWeights* att_gru_weights;  // (channels, channels*2)

    // Linear 层
    LinearParams* att_fc;   // (channels*2, channels)

    // Sigmoid 激活（无参数）
} TRAParams;

/*
 * 创建 TRA 参数
 */
TRAParams* tra_create(int channels);

/*
 * TRA 前向传播（批处理版本）
 * Input:  (B, C, T, F)
 * Output: (B, C, T, F) - 应用注意力权重后
 */
void tra_forward(
    const Tensor* input,
    Tensor* output,
    TRAParams* params
);

/*
 * TRA 流式前向传播（支持状态缓存）
 * Input:  (B, C, T, F) - 通常 T=1 用于实时处理
 * Output: (B, C, T, F) - 应用注意力权重后
 * h_cache: (1, B, channels*2) - GRU隐藏状态缓存，会被更新
 *
 * 用于实时流式处理，保持GRU的隐藏状态在帧之间传递
 */
void tra_forward_stream(
    const Tensor* input,
    Tensor* output,
    float* h_cache,
    TRAParams* params
);

/*
 * 释放 TRA 参数
 */
void tra_free(TRAParams* params);


// ============================================================================
// GRU - Gated Recurrent Unit
// ============================================================================

/*
 * GRU 前向传播（简化版本）
 *
 * Input:  (batch, seq_len, input_size)
 * Output: (batch, seq_len, hidden_size)
 * Hidden: (num_layers, batch, hidden_size)
 */
void gru_forward(
    const float* input,     // (batch, seq_len, input_size)
    float* output,          // (batch, seq_len, hidden_size)
    float* hidden,          // (num_layers, batch, hidden_size) - 可为 NULL
    int batch,
    int seq_len,
    const GRUParams* params
);

/*
 * 创建 GRU 参数
 */
GRUParams* gru_create(
    int input_size,
    int hidden_size,
    int num_layers,
    int bidirectional
);

/*
 * 释放 GRU 参数
 */
void gru_free(GRUParams* params);


// ============================================================================
// 辅助函数
// ============================================================================

/*
 * Hz 到 ERB 转换
 */
float hz2erb(float freq_hz);

/*
 * ERB 到 Hz 转换
 */
float erb2hz(float erb_f);

/*
 * 生成 ERB 滤波器组
 */
void generate_erb_filters(
    float* erb_filters,     // 输出: (erb_subband_2, nfreqs-erb_subband_1)
    int erb_subband_1,
    int erb_subband_2,
    int nfft,
    int high_lim,
    int fs
);

/*
 * 计算能量（用于 TRA）
 */
void compute_energy(
    const Tensor* input,    // (B, C, T, F)
    float* energy,          // (B, C, T)
    int batch,
    int channels,
    int time_steps,
    int freq_bins
);

#endif
