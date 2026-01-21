/**
 * stream_conv.h - Streaming Convolution Operations
 *
 * 实现支持状态缓存的流式卷积操作，用于实时音频处理
 */

#ifndef STREAM_CONV_H
#define STREAM_CONV_H

#include "nn_layers.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * StreamConv2d - 流式2D卷积
 * ============================================================================ */

/**
 * 流式Conv2d前向传播
 *
 * 关键特性:
 * 1. 时间维度因果性: T_pad必须为0，使用缓存实现因果卷积
 * 2. 缓存管理: 保存历史帧用于下一次卷积
 * 3. 单帧处理: 通常 T=1 用于实时处理
 *
 * Python实现参考 (stream/modules/convolution.py:85-93):
 *   inp = torch.cat([cache, x], dim=2)      # 拼接缓存和当前帧
 *   outp = self.Conv2d(inp)                 # 卷积
 *   out_cache = inp[:,:, 1:]                # 更新缓存
 *
 * @param input       输入tensor (B, C_in, T, F) - 通常 T=1
 * @param output      输出tensor (B, C_out, T_out, F_out)
 * @param conv_cache  卷积缓存 (B, C_in, kT-1, F) - 保存历史帧
 *                    其中 kT 是时间维度的kernel size
 * @param conv_params 卷积参数（包含权重、偏置等）
 *
 * 注意:
 * - conv_cache 会被原地更新
 * - 要求 conv_params->padding_h == 0 (时间维度无padding)
 * - 时间维度的kernel size决定缓存大小: cache_size = (kT - 1) * dilation_h
 */
void stream_conv2d_forward(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,
    const Conv2dParams* conv_params
);

/* ============================================================================
 * StreamConvTranspose2d - 流式2D转置卷积
 * ============================================================================ */

/**
 * 流式ConvTranspose2d前向传播
 *
 * 使用Conv2d实现ConvTranspose2d（权重时间反转）
 *
 * Python实现参考 (stream/modules/convolution.py:232-262):
 *   inp = torch.cat([cache, x], dim=2)      # 拼接缓存和当前帧
 *   out_cache = inp[:, :, 1:]               # 更新缓存
 *   # 频率维度上采样（如果F_stride > 1）
 *   # 频率维度padding
 *   outp = self.ConvTranspose2d(inp)        # 使用Conv2d实现
 *
 * @param input       输入tensor (B, C_in, T, F) - 通常 T=1
 * @param output      输出tensor (B, C_out, T_out, F_out)
 * @param conv_cache  卷积缓存 (B, C_in, kT-1, F)
 * @param conv_params 卷积参数
 *
 * 注意:
 * - 时间维度stride必须为1 (T_stride == 1)
 * - 时间维度padding必须为0 (T_pad == 0)
 * - 频率维度可以有stride和padding
 */
void stream_conv_transpose2d_forward(
    const Tensor* input,
    Tensor* output,
    float* conv_cache,
    const Conv2dParams* conv_params
);

/* ============================================================================
 * 辅助函数
 * ============================================================================ */

/**
 * 计算流式卷积所需的缓存大小
 *
 * @param kernel_h    时间维度kernel size
 * @param dilation_h  时间维度dilation
 * @return 缓存大小 = (kernel_h - 1) * dilation_h
 */
static inline int stream_conv_cache_size(int kernel_h, int dilation_h) {
    return (kernel_h - 1) * dilation_h;
}

/**
 * 初始化流式卷积缓存（全零）
 *
 * @param cache       缓存指针
 * @param batch       batch size
 * @param channels    通道数
 * @param cache_size  缓存大小（时间维度）
 * @param freq_bins   频率bins数量
 */
void stream_conv_cache_init(
    float* cache,
    int batch,
    int channels,
    int cache_size,
    int freq_bins
);

/**
 * 更新流式卷积缓存
 *
 * 将 input 的最后 (cache_size-1) 帧移到缓存开头，
 * 然后将新帧添加到缓存末尾
 *
 * @param cache       缓存指针 (B, C, cache_size, F)
 * @param input       输入数据 (B, C, T, F)
 * @param batch       batch size
 * @param channels    通道数
 * @param cache_size  缓存大小
 * @param input_time  输入时间帧数
 * @param freq_bins   频率bins数量
 */
void stream_conv_cache_update(
    float* cache,
    const float* input,
    int batch,
    int channels,
    int cache_size,
    int input_time,
    int freq_bins
);

#ifdef __cplusplus
}
#endif

#endif // STREAM_CONV_H
