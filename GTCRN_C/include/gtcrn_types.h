/// <file>gtcrn_types.h</file>
/// <summary>GTCRN通用类型定义</summary>
/// <author>江月希 李文轩</author>
/// <remarks>包含GTCRN语音增强库中使用的所有通用类型定义和常量</remarks>

#ifndef GTCRN_TYPES_H
#define GTCRN_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 浮点精度选择 */
#ifndef GTCRN_USE_DOUBLE
typedef float gtcrn_float;
#else
typedef double gtcrn_float;
#endif

/* 状态码 */
typedef enum {
    GTCRN_OK = 0,
    GTCRN_ERROR_NULL_POINTER = -1,
    GTCRN_ERROR_INVALID_PARAM = -2,
    GTCRN_ERROR_MEMORY_ALLOC = -3,
    GTCRN_ERROR_FILE_IO = -4,
    GTCRN_ERROR_INVALID_FORMAT = -5,
    GTCRN_ERROR_NOT_INITIALIZED = -6
} gtcrn_status_t;

/* 16kHz模型配置常量 */
#define GTCRN_SAMPLE_RATE       16000
#define GTCRN_FFT_SIZE          512
#define GTCRN_HOP_SIZE          256
#define GTCRN_WIN_SIZE          512
#define GTCRN_FREQ_BINS         257     /* FFT_SIZE / 2 + 1 */

/* ERB配置 */
#define GTCRN_ERB_SUBBAND_1     65      /* 保持原样的低频bin */
#define GTCRN_ERB_SUBBAND_2     64      /* ERB压缩频带 */
#define GTCRN_ERB_TOTAL         129     /* ERB_SUBBAND_1 + ERB_SUBBAND_2 */

/* 网络架构常量 */
#define GTCRN_CHANNELS          16      /* 网络中特征通道数 */
#define GTCRN_SFE_KERNEL        3       /* SFE核大小 */
#define GTCRN_DPGRNN_WIDTH      33      /* 下采样后DPGRNN频率宽度 */
#define GTCRN_HIDDEN_SIZE       16      /* GRU隐藏层大小 */

/* 编码器膨胀因子 */
#define GTCRN_EN_DILATION_1     1
#define GTCRN_EN_DILATION_2     2
#define GTCRN_EN_DILATION_3     5

/* 流式缓存大小 */
#define GTCRN_CONV_CACHE_T      16      /* 总时间缓存: 2 + 4 + 10 = 16 */
#define GTCRN_TRA_CACHE_LAYERS  3       /* 编码器/解码器中GTConvBlock层数 */

/* 复数结构体 */
typedef struct {
    gtcrn_float real;
    gtcrn_float imag;
} gtcrn_complex_t;

/* 张量形状描述符(最多4维) */
typedef struct {
    int dims[4];    /* 维度: [batch, channels, time, freq] 或类似 */
    int ndim;       /* 维度数 */
} gtcrn_shape_t;

/* 用于权重存储的简单张量结构 */
typedef struct {
    gtcrn_float* data;
    gtcrn_shape_t shape;
    size_t size;    /* 元素总数 */
} gtcrn_tensor_t;

/* 工具宏 */
#define GTCRN_MAX(a, b) ((a) > (b) ? (a) : (b))
#define GTCRN_MIN(a, b) ((a) < (b) ? (a) : (b))
#define GTCRN_ABS(x) ((x) < 0 ? -(x) : (x))

/* SIMD内存对齐(如将来需要) */
#define GTCRN_ALIGN 16

/* 常用布局的张量索引宏 */
/* 4维张量: (B, C, H, W)布局 */
#define GTCRN_IDX4(b, c, h, w, C, H, W) \
    (((b) * (C) + (c)) * (H) * (W) + (h) * (W) + (w))

/* 3维张量: (B, H, W)布局 */
#define GTCRN_IDX3(b, h, w, H, W) \
    ((b) * (H) * (W) + (h) * (W) + (w))

/* 2维张量: (H, W)布局 */
#define GTCRN_IDX2(h, w, W) \
    ((h) * (W) + (w))

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_TYPES_H */
