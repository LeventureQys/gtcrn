/// <file>gtcrn_model.h</file>
/// <summary>GTCRN模型定义</summary>
/// <author>江月希 李文轩</author>
/// <remarks>16kHz音频的完整GTCRN语音增强模型</remarks>

#ifndef GTCRN_MODEL_H
#define GTCRN_MODEL_H

#include "gtcrn_types.h"
#include "gtcrn_layers.h"
#include "gtcrn_fft.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 前向声明 */
typedef struct gtcrn_weights_s gtcrn_weights_t;
typedef struct gtcrn_state_s gtcrn_state_t;
typedef struct gtcrn_t gtcrn_t;

// 权重结构定义
// 基于PyTorch模型分析: 总状态字典条目48847个浮点数,预期二进制文件大小195396字节(含8字节头)
// 关键观察: PReLU权重为[1](共享),非每通道; DPGRNN intra_rnn: GRU(input=8, hidden=4); DPGRNN inter_rnn: GRU(input=8, hidden=8)

/// <summary>完整GTCRN模型权重</summary>
struct gtcrn_weights_s {
    /* ERB滤波器组(可学习) */
    gtcrn_float erb_fc_weight[64 * 192];   /* (64, 192) */
    gtcrn_float ierb_fc_weight[192 * 64];  /* (192, 64) */

    /* 编码器ConvBlock 0: Conv2d(9, 16, (1,5), stride=(1,2), padding=(0,2)) */
    gtcrn_float en_conv0_weight[16 * 9 * 1 * 5];   /* 720 */
    gtcrn_float en_conv0_bias[16];
    gtcrn_float en_bn0_gamma[16];
    gtcrn_float en_bn0_beta[16];
    gtcrn_float en_bn0_mean[16];
    gtcrn_float en_bn0_var[16];
    gtcrn_float en_prelu0[1];  /* 单权重PReLU */

    /* 编码器ConvBlock 1: Conv2d(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2) */
    gtcrn_float en_conv1_weight[16 * 8 * 1 * 5];   /* 640, groups=2所以in_ch/groups=8 */
    gtcrn_float en_conv1_bias[16];
    gtcrn_float en_bn1_gamma[16];
    gtcrn_float en_bn1_beta[16];
    gtcrn_float en_bn1_mean[16];
    gtcrn_float en_bn1_var[16];
    gtcrn_float en_prelu1[1];

    /* 编码器GTConvBlock 2 (dilation=1): point_conv1, depth_conv, point_conv2, TRA */
    gtcrn_float en_gt2_pc1_weight[16 * 24 * 1 * 1];  /* (16, 24, 1, 1) = 384 */
    gtcrn_float en_gt2_pc1_bias[16];
    gtcrn_float en_gt2_bn1_gamma[16];
    gtcrn_float en_gt2_bn1_beta[16];
    gtcrn_float en_gt2_bn1_mean[16];
    gtcrn_float en_gt2_bn1_var[16];
    gtcrn_float en_gt2_prelu1[1];

    gtcrn_float en_gt2_dc_weight[16 * 1 * 3 * 3];  /* 深度可分离: (16, 1, 3, 3) = 144 */
    gtcrn_float en_gt2_dc_bias[16];
    gtcrn_float en_gt2_bn2_gamma[16];
    gtcrn_float en_gt2_bn2_beta[16];
    gtcrn_float en_gt2_bn2_mean[16];
    gtcrn_float en_gt2_bn2_var[16];
    gtcrn_float en_gt2_prelu2[1];

    gtcrn_float en_gt2_pc2_weight[8 * 16 * 1 * 1];  /* (8, 16, 1, 1) = 128 */
    gtcrn_float en_gt2_pc2_bias[8];
    gtcrn_float en_gt2_bn3_gamma[8];
    gtcrn_float en_gt2_bn3_beta[8];
    gtcrn_float en_gt2_bn3_mean[8];
    gtcrn_float en_gt2_bn3_var[8];

    /* GTConvBlock 2中的TRA: GRU(input=8, hidden=16) */
    gtcrn_float en_gt2_tra_gru_ih[48 * 8];   /* (3*16, 8) = 384 */
    gtcrn_float en_gt2_tra_gru_hh[48 * 16];  /* (3*16, 16) = 768 */
    gtcrn_float en_gt2_tra_gru_bih[48];
    gtcrn_float en_gt2_tra_gru_bhh[48];
    gtcrn_float en_gt2_tra_fc_weight[8 * 16];  /* (8, 16) = 128 */
    gtcrn_float en_gt2_tra_fc_bias[8];

    /* 编码器GTConvBlock 3 (dilation=2) - 相同结构 */
    gtcrn_float en_gt3_pc1_weight[16 * 24 * 1 * 1];
    gtcrn_float en_gt3_pc1_bias[16];
    gtcrn_float en_gt3_bn1_gamma[16];
    gtcrn_float en_gt3_bn1_beta[16];
    gtcrn_float en_gt3_bn1_mean[16];
    gtcrn_float en_gt3_bn1_var[16];
    gtcrn_float en_gt3_prelu1[1];
    gtcrn_float en_gt3_dc_weight[16 * 1 * 3 * 3];
    gtcrn_float en_gt3_dc_bias[16];
    gtcrn_float en_gt3_bn2_gamma[16];
    gtcrn_float en_gt3_bn2_beta[16];
    gtcrn_float en_gt3_bn2_mean[16];
    gtcrn_float en_gt3_bn2_var[16];
    gtcrn_float en_gt3_prelu2[1];
    gtcrn_float en_gt3_pc2_weight[8 * 16 * 1 * 1];
    gtcrn_float en_gt3_pc2_bias[8];
    gtcrn_float en_gt3_bn3_gamma[8];
    gtcrn_float en_gt3_bn3_beta[8];
    gtcrn_float en_gt3_bn3_mean[8];
    gtcrn_float en_gt3_bn3_var[8];
    gtcrn_float en_gt3_tra_gru_ih[48 * 8];
    gtcrn_float en_gt3_tra_gru_hh[48 * 16];
    gtcrn_float en_gt3_tra_gru_bih[48];
    gtcrn_float en_gt3_tra_gru_bhh[48];
    gtcrn_float en_gt3_tra_fc_weight[8 * 16];
    gtcrn_float en_gt3_tra_fc_bias[8];

    /* Encoder GTConvBlock 4 (dilation=5) - same structure */
    gtcrn_float en_gt4_pc1_weight[16 * 24 * 1 * 1];
    gtcrn_float en_gt4_pc1_bias[16];
    gtcrn_float en_gt4_bn1_gamma[16];
    gtcrn_float en_gt4_bn1_beta[16];
    gtcrn_float en_gt4_bn1_mean[16];
    gtcrn_float en_gt4_bn1_var[16];
    gtcrn_float en_gt4_prelu1[1];
    gtcrn_float en_gt4_dc_weight[16 * 1 * 3 * 3];
    gtcrn_float en_gt4_dc_bias[16];
    gtcrn_float en_gt4_bn2_gamma[16];
    gtcrn_float en_gt4_bn2_beta[16];
    gtcrn_float en_gt4_bn2_mean[16];
    gtcrn_float en_gt4_bn2_var[16];
    gtcrn_float en_gt4_prelu2[1];
    gtcrn_float en_gt4_pc2_weight[8 * 16 * 1 * 1];
    gtcrn_float en_gt4_pc2_bias[8];
    gtcrn_float en_gt4_bn3_gamma[8];
    gtcrn_float en_gt4_bn3_beta[8];
    gtcrn_float en_gt4_bn3_mean[8];
    gtcrn_float en_gt4_bn3_var[8];
    gtcrn_float en_gt4_tra_gru_ih[48 * 8];
    gtcrn_float en_gt4_tra_gru_hh[48 * 16];
    gtcrn_float en_gt4_tra_gru_bih[48];
    gtcrn_float en_gt4_tra_gru_bhh[48];
    gtcrn_float en_gt4_tra_fc_weight[8 * 16];
    gtcrn_float en_gt4_tra_fc_bias[8];

    /* DPGRNN 1 */
    /* 帧内RNN: 双向分组GRU,每个GRU(input=8, hidden=4) */
    /* 总计: 2组 x 2方向 x GRU(8,4) = rnn1 + rnn2 */
    gtcrn_float dp1_intra_rnn1_ih[12 * 8];    /* (3*4, 8) = 96 */
    gtcrn_float dp1_intra_rnn1_hh[12 * 4];    /* (3*4, 4) = 48 */
    gtcrn_float dp1_intra_rnn1_bih[12];
    gtcrn_float dp1_intra_rnn1_bhh[12];
    gtcrn_float dp1_intra_rnn1_ih_rev[12 * 8];
    gtcrn_float dp1_intra_rnn1_hh_rev[12 * 4];
    gtcrn_float dp1_intra_rnn1_bih_rev[12];
    gtcrn_float dp1_intra_rnn1_bhh_rev[12];
    gtcrn_float dp1_intra_rnn2_ih[12 * 8];
    gtcrn_float dp1_intra_rnn2_hh[12 * 4];
    gtcrn_float dp1_intra_rnn2_bih[12];
    gtcrn_float dp1_intra_rnn2_bhh[12];
    gtcrn_float dp1_intra_rnn2_ih_rev[12 * 8];
    gtcrn_float dp1_intra_rnn2_hh_rev[12 * 4];
    gtcrn_float dp1_intra_rnn2_bih_rev[12];
    gtcrn_float dp1_intra_rnn2_bhh_rev[12];
    gtcrn_float dp1_intra_fc_weight[16 * 16];  /* 256 */
    gtcrn_float dp1_intra_fc_bias[16];
    gtcrn_float dp1_intra_ln_gamma[GTCRN_DPGRNN_WIDTH * 16];  /* (33, 16) = 528 */
    gtcrn_float dp1_intra_ln_beta[GTCRN_DPGRNN_WIDTH * 16];

    /* 帧间RNN: 单向分组GRU,每个GRU(input=8, hidden=8) */
    gtcrn_float dp1_inter_rnn1_ih[24 * 8];   /* (3*8, 8) = 192 */
    gtcrn_float dp1_inter_rnn1_hh[24 * 8];   /* (3*8, 8) = 192 */
    gtcrn_float dp1_inter_rnn1_bih[24];
    gtcrn_float dp1_inter_rnn1_bhh[24];
    gtcrn_float dp1_inter_rnn2_ih[24 * 8];
    gtcrn_float dp1_inter_rnn2_hh[24 * 8];
    gtcrn_float dp1_inter_rnn2_bih[24];
    gtcrn_float dp1_inter_rnn2_bhh[24];
    gtcrn_float dp1_inter_fc_weight[16 * 16];
    gtcrn_float dp1_inter_fc_bias[16];
    gtcrn_float dp1_inter_ln_gamma[GTCRN_DPGRNN_WIDTH * 16];
    gtcrn_float dp1_inter_ln_beta[GTCRN_DPGRNN_WIDTH * 16];

    /* DPGRNN 2 - 与DPGRNN 1相同结构 */
    gtcrn_float dp2_intra_rnn1_ih[12 * 8];
    gtcrn_float dp2_intra_rnn1_hh[12 * 4];
    gtcrn_float dp2_intra_rnn1_bih[12];
    gtcrn_float dp2_intra_rnn1_bhh[12];
    gtcrn_float dp2_intra_rnn1_ih_rev[12 * 8];
    gtcrn_float dp2_intra_rnn1_hh_rev[12 * 4];
    gtcrn_float dp2_intra_rnn1_bih_rev[12];
    gtcrn_float dp2_intra_rnn1_bhh_rev[12];
    gtcrn_float dp2_intra_rnn2_ih[12 * 8];
    gtcrn_float dp2_intra_rnn2_hh[12 * 4];
    gtcrn_float dp2_intra_rnn2_bih[12];
    gtcrn_float dp2_intra_rnn2_bhh[12];
    gtcrn_float dp2_intra_rnn2_ih_rev[12 * 8];
    gtcrn_float dp2_intra_rnn2_hh_rev[12 * 4];
    gtcrn_float dp2_intra_rnn2_bih_rev[12];
    gtcrn_float dp2_intra_rnn2_bhh_rev[12];
    gtcrn_float dp2_intra_fc_weight[16 * 16];
    gtcrn_float dp2_intra_fc_bias[16];
    gtcrn_float dp2_intra_ln_gamma[GTCRN_DPGRNN_WIDTH * 16];
    gtcrn_float dp2_intra_ln_beta[GTCRN_DPGRNN_WIDTH * 16];
    gtcrn_float dp2_inter_rnn1_ih[24 * 8];
    gtcrn_float dp2_inter_rnn1_hh[24 * 8];
    gtcrn_float dp2_inter_rnn1_bih[24];
    gtcrn_float dp2_inter_rnn1_bhh[24];
    gtcrn_float dp2_inter_rnn2_ih[24 * 8];
    gtcrn_float dp2_inter_rnn2_hh[24 * 8];
    gtcrn_float dp2_inter_rnn2_bih[24];
    gtcrn_float dp2_inter_rnn2_bhh[24];
    gtcrn_float dp2_inter_fc_weight[16 * 16];
    gtcrn_float dp2_inter_fc_bias[16];
    gtcrn_float dp2_inter_ln_gamma[GTCRN_DPGRNN_WIDTH * 16];
    gtcrn_float dp2_inter_ln_beta[GTCRN_DPGRNN_WIDTH * 16];

    /* 解码器GTConvBlock 0 (dilation=5) */
    /* 注意: 解码器使用ConvTranspose2d,输入有跳跃连接 */
    /* point_conv1: (24, 16, 1, 1) - 与编码器相反 */
    gtcrn_float de_gt0_pc1_weight[24 * 16 * 1 * 1];  /* (24, 16, 1, 1) = 384 */
    gtcrn_float de_gt0_pc1_bias[16];
    gtcrn_float de_gt0_bn1_gamma[16];
    gtcrn_float de_gt0_bn1_beta[16];
    gtcrn_float de_gt0_bn1_mean[16];
    gtcrn_float de_gt0_bn1_var[16];
    gtcrn_float de_gt0_prelu1[1];
    gtcrn_float de_gt0_dc_weight[16 * 1 * 3 * 3];
    gtcrn_float de_gt0_dc_bias[16];
    gtcrn_float de_gt0_bn2_gamma[16];
    gtcrn_float de_gt0_bn2_beta[16];
    gtcrn_float de_gt0_bn2_mean[16];
    gtcrn_float de_gt0_bn2_var[16];
    gtcrn_float de_gt0_prelu2[1];
    /* point_conv2: (16, 8, 1, 1) - ConvTranspose2d */
    gtcrn_float de_gt0_pc2_weight[16 * 8 * 1 * 1];  /* (16, 8, 1, 1) = 128 */
    gtcrn_float de_gt0_pc2_bias[8];
    gtcrn_float de_gt0_bn3_gamma[8];
    gtcrn_float de_gt0_bn3_beta[8];
    gtcrn_float de_gt0_bn3_mean[8];
    gtcrn_float de_gt0_bn3_var[8];
    gtcrn_float de_gt0_tra_gru_ih[48 * 8];
    gtcrn_float de_gt0_tra_gru_hh[48 * 16];
    gtcrn_float de_gt0_tra_gru_bih[48];
    gtcrn_float de_gt0_tra_gru_bhh[48];
    gtcrn_float de_gt0_tra_fc_weight[8 * 16];
    gtcrn_float de_gt0_tra_fc_bias[8];

    /* 解码器GTConvBlock 1 (dilation=2) */
    gtcrn_float de_gt1_pc1_weight[24 * 16 * 1 * 1];
    gtcrn_float de_gt1_pc1_bias[16];
    gtcrn_float de_gt1_bn1_gamma[16];
    gtcrn_float de_gt1_bn1_beta[16];
    gtcrn_float de_gt1_bn1_mean[16];
    gtcrn_float de_gt1_bn1_var[16];
    gtcrn_float de_gt1_prelu1[1];
    gtcrn_float de_gt1_dc_weight[16 * 1 * 3 * 3];
    gtcrn_float de_gt1_dc_bias[16];
    gtcrn_float de_gt1_bn2_gamma[16];
    gtcrn_float de_gt1_bn2_beta[16];
    gtcrn_float de_gt1_bn2_mean[16];
    gtcrn_float de_gt1_bn2_var[16];
    gtcrn_float de_gt1_prelu2[1];
    gtcrn_float de_gt1_pc2_weight[16 * 8 * 1 * 1];
    gtcrn_float de_gt1_pc2_bias[8];
    gtcrn_float de_gt1_bn3_gamma[8];
    gtcrn_float de_gt1_bn3_beta[8];
    gtcrn_float de_gt1_bn3_mean[8];
    gtcrn_float de_gt1_bn3_var[8];
    gtcrn_float de_gt1_tra_gru_ih[48 * 8];
    gtcrn_float de_gt1_tra_gru_hh[48 * 16];
    gtcrn_float de_gt1_tra_gru_bih[48];
    gtcrn_float de_gt1_tra_gru_bhh[48];
    gtcrn_float de_gt1_tra_fc_weight[8 * 16];
    gtcrn_float de_gt1_tra_fc_bias[8];

    /* 解码器GTConvBlock 2 (dilation=1) */
    gtcrn_float de_gt2_pc1_weight[24 * 16 * 1 * 1];
    gtcrn_float de_gt2_pc1_bias[16];
    gtcrn_float de_gt2_bn1_gamma[16];
    gtcrn_float de_gt2_bn1_beta[16];
    gtcrn_float de_gt2_bn1_mean[16];
    gtcrn_float de_gt2_bn1_var[16];
    gtcrn_float de_gt2_prelu1[1];
    gtcrn_float de_gt2_dc_weight[16 * 1 * 3 * 3];
    gtcrn_float de_gt2_dc_bias[16];
    gtcrn_float de_gt2_bn2_gamma[16];
    gtcrn_float de_gt2_bn2_beta[16];
    gtcrn_float de_gt2_bn2_mean[16];
    gtcrn_float de_gt2_bn2_var[16];
    gtcrn_float de_gt2_prelu2[1];
    gtcrn_float de_gt2_pc2_weight[16 * 8 * 1 * 1];
    gtcrn_float de_gt2_pc2_bias[8];
    gtcrn_float de_gt2_bn3_gamma[8];
    gtcrn_float de_gt2_bn3_beta[8];
    gtcrn_float de_gt2_bn3_mean[8];
    gtcrn_float de_gt2_bn3_var[8];
    gtcrn_float de_gt2_tra_gru_ih[48 * 8];
    gtcrn_float de_gt2_tra_gru_hh[48 * 16];
    gtcrn_float de_gt2_tra_gru_bih[48];
    gtcrn_float de_gt2_tra_gru_bhh[48];
    gtcrn_float de_gt2_tra_fc_weight[8 * 16];
    gtcrn_float de_gt2_tra_fc_bias[8];

    /* 解码器ConvBlock 3: ConvTranspose2d(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2) */
    /* 权重形状: (16, 8, 1, 5) - in_ch, out_ch/groups, kH, kW */
    gtcrn_float de_conv3_weight[16 * 8 * 1 * 5];  /* 640 */
    gtcrn_float de_conv3_bias[16];
    gtcrn_float de_bn3_gamma[16];
    gtcrn_float de_bn3_beta[16];
    gtcrn_float de_bn3_mean[16];
    gtcrn_float de_bn3_var[16];
    gtcrn_float de_prelu3[1];

    /* 解码器ConvBlock 4: ConvTranspose2d(16, 2, (1,5), stride=(1,2), padding=(0,2)) */
    /* 权重形状: (16, 2, 1, 5) - in_ch, out_ch, kH, kW(无Tanh权重) */
    gtcrn_float de_conv4_weight[16 * 2 * 1 * 5];  /* 160 */
    gtcrn_float de_conv4_bias[2];
    gtcrn_float de_bn4_gamma[2];
    gtcrn_float de_bn4_beta[2];
    gtcrn_float de_bn4_mean[2];
    gtcrn_float de_bn4_var[2];
    /* 无PReLU,使用Tanh激活(无可学习参数) */
};

// 流式状态结构

/// <summary>GTCRN流式推理状态</summary>
/// <remarks>缓存布局: 每个GTConvBlock使用独立的卷积缓存,避免复杂的偏移计算</remarks>
struct gtcrn_state_s {
    /* 编码器卷积缓存: 每个GTConvBlock独立存储 */
    /* 布局: (channels, cache_t, freq) 用于每个块 */
    /* cache_t = (kernel_t - 1) * dilation_t + 1 (包含当前帧更新后) */
    gtcrn_float en_gt2_cache[GTCRN_CHANNELS * 3 * GTCRN_DPGRNN_WIDTH];   /* dilation=1, cache_t=3 */
    gtcrn_float en_gt3_cache[GTCRN_CHANNELS * 5 * GTCRN_DPGRNN_WIDTH];   /* dilation=2, cache_t=5 */
    gtcrn_float en_gt4_cache[GTCRN_CHANNELS * 11 * GTCRN_DPGRNN_WIDTH];  /* dilation=5, cache_t=11 */

    /* 编码器TRA GRU隐藏状态: 3个GTConvBlock各(16,) */
    /* TRA GRU的hidden=channels*2=16,当half_channels=8时 */
    gtcrn_float en_tra_h2[GTCRN_CHANNELS];  /* GTConvBlock 2, dilation=1 */
    gtcrn_float en_tra_h3[GTCRN_CHANNELS];  /* GTConvBlock 3, dilation=2 */
    gtcrn_float en_tra_h4[GTCRN_CHANNELS];  /* GTConvBlock 4, dilation=5 */

    /* 解码器卷积缓存: 每个GTConvBlock独立存储 */
    /* cache_t = (kernel_t - 1) * dilation_t (不含当前帧) */
    gtcrn_float de_gt0_cache[GTCRN_CHANNELS * 10 * GTCRN_DPGRNN_WIDTH];  /* dilation=5, cache_t=10 */
    gtcrn_float de_gt1_cache[GTCRN_CHANNELS * 4 * GTCRN_DPGRNN_WIDTH];   /* dilation=2, cache_t=4 */
    gtcrn_float de_gt2_cache[GTCRN_CHANNELS * 2 * GTCRN_DPGRNN_WIDTH];   /* dilation=1, cache_t=2 */

    /* 解码器TRA GRU隐藏状态 */
    gtcrn_float de_tra_h0[GTCRN_CHANNELS];  /* GTConvBlock 0, dilation=5 */
    gtcrn_float de_tra_h1[GTCRN_CHANNELS];  /* GTConvBlock 1, dilation=2 */
    gtcrn_float de_tra_h2[GTCRN_CHANNELS];  /* GTConvBlock 2, dilation=1 */

    /* DPGRNN帧间GRU隐藏状态: 每个DPGRNN为(33, 16) */
    /* 帧间RNN使用分组GRU,每组hidden=8,总计16 */
    gtcrn_float dp1_inter_h[GTCRN_DPGRNN_WIDTH * GTCRN_HIDDEN_SIZE];
    gtcrn_float dp2_inter_h[GTCRN_DPGRNN_WIDTH * GTCRN_HIDDEN_SIZE];

    /* 编码器跳跃连接缓冲区用于解码器(单帧) */
    gtcrn_float en_out0[GTCRN_CHANNELS * 65];       /* EnConv0后(16, 65) */
    gtcrn_float en_out1[GTCRN_CHANNELS * GTCRN_DPGRNN_WIDTH];  /* EnConv1后(16, 33) */
    gtcrn_float en_out2[GTCRN_CHANNELS * GTCRN_DPGRNN_WIDTH];  /* EnGT2后(16, 33) */
    gtcrn_float en_out3[GTCRN_CHANNELS * GTCRN_DPGRNN_WIDTH];  /* EnGT3后(16, 33) */
    gtcrn_float en_out4[GTCRN_CHANNELS * GTCRN_DPGRNN_WIDTH];  /* EnGT4后(16, 33) */

    /* 流式STFT输入缓冲区: 存储前一帧的256个采样点 */
    gtcrn_float stft_input_buffer[GTCRN_HOP_SIZE];

    /* 流式ISTFT重叠相加缓冲区: 存储前一帧ISTFT输出的后半部分 */
    gtcrn_float ola_buffer[GTCRN_WIN_SIZE];
    int first_frame;  /* 标记是否是第一帧 */
};

// 主GTCRN句柄

/// <summary>主GTCRN模型句柄</summary>
struct gtcrn_t {
    gtcrn_weights_t* weights;
    gtcrn_state_t* state;
    gtcrn_stft_t* stft;

    /* 中间计算工作空间 */
    gtcrn_float* workspace;
    size_t workspace_size;

    int is_initialized;
};

/// <summary>创建GTCRN模型实例</summary>
/// <remarks>
/// 分配并初始化GTCRN模型所需的所有内存,包括权重结构、流式状态、STFT处理器和工作空间。
/// 创建后需要调用gtcrn_load_weights()加载模型权重才能使用。
/// 使用完毕后必须调用gtcrn_destroy()释放资源。
/// </remarks>
/// <returns>成功返回新的GTCRN句柄指针,失败返回NULL(内存分配失败)</returns>
/// <example>
///   gtcrn_t* model = gtcrn_create();
///   if (!model) {
///       fprintf(stderr, "创建模型失败\n");
///       return -1;
///   }
/// </example>
gtcrn_t* gtcrn_create(void);

/// <summary>销毁GTCRN模型实例</summary>
/// <remarks>释放模型占用的所有内存,包括权重、状态、STFT处理器和工作空间。调用后model指针将无效,不应再次使用。重复调用是安全的(传入NULL)。</remarks>
/// <param name="model">GTCRN模型句柄指针,可为NULL(安全处理)</param>
void gtcrn_destroy(gtcrn_t* model);

/// <summary>从二进制文件加载模型权重</summary>
/// <remarks>
/// 从指定的二进制文件读取GTCRN模型权重并填充到模型结构中。
/// 权重文件格式: 8字节头(版本/大小信息) + 48847个float32权重数据。
/// 必须在gtcrn_create()之后调用,权重文件必须与当前代码版本兼容。
/// 加载失败后模型状态未定义,建议销毁重建。
/// </remarks>
/// <param name="model">GTCRN模型句柄,必须已通过gtcrn_create()创建</param>
/// <param name="filepath">权重文件的完整路径,必须是有效的二进制权重文件</param>
/// <returns>成功返回GTCRN_OK(0),失败返回错误码: GTCRN_ERROR_NULL_POINTER/model或filepath为NULL, GTCRN_ERROR_FILE_IO/文件打开/读取失败, GTCRN_ERROR_INVALID_FORMAT/文件格式不正确, GTCRN_ERROR_MEMORY_ALLOC/内存分配失败</returns>
/// <example>
///   gtcrn_status_t status = gtcrn_load_weights(model, "weights/gtcrn_weights.bin");
///   if (status != GTCRN_OK) {
///       fprintf(stderr, "加载权重失败: %d\n", status);
///       gtcrn_destroy(model);
///       return -1;
///   }
/// </example>
gtcrn_status_t gtcrn_load_weights(gtcrn_t* model, const char* filepath);

/// <summary>重置流式状态(在处理新音频流前调用)</summary>
/// <remarks>
/// 清空模型内部的所有流式缓存和隐藏状态,包括编码器/解码器卷积缓存、TRA GRU隐藏状态、DPGRNN帧间GRU隐藏状态、STFT重叠相加缓冲区。
/// 在处理新的音频流之前必须调用此函数,否则会使用上一个流的残留状态。
/// 离线模式(gtcrn_process)内部会自动重置,无需手动调用。流式模式(gtcrn_process_frame)必须在开始新流前调用。可以安全地多次调用。
/// </remarks>
/// <param name="model">GTCRN模型句柄,必须已加载权重</param>
/// <example>
///   // 开始新的音频流
///   gtcrn_reset_state(model);
///   for (int i = 0; i < num_frames; i++) {
///       gtcrn_process_frame(model, input_frame, output_frame);
///   }
/// </example>
void gtcrn_reset_state(gtcrn_t* model);

/// <summary>处理完整音频文件(离线模式)</summary>
/// <remarks>
/// 对完整的音频信号进行语音增强处理。内部会: 1.自动重置流式状态 2.对整个信号进行STFT 3.执行完整的前向传播 4.进行逆STFT重建时域信号。
/// 输入音频必须是16kHz单声道,其他采样率需要先重采样。输出长度可能与输入长度不同(由于STFT帧对齐)。
/// 函数内部会自动重置状态,无需手动调用gtcrn_reset_state()。适合离线批处理,不适合实时应用。
/// </remarks>
/// <param name="model">GTCRN模型句柄,必须已加载权重</param>
/// <param name="input">输入音频采样数组,要求16kHz采样率、单声道、浮点格式(-1.0到1.0)</param>
/// <param name="input_len">输入采样点数,必须大于0</param>
/// <param name="output">输出缓冲区,必须预分配至少input_len个gtcrn_float的空间</param>
/// <param name="output_len">输出参数,函数返回后存储实际输出的采样点数,由于STFT的帧对齐,可能与input_len略有差异</param>
/// <returns>成功返回GTCRN_OK(0),失败返回错误码: GTCRN_ERROR_NULL_POINTER/任何指针参数为NULL, GTCRN_ERROR_INVALID_PARAM/input_len <= 0, GTCRN_ERROR_NOT_INITIALIZED/模型未加载权重</returns>
/// <example>
///   float* input = ...;  // 16kHz单声道音频
///   int input_len = 16000;  // 1秒音频
///   float* output = (float*)malloc(input_len * sizeof(float));
///   int output_len = 0;
///   gtcrn_status_t status = gtcrn_process(model, input, input_len, output, &output_len);
/// </example>
gtcrn_status_t gtcrn_process(gtcrn_t* model,
                             const gtcrn_float* input, int input_len,
                             gtcrn_float* output, int* output_len);

/// <summary>处理单帧(流式模式)</summary>
/// <remarks>
/// 对单个音频帧进行实时语音增强处理。每帧固定256个采样点(16kHz时16ms)。
/// 函数会维护内部状态以实现流式处理,支持连续帧之间的上下文依赖。
/// 输入帧必须是256个采样点,不支持其他帧大小。处理新音频流前必须调用gtcrn_reset_state()。
/// 帧与帧之间会保持状态,适合实时流式处理。延迟约为1帧(16ms),适合低延迟实时应用。输出帧长度固定为256个采样点。
/// </remarks>
/// <param name="model">GTCRN模型句柄,必须已加载权重</param>
/// <param name="input_frame">输入帧数据,必须包含恰好256个采样点(16kHz时16ms),采样值范围应在[-1.0, 1.0]</param>
/// <param name="output_frame">输出帧缓冲区,必须预分配256个gtcrn_float的空间,输出也是256个采样点</param>
/// <returns>成功返回GTCRN_OK(0),失败返回错误码: GTCRN_ERROR_NULL_POINTER/任何指针参数为NULL, GTCRN_ERROR_NOT_INITIALIZED/模型未加载权重</returns>
/// <example>
///   // 开始新流
///   gtcrn_reset_state(model);
///   // 逐帧处理
///   for (int i = 0; i < num_frames; i++) {
///       const float* input_frame = audio_in + i * 256;
///       float* output_frame = audio_out + i * 256;
///       gtcrn_status_t status = gtcrn_process_frame(model, input_frame, output_frame);
///   }
/// </example>
gtcrn_status_t gtcrn_process_frame(gtcrn_t* model,
                                   const gtcrn_float* input_frame,
                                   gtcrn_float* output_frame);

/// <summary>获取所需工作空间大小</summary>
/// <remarks>
/// 返回GTCRN模型进行前向推理所需的工作空间内存大小(字节)。
/// 工作空间用于存储中间计算结果,由模型内部自动管理。
/// 返回值在模型创建时确定,与输入长度无关。可用于评估内存需求或调试。实际内存由gtcrn_create()内部分配,用户无需手动管理。
/// </remarks>
/// <returns>工作空间大小(字节),通常为几百KB到几MB</returns>
/// <example>
///   size_t workspace_size = gtcrn_get_workspace_size();
///   printf("工作空间大小: %.2f KB\n", workspace_size / 1024.0);
/// </example>
size_t gtcrn_get_workspace_size(void);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_MODEL_H */
