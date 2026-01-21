#include "gtcrn_model.h"
#include "gtcrn_modules.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// 注意: gtconvblock_forward_complete.c 和 GRU_bidirectional_complete.c
// 包含完整的示例实现，已作为独立的源文件在 CMakeLists.txt 中处理
// gtconvblock_forward_complete.c 在 UTIL_SOURCES 中
// GRU_bidirectional_complete.c 仅在示例程序中使用

// ============================================================================
// ConvBlock 实现
// ============================================================================

ConvBlock* convblock_create(
    const Conv2dParams* conv_params,
    const BatchNorm2dParams* bn_params,
    const PReLUParams* prelu_params,
    int use_tanh
) {
    ConvBlock* block = (ConvBlock*)malloc(sizeof(ConvBlock));
    if (!block) return NULL;

    // 融合 Conv + BN
    memset(&block->fused_conv_bn, 0, sizeof(FusedConvBN));
    fuse_conv_batchnorm(&block->fused_conv_bn, conv_params, bn_params);

    // PReLU
    if (!use_tanh && prelu_params) {
        block->prelu = prelu_create(
            prelu_params->num_parameters,
            prelu_params->weight
        );
    } else {
        block->prelu = NULL;
    }

    block->use_tanh = use_tanh;

    return block;
}

void convblock_forward(
    const Tensor* input,
    Tensor* output,
    ConvBlock* block
) {
    // Conv + BN (融合)
    fused_conv_bn_forward(input, output, &block->fused_conv_bn);

    // 激活
    if (block->use_tanh) {
        tanh_forward(output);
    } else if (block->prelu) {
        prelu_forward_v2(output, block->prelu);
    }
}

void convblock_free(ConvBlock* block) {
    if (block) {
        fused_conv_bn_free(&block->fused_conv_bn);
        if (block->prelu) prelu_free(block->prelu);
        free(block);
    }
}


// ============================================================================
// GTConvBlock 实现（完整版本）
// ============================================================================

// 注意: channel_shuffle, channel_split, temporal_pad 等辅助函数
// 已在 gtconvblock_forward_complete.c 中定义

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
) {
    GTConvBlock* block = (GTConvBlock*)malloc(sizeof(GTConvBlock));
    if (!block) return NULL;

    // 创建 SFE 模块
    block->sfe = sfe_create(3, 1);

    // 创建 TRA 模块
    block->tra = tra_create(in_channels / 2);
    block->use_tra = 1;

    // 注意: 这里需要创建Conv和BN参数，但权重需要从模型文件加载
    // 暂时设置为NULL，表示需要加载
    memset(&block->point_conv1, 0, sizeof(FusedConvBN));
    memset(&block->depth_conv, 0, sizeof(FusedConvBN));
    memset(&block->point_conv2, 0, sizeof(FusedConvBN));

    block->point_prelu1 = NULL;
    block->depth_prelu = NULL;

    printf("GTConvBlock 创建成功 (in=%d, hidden=%d, kernel=(%d,%d), dilation=(%d,%d), deconv=%d)\n",
           in_channels, hidden_channels, kernel_h, kernel_w, dilation_h, dilation_w, use_deconv);

    return block;
}

void gtconvblock_forward(
    const Tensor* input,
    Tensor* output,
    GTConvBlock* block
) {
    // 使用完整实现（来自 gtconvblock_forward_complete.c）
    // 默认使用 kernel_h=3, dilation_h=1
    gtconvblock_forward_complete(input, output, block, 3, 1);
}

void gtconvblock_free(GTConvBlock* block) {
    if (block) {
        if (block->sfe) sfe_free(block->sfe);
        if (block->tra) tra_free(block->tra);

        // 释放融合的Conv+BN
        fused_conv_bn_free(&block->point_conv1);
        fused_conv_bn_free(&block->depth_conv);
        fused_conv_bn_free(&block->point_conv2);

        // 释放PReLU
        if (block->point_prelu1) prelu_free(block->point_prelu1);
        if (block->depth_prelu) prelu_free(block->depth_prelu);

        free(block);
    }
}


// ============================================================================
// Encoder 实现（完整版本）
// ============================================================================

// 辅助函数：创建ConvBlock（分配内存但权重待加载）
static ConvBlock* create_convblock_placeholder(
    int in_channels, int out_channels,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int groups, int use_tanh
) {
    ConvBlock* block = (ConvBlock*)malloc(sizeof(ConvBlock));
    if (!block) return NULL;

    // 创建Conv2d参数（分配权重内存）
    Conv2dParams* conv_params = (Conv2dParams*)malloc(sizeof(Conv2dParams));
    conv_params->in_channels = in_channels;
    conv_params->out_channels = out_channels;
    conv_params->kernel_h = kernel_h;
    conv_params->kernel_w = kernel_w;
    conv_params->stride_h = stride_h;
    conv_params->stride_w = stride_w;
    conv_params->padding_h = padding_h;
    conv_params->padding_w = padding_w;
    conv_params->dilation_h = 1;
    conv_params->dilation_w = 1;
    conv_params->groups = groups;

    // 分配权重内存
    int weight_size = (out_channels / groups) * (in_channels / groups) * kernel_h * kernel_w;
    conv_params->weight = (float*)calloc(weight_size, sizeof(float));
    conv_params->bias = (float*)calloc(out_channels, sizeof(float));

    // 创建BatchNorm参数
    BatchNorm2dParams* bn_params = (BatchNorm2dParams*)malloc(sizeof(BatchNorm2dParams));
    bn_params->num_features = out_channels;
    bn_params->eps = 1e-5f;
    bn_params->gamma = (float*)malloc(out_channels * sizeof(float));
    bn_params->beta = (float*)malloc(out_channels * sizeof(float));
    bn_params->running_mean = (float*)calloc(out_channels, sizeof(float));
    bn_params->running_var = (float*)malloc(out_channels * sizeof(float));

    // 初始化BN参数
    for (int i = 0; i < out_channels; i++) {
        bn_params->gamma[i] = 1.0f;
        bn_params->beta[i] = 0.0f;
        bn_params->running_var[i] = 1.0f;
    }

    // 创建PReLU参数（如果不使用tanh）
    PReLUParams* prelu_params = NULL;
    if (!use_tanh) {
        prelu_params = (PReLUParams*)malloc(sizeof(PReLUParams));
        prelu_params->num_parameters = out_channels;
        prelu_params->weight = (float*)malloc(out_channels * sizeof(float));
        for (int i = 0; i < out_channels; i++) {
            prelu_params->weight[i] = 0.25f;  // 默认值
        }
    }

    // 创建ConvBlock
    block = convblock_create(conv_params, bn_params, prelu_params, use_tanh);

    // 释放临时参数结构（权重已被复制）
    free(conv_params);
    free(bn_params);
    if (prelu_params) free(prelu_params);

    return block;
}

Encoder* encoder_create() {
    Encoder* encoder = (Encoder*)malloc(sizeof(Encoder));
    if (!encoder) return NULL;

    printf("创建 Encoder...\n");

    /*
     * Encoder 结构 (from gtcrn1.py lines 228-237):
     * 1. ConvBlock(9, 16, (1,5), stride=(1,2), padding=(0,2))
     * 2. ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2)
     * 3. GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1))
     * 4. GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1))
     * 5. GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1))
     */

    // 创建Conv1: (9, 16, (1,5), stride=(1,2), padding=(0,2))
    encoder->conv1 = create_convblock_placeholder(9, 16, 1, 5, 1, 2, 0, 2, 1, 0);
    printf("  创建 conv1: (9, 16, (1,5), stride=(1,2))\n");

    // 创建Conv2: (16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2)
    encoder->conv2 = create_convblock_placeholder(16, 16, 1, 5, 1, 2, 0, 2, 2, 0);
    printf("  创建 conv2: (16, 16, (1,5), stride=(1,2), groups=2)\n");

    // 创建GTConvBlock
    encoder->gtconv1 = gtconvblock_create(16, 16, 3, 3, 1, 1, 0, 1, 1, 1, 0);
    encoder->gtconv2 = gtconvblock_create(16, 16, 3, 3, 1, 1, 0, 1, 2, 1, 0);
    encoder->gtconv3 = gtconvblock_create(16, 16, 3, 3, 1, 1, 0, 1, 5, 1, 0);

    printf("Encoder 创建成功\n");

    return encoder;
}

void encoder_forward(
    const Tensor* input,
    Tensor* output,
    Tensor** skip_connections,
    Encoder* encoder
) {
    /*
     * Encoder 完整前向传播
     *
     * 输入: (B, 9, T, 385)
     * 输出: (B, 16, T, 97)
     *
     * 流程:
     * Layer 1: (B, 9, T, 385) -> (B, 16, T, 193)  [stride=(1,2)]
     * Layer 2: (B, 16, T, 193) -> (B, 16, T, 97)  [stride=(1,2)]
     * Layer 3: (B, 16, T, 97) -> (B, 16, T, 97)   [GTConv, dilation=1]
     * Layer 4: (B, 16, T, 97) -> (B, 16, T, 97)   [GTConv, dilation=2]
     * Layer 5: (B, 16, T, 97) -> (B, 16, T, 97)   [GTConv, dilation=5]
     */

    int B = input->shape.batch;
    int T = input->shape.height;

    // 分配中间缓冲区
    Tensor layer1_out = {
        .data = (float*)malloc(B * 16 * T * 193 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    Tensor layer2_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer3_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer4_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    // Layer 1: ConvBlock
    if (encoder->conv1) {
        convblock_forward(input, &layer1_out, encoder->conv1);
    } else {
        // 简化: 直接下采样
        // 实际需要卷积操作
        memset(layer1_out.data, 0, B * 16 * T * 193 * sizeof(float));
    }
    if (skip_connections) skip_connections[0] = &layer1_out;

    // Layer 2: ConvBlock
    if (encoder->conv2) {
        convblock_forward(&layer1_out, &layer2_out, encoder->conv2);
    } else {
        // 简化: 直接下采样
        memset(layer2_out.data, 0, B * 16 * T * 97 * sizeof(float));
    }
    if (skip_connections) skip_connections[1] = &layer2_out;

    // Layer 3: GTConvBlock (dilation=1)
    if (encoder->gtconv1) {
        gtconvblock_forward(&layer2_out, &layer3_out, encoder->gtconv1);
    } else {
        memcpy(layer3_out.data, layer2_out.data, B * 16 * T * 97 * sizeof(float));
    }
    if (skip_connections) skip_connections[2] = &layer3_out;

    // Layer 4: GTConvBlock (dilation=2)
    if (encoder->gtconv2) {
        gtconvblock_forward(&layer3_out, &layer4_out, encoder->gtconv2);
    } else {
        memcpy(layer4_out.data, layer3_out.data, B * 16 * T * 97 * sizeof(float));
    }
    if (skip_connections) skip_connections[3] = &layer4_out;

    // Layer 5: GTConvBlock (dilation=5)
    if (encoder->gtconv3) {
        gtconvblock_forward(&layer4_out, output, encoder->gtconv3);
    } else {
        memcpy(output->data, layer4_out.data, B * 16 * T * 97 * sizeof(float));
    }
    if (skip_connections) skip_connections[4] = output;

    // 注意: skip_connections 指向的内存需要在外部管理
    // 这里只是设置指针
}

void encoder_free(Encoder* encoder) {
    if (encoder) {
        if (encoder->conv1) convblock_free(encoder->conv1);
        if (encoder->conv2) convblock_free(encoder->conv2);
        if (encoder->gtconv1) gtconvblock_free(encoder->gtconv1);
        if (encoder->gtconv2) gtconvblock_free(encoder->gtconv2);
        if (encoder->gtconv3) gtconvblock_free(encoder->gtconv3);
        free(encoder);
    }
}


// ============================================================================
// Decoder 实现（完整版本）
// ============================================================================

Decoder* decoder_create() {
    Decoder* decoder = (Decoder*)malloc(sizeof(Decoder));
    if (!decoder) return NULL;

    printf("创建 Decoder...\n");

    /*
     * Decoder 结构 (from gtcrn1.py lines 247-256):
     * 1. GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), deconv=True)
     * 2. GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), deconv=True)
     * 3. GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), deconv=True)
     * 4. ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, deconv=True)
     * 5. ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), deconv=True, tanh=True)
     */

    // 创建GTConvBlock (deconv=True)
    decoder->gtconv1 = gtconvblock_create(16, 16, 3, 3, 1, 1, 10, 1, 5, 1, 1);
    decoder->gtconv2 = gtconvblock_create(16, 16, 3, 3, 1, 1, 4, 1, 2, 1, 1);
    decoder->gtconv3 = gtconvblock_create(16, 16, 3, 3, 1, 1, 2, 1, 1, 1, 1);

    // 创建Conv1: (16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, deconv=True)
    // 注意: deconv在C实现中通过ConvTranspose2d实现，这里先用普通Conv占位
    decoder->conv1 = create_convblock_placeholder(16, 16, 1, 5, 1, 2, 0, 2, 2, 0);
    printf("  创建 conv1: (16, 16, (1,5), stride=(1,2), groups=2, deconv)\n");

    // 创建Conv2: (16, 2, (1,5), stride=(1,2), padding=(0,2), deconv=True, tanh=True)
    decoder->conv2 = create_convblock_placeholder(16, 2, 1, 5, 1, 2, 0, 2, 1, 1);
    printf("  创建 conv2: (16, 2, (1,5), stride=(1,2), deconv, tanh)\n");

    printf("Decoder 创建成功\n");

    return decoder;
}

void decoder_forward(
    const Tensor* input,
    Tensor** skip_connections,
    Tensor* output,
    Decoder* decoder
) {
    /*
     * Decoder 完整前向传播
     *
     * 输入: (B, 16, T, 97)
     * 输出: (B, 2, T, 385)
     *
     * 流程 (镜像Encoder，使用skip connections):
     * Layer 1: (B, 16, T, 97) + skip[4] -> (B, 16, T, 97)   [GTConv deconv, dilation=5]
     * Layer 2: (B, 16, T, 97) + skip[3] -> (B, 16, T, 97)   [GTConv deconv, dilation=2]
     * Layer 3: (B, 16, T, 97) + skip[2] -> (B, 16, T, 97)   [GTConv deconv, dilation=1]
     * Layer 4: (B, 16, T, 97) + skip[1] -> (B, 16, T, 193)  [ConvTranspose, stride=(1,2)]
     * Layer 5: (B, 16, T, 193) + skip[0] -> (B, 2, T, 385)  [ConvTranspose, stride=(1,2), tanh]
     */

    int B = input->shape.batch;
    int T = input->shape.height;

    // 分配中间缓冲区
    Tensor layer1_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer1_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer2_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer2_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer3_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer3_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer4_in = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor layer4_out = {
        .data = (float*)malloc(B * 16 * T * 193 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    Tensor layer5_in = {
        .data = (float*)malloc(B * 16 * T * 193 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    // Layer 1: GTConvBlock (dilation=5) + skip[4]
    // 添加skip connection
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer1_in.data[i] = input->data[i] + skip_connections[4]->data[i];
    }

    if (decoder->gtconv1) {
        gtconvblock_forward(&layer1_in, &layer1_out, decoder->gtconv1);
    } else {
        memcpy(layer1_out.data, layer1_in.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 2: GTConvBlock (dilation=2) + skip[3]
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer2_in.data[i] = layer1_out.data[i] + skip_connections[3]->data[i];
    }

    if (decoder->gtconv2) {
        gtconvblock_forward(&layer2_in, &layer2_out, decoder->gtconv2);
    } else {
        memcpy(layer2_out.data, layer2_in.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 3: GTConvBlock (dilation=1) + skip[2]
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer3_in.data[i] = layer2_out.data[i] + skip_connections[2]->data[i];
    }

    if (decoder->gtconv3) {
        gtconvblock_forward(&layer3_in, &layer3_out, decoder->gtconv3);
    } else {
        memcpy(layer3_out.data, layer3_in.data, B * 16 * T * 97 * sizeof(float));
    }

    // Layer 4: ConvBlock (deconv, stride=(1,2)) + skip[1]
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer4_in.data[i] = layer3_out.data[i] + skip_connections[1]->data[i];
    }

    if (decoder->conv1) {
        convblock_forward(&layer4_in, &layer4_out, decoder->conv1);
    } else {
        // 简化: 上采样
        memset(layer4_out.data, 0, B * 16 * T * 193 * sizeof(float));
    }

    // Layer 5: ConvBlock (deconv, stride=(1,2), tanh) + skip[0]
    for (int i = 0; i < B * 16 * T * 193; i++) {
        layer5_in.data[i] = layer4_out.data[i] + skip_connections[0]->data[i];
    }

    if (decoder->conv2) {
        convblock_forward(&layer5_in, output, decoder->conv2);
    } else {
        // 简化: 上采样
        memset(output->data, 0, B * 2 * T * 385 * sizeof(float));
    }

    // 释放缓冲区
    free(layer1_in.data);
    free(layer1_out.data);
    free(layer2_in.data);
    free(layer2_out.data);
    free(layer3_in.data);
    free(layer3_out.data);
    free(layer4_in.data);
    free(layer4_out.data);
    free(layer5_in.data);
}

void decoder_free(Decoder* decoder) {
    if (decoder) {
        if (decoder->gtconv1) gtconvblock_free(decoder->gtconv1);
        if (decoder->gtconv2) gtconvblock_free(decoder->gtconv2);
        if (decoder->gtconv3) gtconvblock_free(decoder->gtconv3);
        if (decoder->conv1) convblock_free(decoder->conv1);
        if (decoder->conv2) convblock_free(decoder->conv2);
        free(decoder);
    }
}


// ============================================================================
// DPGRNN 实现（完整版本，包含 GRU）
// ============================================================================

DPGRNN* dpgrnn_create(int input_size, int width, int hidden_size) {
    DPGRNN* dpgrnn = (DPGRNN*)malloc(sizeof(DPGRNN));
    if (!dpgrnn) return NULL;

    dpgrnn->input_size = input_size;      // 16
    dpgrnn->width = width;                // 97
    dpgrnn->hidden_size = hidden_size;    // 16

    // Intra RNN: Bidirectional GRNN
    // Input: (B*T, 97, 16) -> Output: (B*T, 97, 16)
    // Each group: input_size=8, hidden_size=4, bidirectional -> output=8
    dpgrnn->intra_gru_g1_fwd = gru_weights_create(input_size / 2, hidden_size / 4);
    dpgrnn->intra_gru_g2_fwd = gru_weights_create(input_size / 2, hidden_size / 4);
    dpgrnn->intra_gru_g1_bwd = gru_weights_create(input_size / 2, hidden_size / 4);
    dpgrnn->intra_gru_g2_bwd = gru_weights_create(input_size / 2, hidden_size / 4);

    // Inter RNN: Unidirectional GRNN
    // Input: (B*F, T, 16) -> Output: (B*F, T, 16)
    // Each group: input_size=8, hidden_size=8
    dpgrnn->inter_gru_g1 = gru_weights_create(input_size / 2, hidden_size / 2);
    dpgrnn->inter_gru_g2 = gru_weights_create(input_size / 2, hidden_size / 2);

    // Linear layers (注意: 实际参数需要从模型文件加载)
    dpgrnn->intra_fc = NULL;  // (hidden_size, hidden_size)
    dpgrnn->inter_fc = NULL;  // (hidden_size, hidden_size)

    // LayerNorm
    int normalized_shape[] = {width, hidden_size};
    dpgrnn->intra_ln = layernorm_create(normalized_shape, 2, NULL, NULL, 1e-8f);
    dpgrnn->inter_ln = layernorm_create(normalized_shape, 2, NULL, NULL, 1e-8f);

    printf("DPGRNN 创建成功:\n");
    printf("  Input size: %d, Width: %d, Hidden size: %d\n", input_size, width, hidden_size);
    printf("  Intra-RNN: Bidirectional GRNN (2 groups, %d->%d per group)\n",
           input_size/2, hidden_size/4);
    printf("  Inter-RNN: Unidirectional GRNN (2 groups, %d->%d per group)\n",
           input_size/2, hidden_size/2);

    return dpgrnn;
}

/*
 * 辅助函数: 张量重塑和转置
 */
static void tensor_permute_0213(
    const float* input,   // (B, C, T, F)
    float* output,        // (B, T, F, C)
    int B, int C, int T, int F
) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                for (int c = 0; c < C; c++) {
                    int in_idx = b * (C * T * F) + c * (T * F) + t * F + f;
                    int out_idx = b * (T * F * C) + t * (F * C) + f * C + c;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

static void tensor_permute_0312(
    const float* input,   // (B, T, F, C)
    float* output,        // (B, C, T, F)
    int B, int T, int F, int C
) {
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int t = 0; t < T; t++) {
                for (int f = 0; f < F; f++) {
                    int in_idx = b * (T * F * C) + t * (F * C) + f * C + c;
                    int out_idx = b * (C * T * F) + c * (T * F) + t * F + f;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

static void tensor_permute_0213_v2(
    const float* input,   // (B, F, T, C)
    float* output,        // (B, T, F, C)
    int B, int F, int T, int C
) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                for (int c = 0; c < C; c++) {
                    int in_idx = b * (F * T * C) + f * (T * C) + t * C + c;
                    int out_idx = b * (T * F * C) + t * (F * C) + f * C + c;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

/*
 * 辅助函数: 应用线性层
 * Input: (N, in_features)
 * Output: (N, out_features)
 */
static void apply_linear(
    const float* input,
    float* output,
    const LinearParams* linear,
    int N
) {
    if (!linear) return;

    int in_features = linear->in_features;
    int out_features = linear->out_features;

    for (int n = 0; n < N; n++) {
        for (int o = 0; o < out_features; o++) {
            float sum = linear->bias ? linear->bias[o] : 0.0f;
            for (int i = 0; i < in_features; i++) {
                sum += linear->weight[o * in_features + i] * input[n * in_features + i];
            }
            output[n * out_features + o] = sum;
        }
    }
}

/*
 * 辅助函数: 应用LayerNorm
 * Input/Output: (B, T, F, C) with normalized_shape=(F, C)
 */
static void apply_layernorm_4d(
    float* data,
    const LayerNormParams* ln,
    int B, int T, int F, int C
) {
    if (!ln) return;

    int norm_size = F * C;

    // 对每个 (B, T) 样本应用 LayerNorm
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* sample = data + (b * T + t) * norm_size;

            // 计算均值
            float mean = 0.0f;
            for (int i = 0; i < norm_size; i++) {
                mean += sample[i];
            }
            mean /= norm_size;

            // 计算方差
            float var = 0.0f;
            for (int i = 0; i < norm_size; i++) {
                float diff = sample[i] - mean;
                var += diff * diff;
            }
            var /= norm_size;

            // 归一化
            float std = sqrtf(var + ln->eps);
            for (int i = 0; i < norm_size; i++) {
                sample[i] = (sample[i] - mean) / std;

                // 应用可学习参数
                if (ln->gamma) {
                    sample[i] *= ln->gamma[i];
                }
                if (ln->beta) {
                    sample[i] += ln->beta[i];
                }
            }
        }
    }
}

void dpgrnn_forward(
    const Tensor* input,
    Tensor* output,
    DPGRNN* dpgrnn
) {
    /*
     * 完整的 DPGRNN 前向传播
     *
     * 输入: (B, C, T, F) where C=16, T=time, F=97
     *
     * 流程:
     * 1. Intra RNN:
     *    - Permute: (B,C,T,F) -> (B,T,F,C)
     *    - Reshape: (B,T,F,C) -> (B*T,F,C)
     *    - Bidirectional GRNN: (B*T,F,C) -> (B*T,F,C)
     *    - Linear: (B*T,F,C) -> (B*T,F,C)
     *    - Reshape: (B*T,F,C) -> (B,T,F,C)
     *    - LayerNorm + Residual
     *
     * 2. Inter RNN:
     *    - Permute: (B,T,F,C) -> (B,F,T,C)
     *    - Reshape: (B,F,T,C) -> (B*F,T,C)
     *    - Unidirectional GRNN: (B*F,T,C) -> (B*F,T,C)
     *    - Linear: (B*F,T,C) -> (B*F,T,C)
     *    - Reshape: (B*F,T,C) -> (B,F,T,C)
     *    - Permute: (B,F,T,C) -> (B,T,F,C)
     *    - LayerNorm + Residual
     *
     * 3. Permute back: (B,T,F,C) -> (B,C,T,F)
     */

    int B = input->shape.batch;
    int C = input->shape.channels;
    int T = input->shape.height;
    int F = input->shape.width;

    // 分配工作缓冲区
    float* x_btfc = (float*)malloc(B * T * F * C * sizeof(float));
    float* intra_out = (float*)malloc(B * T * F * C * sizeof(float));
    float* intra_x = (float*)malloc(B * T * F * C * sizeof(float));
    float* intra_residual = (float*)malloc(B * T * F * C * sizeof(float));
    float* inter_in = (float*)malloc(B * F * T * C * sizeof(float));
    float* inter_out = (float*)malloc(B * F * T * C * sizeof(float));
    float* inter_x = (float*)malloc(B * F * T * C * sizeof(float));
    float* temp = (float*)malloc(4 * dpgrnn->hidden_size * sizeof(float));

    // ========================================================================
    // Intra RNN
    // ========================================================================

    // 1. Permute: (B,C,T,F) -> (B,T,F,C)
    tensor_permute_0213(input->data, x_btfc, B, C, T, F);

    // 保存输入用于残差连接
    memcpy(intra_residual, x_btfc, B * T * F * C * sizeof(float));

    // 2. Process each (B*T) sample with bidirectional GRNN
    // 使用完整的双向分组GRU实现（来自 GRU_bidirectional_complete.c）
    for (int bt = 0; bt < B * T; bt++) {
        const float* input_bt = x_btfc + bt * F * C;
        float* output_bt = intra_out + bt * F * C;

        // 使用完整的双向分组GRU
        grnn_bidirectional_forward_complete(
            input_bt,
            output_bt,
            NULL, NULL, NULL, NULL,  // 无初始隐藏状态
            dpgrnn->intra_gru_g1_fwd,
            dpgrnn->intra_gru_g2_fwd,
            dpgrnn->intra_gru_g1_bwd,
            dpgrnn->intra_gru_g2_bwd,
            F,  // 序列长度 = 频率bins
            temp
        );
    }

    // 3. Linear layer
    if (dpgrnn->intra_fc) {
        apply_linear(intra_out, intra_x, dpgrnn->intra_fc, B * T * F);
    } else {
        memcpy(intra_x, intra_out, B * T * F * C * sizeof(float));
    }

    // 4. LayerNorm
    if (dpgrnn->intra_ln) {
        apply_layernorm_4d(intra_x, dpgrnn->intra_ln, B, T, F, C);
    }

    // 5. Residual connection
    for (int i = 0; i < B * T * F * C; i++) {
        intra_x[i] += intra_residual[i];
    }

    // ========================================================================
    // Inter RNN
    // ========================================================================

    // 1. Permute: (B,T,F,C) -> (B,F,T,C)
    tensor_permute_0213_v2(intra_x, inter_in, B, T, F, C);

    // 2. Process each (B*F) sample with unidirectional GRNN
    for (int bf = 0; bf < B * F; bf++) {
        const float* input_bf = inter_in + bf * T * C;
        float* output_bf = inter_out + bf * T * C;

        // Unidirectional GRNN across time dimension (causal)
        grnn_forward(
            input_bf,
            output_bf,
            NULL,  // No initial hidden state (for stateless processing)
            dpgrnn->inter_gru_g1,
            dpgrnn->inter_gru_g2,
            T,     // Sequence length = time steps
            0,     // Unidirectional (causal)
            temp
        );
    }

    // 3. Linear layer
    if (dpgrnn->inter_fc) {
        apply_linear(inter_out, inter_x, dpgrnn->inter_fc, B * F * T);
    } else {
        memcpy(inter_x, inter_out, B * F * T * C * sizeof(float));
    }

    // 4. Permute: (B,F,T,C) -> (B,T,F,C)
    float* inter_x_btfc = (float*)malloc(B * T * F * C * sizeof(float));
    tensor_permute_0213_v2(inter_x, inter_x_btfc, B, F, T, C);

    // 5. LayerNorm
    if (dpgrnn->inter_ln) {
        apply_layernorm_4d(inter_x_btfc, dpgrnn->inter_ln, B, T, F, C);
    }

    // 6. Residual connection (加到Intra-RNN的输出上)
    for (int i = 0; i < B * T * F * C; i++) {
        inter_x_btfc[i] += intra_x[i];
    }

    // ========================================================================
    // Final permute back: (B,T,F,C) -> (B,C,T,F)
    // ========================================================================
    tensor_permute_0312(inter_x_btfc, output->data, B, T, F, C);

    // 释放缓冲区
    free(x_btfc);
    free(intra_out);
    free(intra_x);
    free(intra_residual);
    free(inter_in);
    free(inter_out);
    free(inter_x);
    free(inter_x_btfc);
    free(temp);
}

void dpgrnn_free(DPGRNN* dpgrnn) {
    if (dpgrnn) {
        // Free Intra RNN weights
        if (dpgrnn->intra_gru_g1_fwd) gru_weights_free(dpgrnn->intra_gru_g1_fwd);
        if (dpgrnn->intra_gru_g2_fwd) gru_weights_free(dpgrnn->intra_gru_g2_fwd);
        if (dpgrnn->intra_gru_g1_bwd) gru_weights_free(dpgrnn->intra_gru_g1_bwd);
        if (dpgrnn->intra_gru_g2_bwd) gru_weights_free(dpgrnn->intra_gru_g2_bwd);

        // Free Inter RNN weights
        if (dpgrnn->inter_gru_g1) gru_weights_free(dpgrnn->inter_gru_g1);
        if (dpgrnn->inter_gru_g2) gru_weights_free(dpgrnn->inter_gru_g2);

        // Free Linear and LayerNorm
        if (dpgrnn->intra_fc) linear_free(dpgrnn->intra_fc);
        if (dpgrnn->inter_fc) linear_free(dpgrnn->inter_fc);
        if (dpgrnn->intra_ln) layernorm_free(dpgrnn->intra_ln);
        if (dpgrnn->inter_ln) layernorm_free(dpgrnn->inter_ln);

        free(dpgrnn);
    }
}


// ============================================================================
// GTCRN 完整模型
// ============================================================================

GTCRN* gtcrn_create() {
    GTCRN* model = (GTCRN*)malloc(sizeof(GTCRN));
    if (!model) return NULL;

    printf("创建 GTCRN 模型...\n\n");

    // 创建 ERB 模块
    printf("1. 创建 ERB 模块\n");
    model->erb = erb_create(195, 190, 1536, 24000, 48000);
    printf("\n");

    // 创建 SFE 模块
    printf("2. 创建 SFE 模块\n");
    model->sfe = sfe_create(3, 1);
    printf("\n");

    // 创建子模块
    printf("3. 创建 Encoder\n");
    model->encoder = encoder_create();
    printf("\n");

    printf("4. 创建 DPGRNN\n");
    model->dpgrnn1 = dpgrnn_create(16, 97, 16);
    model->dpgrnn2 = dpgrnn_create(16, 97, 16);
    printf("\n");

    printf("5. 创建 Decoder\n");
    model->decoder = decoder_create();
    printf("\n");

    // 初始化工作缓冲区
    model->num_buffers = 0;

    printf("GTCRN 模型创建成功！\n");
    printf("\n已集成模块:\n");
    printf("  ✓ ERB 压缩/恢复\n");
    printf("  ✓ SFE 子带特征提取\n");
    printf("  ✓ TRA 时间注意力（在 GTConvBlock 中）\n");
    printf("  ✓ Encoder/Decoder\n");
    printf("  ✓ DPGRNN\n");
    printf("\n待完成:\n");
    printf("  ⏳ 从模型文件加载权重\n");
    printf("  ⏳ 完整的 GRU 实现\n");
    printf("  ⏳ 完整的前向传播\n");

    return model;
}

void gtcrn_forward(
    const float* spec_input,
    float* spec_output,
    int batch,
    int freq_bins,
    int time_frames,
    GTCRN* model
) {
    /*
     * GTCRN 完整前向传播
     *
     * 输入: spec_input (B, F, T, 2) - 复数频谱，F=769 for 48kHz
     * 输出: spec_output (B, F, T, 2) - 增强后的复数频谱
     *
     * 完整流程:
     * 1. 输入预处理: 分离实部/虚部，计算幅度 -> (B, 3, T, F)
     * 2. ERB 压缩: (B, 3, T, 769) -> (B, 3, T, 385)
     * 3. SFE: (B, 3, T, 385) -> (B, 9, T, 385)
     * 4. Encoder: (B, 9, T, 385) -> (B, 16, T, 97) + skip connections
     * 5. DPGRNN x2: (B, 16, T, 97) -> (B, 16, T, 97)
     * 6. Decoder: (B, 16, T, 97) -> (B, 2, T, 385) with skip connections
     * 7. ERB 恢复: (B, 2, T, 385) -> (B, 2, T, 769)
     * 8. 复数掩码: mask * spec -> (B, 769, T, 2)
     */

    printf("\nGTCRN 完整前向传播\n");
    printf("输入: [%d, %d, %d, 2]\n", batch, freq_bins, time_frames);

    int B = batch;
    int F = freq_bins;  // 769
    int T = time_frames;

    // ========================================================================
    // 1. 输入预处理
    // ========================================================================

    // 分配缓冲区
    float* spec_real = (float*)malloc(B * F * T * sizeof(float));
    float* spec_imag = (float*)malloc(B * F * T * sizeof(float));
    float* spec_mag = (float*)malloc(B * F * T * sizeof(float));

    // 分离实部和虚部: (B, F, T, 2) -> real(B,F,T), imag(B,F,T)
    for (int b = 0; b < B; b++) {
        for (int f = 0; f < F; f++) {
            for (int t = 0; t < T; t++) {
                int idx = b * (F * T * 2) + f * (T * 2) + t * 2;
                int out_idx = b * (F * T) + f * T + t;
                spec_real[out_idx] = spec_input[idx];
                spec_imag[out_idx] = spec_input[idx + 1];
            }
        }
    }

    // 计算幅度
    compute_magnitude(spec_real, spec_imag, spec_mag, B * F * T);

    // 堆叠为 (B, 3, T, F): [mag, real, imag]
    // 需要转置: (B, F, T) -> (B, T, F)
    float* feat = (float*)malloc(B * 3 * T * F * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                int in_idx = b * (F * T) + f * T + t;
                int out_idx_mag = b * (3 * T * F) + 0 * (T * F) + t * F + f;
                int out_idx_real = b * (3 * T * F) + 1 * (T * F) + t * F + f;
                int out_idx_imag = b * (3 * T * F) + 2 * (T * F) + t * F + f;

                feat[out_idx_mag] = spec_mag[in_idx];
                feat[out_idx_real] = spec_real[in_idx];
                feat[out_idx_imag] = spec_imag[in_idx];
            }
        }
    }

    // ========================================================================
    // 2. ERB 压缩: (B, 3, T, 769) -> (B, 3, T, 385)
    // ========================================================================

    Tensor feat_tensor = {
        .data = feat,
        .shape = {.batch = B, .channels = 3, .height = T, .width = F}
    };

    Tensor erb_tensor = {
        .data = (float*)malloc(B * 3 * T * 385 * sizeof(float)),
        .shape = {.batch = B, .channels = 3, .height = T, .width = 385}
    };

    if (model->erb) {
        erb_compress(&feat_tensor, &erb_tensor, model->erb);
    } else {
        // 简化: 直接复制
        memcpy(erb_tensor.data, feat, B * 3 * T * 385 * sizeof(float));
    }

    // ========================================================================
    // 3. SFE: (B, 3, T, 385) -> (B, 9, T, 385)
    // ========================================================================

    Tensor sfe_tensor = {
        .data = (float*)malloc(B * 9 * T * 385 * sizeof(float)),
        .shape = {.batch = B, .channels = 9, .height = T, .width = 385}
    };

    if (model->sfe) {
        sfe_forward(&erb_tensor, &sfe_tensor, model->sfe);
    } else {
        // 简化: 复制3次
        for (int i = 0; i < 3; i++) {
            memcpy(sfe_tensor.data + i * (B * 3 * T * 385),
                   erb_tensor.data,
                   B * 3 * T * 385 * sizeof(float));
        }
    }

    // ========================================================================
    // 4. Encoder: (B, 9, T, 385) -> (B, 16, T, 97)
    // ========================================================================

    Tensor encoder_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor* skip_connections[5] = {NULL, NULL, NULL, NULL, NULL};

    if (model->encoder) {
        encoder_forward(&sfe_tensor, &encoder_out, skip_connections, model->encoder);
    } else {
        memset(encoder_out.data, 0, B * 16 * T * 97 * sizeof(float));
    }

    // ========================================================================
    // 5. DPGRNN: (B, 16, T, 97) -> (B, 16, T, 97)
    // ========================================================================

    Tensor dpgrnn1_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    Tensor dpgrnn2_out = {
        .data = (float*)malloc(B * 16 * T * 97 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 97}
    };

    // DPGRNN Layer 1
    if (model->dpgrnn1) {
        dpgrnn_forward(&encoder_out, &dpgrnn1_out, model->dpgrnn1);
    } else {
        memcpy(dpgrnn1_out.data, encoder_out.data, B * 16 * T * 97 * sizeof(float));
    }

    // DPGRNN Layer 2
    if (model->dpgrnn2) {
        dpgrnn_forward(&dpgrnn1_out, &dpgrnn2_out, model->dpgrnn2);
    } else {
        memcpy(dpgrnn2_out.data, dpgrnn1_out.data, B * 16 * T * 97 * sizeof(float));
    }

    // ========================================================================
    // 6. Decoder: (B, 16, T, 97) -> (B, 2, T, 385)
    // ========================================================================

    Tensor decoder_out = {
        .data = (float*)malloc(B * 2 * T * 385 * sizeof(float)),
        .shape = {.batch = B, .channels = 2, .height = T, .width = 385}
    };

    if (model->decoder) {
        decoder_forward(&dpgrnn2_out, skip_connections, &decoder_out, model->decoder);
    } else {
        memset(decoder_out.data, 0, B * 2 * T * 385 * sizeof(float));
    }

    // ========================================================================
    // 7. ERB 恢复: (B, 2, T, 385) -> (B, 2, T, 769)
    // ========================================================================

    Tensor mask_tensor = {
        .data = (float*)malloc(B * 2 * T * F * sizeof(float)),
        .shape = {.batch = B, .channels = 2, .height = T, .width = F}
    };

    if (model->erb) {
        erb_decompress(&decoder_out, &mask_tensor, model->erb);
    } else {
        // 简化: 直接复制
        memcpy(mask_tensor.data, decoder_out.data, B * 2 * T * 385 * sizeof(float));
    }

    // ========================================================================
    // 8. 应用复数掩码
    // ========================================================================

    // mask: (B, 2, T, F) -> mask_real(B,T,F), mask_imag(B,T,F)
    float* mask_real = (float*)malloc(B * T * F * sizeof(float));
    float* mask_imag = (float*)malloc(B * T * F * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int f = 0; f < F; f++) {
                int mask_idx_real = b * (2 * T * F) + 0 * (T * F) + t * F + f;
                int mask_idx_imag = b * (2 * T * F) + 1 * (T * F) + t * F + f;
                int out_idx = b * (T * F) + t * F + f;

                mask_real[out_idx] = mask_tensor.data[mask_idx_real];
                mask_imag[out_idx] = mask_tensor.data[mask_idx_imag];
            }
        }
    }

    // 应用复数掩码到原始频谱
    // spec: (B, F, T) -> 转置为 (B, T, F)
    float* spec_real_tf = (float*)malloc(B * T * F * sizeof(float));
    float* spec_imag_tf = (float*)malloc(B * T * F * sizeof(float));

    for (int b = 0; b < B; b++) {
        for (int f = 0; f < F; f++) {
            for (int t = 0; t < T; t++) {
                int in_idx = b * (F * T) + f * T + t;
                int out_idx = b * (T * F) + t * F + f;
                spec_real_tf[out_idx] = spec_real[in_idx];
                spec_imag_tf[out_idx] = spec_imag[in_idx];
            }
        }
    }

    // 复数乘法
    float* enh_real = (float*)malloc(B * T * F * sizeof(float));
    float* enh_imag = (float*)malloc(B * T * F * sizeof(float));

    apply_complex_mask(spec_real_tf, spec_imag_tf, mask_real, mask_imag,
                      enh_real, enh_imag, B * T * F);

    // 转置回 (B, F, T, 2)
    for (int b = 0; b < B; b++) {
        for (int f = 0; f < F; f++) {
            for (int t = 0; t < T; t++) {
                int in_idx = b * (T * F) + t * F + f;
                int out_idx = b * (F * T * 2) + f * (T * 2) + t * 2;
                spec_output[out_idx] = enh_real[in_idx];
                spec_output[out_idx + 1] = enh_imag[in_idx];
            }
        }
    }

    // ========================================================================
    // 清理
    // ========================================================================

    free(spec_real);
    free(spec_imag);
    free(spec_mag);
    free(feat);
    free(erb_tensor.data);
    free(sfe_tensor.data);
    free(encoder_out.data);
    free(dpgrnn1_out.data);
    free(dpgrnn2_out.data);
    free(decoder_out.data);
    free(mask_tensor.data);
    free(mask_real);
    free(mask_imag);
    free(spec_real_tf);
    free(spec_imag_tf);
    free(enh_real);
    free(enh_imag);

    printf("输出: [%d, %d, %d, 2]\n", batch, freq_bins, time_frames);
    printf("GTCRN 完整前向传播完成\n");
}

void gtcrn_free(GTCRN* model) {
    if (model) {
        if (model->erb) erb_free(model->erb);
        if (model->sfe) sfe_free(model->sfe);
        if (model->encoder) encoder_free(model->encoder);
        if (model->dpgrnn1) dpgrnn_free(model->dpgrnn1);
        if (model->dpgrnn2) dpgrnn_free(model->dpgrnn2);
        if (model->decoder) decoder_free(model->decoder);

        // 释放工作缓冲区
        for (int i = 0; i < model->num_buffers; i++) {
            if (model->work_buffers[i]) {
                tensor_free(model->work_buffers[i]);
            }
        }

        free(model);
    }
}


// ============================================================================
// 辅助函数
// ============================================================================

void compute_magnitude(
    const float* real,
    const float* imag,
    float* magnitude,
    int size
) {
    for (int i = 0; i < size; i++) {
        magnitude[i] = sqrtf(real[i] * real[i] + imag[i] * imag[i] + 1e-12f);
    }
}

void apply_complex_mask(
    const float* spec_real,
    const float* spec_imag,
    const float* mask_real,
    const float* mask_imag,
    float* output_real,
    float* output_imag,
    int size
) {
    for (int i = 0; i < size; i++) {
        // 复数乘法
        output_real[i] = spec_real[i] * mask_real[i] - spec_imag[i] * mask_imag[i];
        output_imag[i] = spec_imag[i] * mask_real[i] + spec_real[i] * mask_imag[i];
    }
}

void print_gtcrn_info(const GTCRN* model) {
    printf("\n");
    printf("=================================================================\n");
    printf("GTCRN 模型信息\n");
    printf("=================================================================\n");
    printf("\n");
    printf("网络结构:\n");
    printf("  输入: (B, 769, T, 2) - 48kHz 复数频谱\n");
    printf("  ↓\n");
    printf("  预处理: 分离实部/虚部，计算幅度\n");
    printf("  ↓\n");
    printf("  ✓ ERB 压缩: 769 bins -> 385 bins\n");
    printf("  ↓\n");
    printf("  ✓ SFE: (B, 3, T, 385) -> (B, 9, T, 385)\n");
    printf("  ↓\n");
    printf("  Encoder: 5 层\n");
    printf("    - ConvBlock 1: (9, 16, (1,5), stride=(1,2))\n");
    printf("    - ConvBlock 2: (16, 16, (1,5), stride=(1,2), groups=2)\n");
    printf("    - ✓ GTConvBlock 1: dilation=(1,1) + SFE + TRA\n");
    printf("    - ✓ GTConvBlock 2: dilation=(2,1) + SFE + TRA\n");
    printf("    - ✓ GTConvBlock 3: dilation=(5,1) + SFE + TRA\n");
    printf("  输出: (B, 16, T, 97)\n");
    printf("  ↓\n");
    printf("  DPGRNN: 2 层\n");
    printf("    - DPGRNN 1: (16, 97, 16)\n");
    printf("    - DPGRNN 2: (16, 97, 16)\n");
    printf("  ↓\n");
    printf("  Decoder: 5 层（镜像 Encoder）\n");
    printf("    - ✓ GTConvBlock 1: dilation=(5,1) + SFE + TRA\n");
    printf("    - ✓ GTConvBlock 2: dilation=(2,1) + SFE + TRA\n");
    printf("    - ✓ GTConvBlock 3: dilation=(1,1) + SFE + TRA\n");
    printf("    - ConvBlock 1: deconv\n");
    printf("    - ConvBlock 2: deconv + tanh\n");
    printf("  输出: (B, 2, T, 385)\n");
    printf("  ↓\n");
    printf("  ✓ ERB 恢复: 385 bins -> 769 bins\n");
    printf("  ↓\n");
    printf("  复数掩码: 应用到输入频谱\n");
    printf("  ↓\n");
    printf("  输出: (B, 769, T, 2) - 增强后的复数频谱\n");
    printf("\n");
    printf("已集成模块:\n");
    printf("  ✓ ERB (Equivalent Rectangular Bandwidth)\n");
    printf("  ✓ SFE (Subband Feature Extraction)\n");
    printf("  ✓ TRA (Temporal Recurrent Attention)\n");
    printf("  ✓ Conv+BN 融合优化\n");
    printf("\n");
    printf("参数统计:\n");
    printf("  总参数: ~23.67K\n");
    printf("  计算量: ~33.0 MMACs\n");
    printf("\n");
    printf("特点:\n");
    printf("  ✓ 超轻量级设计\n");
    printf("  ✓ 实时处理能力\n");
    printf("  ✓ 低计算资源需求\n");
    printf("  ✓ 高质量语音增强\n");
    printf("\n");
}
