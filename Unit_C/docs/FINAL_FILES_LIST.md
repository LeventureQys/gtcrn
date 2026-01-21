# GTCRN 实时降噪处理 - 最终文件清单

## 核心文件（必需）

### 1. 主要实现文件

| 文件 | 用途 | 状态 |
|------|------|------|
| **gtcrn_streaming.h** | 流式处理API头文件 | ✅ 已存在 |
| **gtcrn_streaming.c** | 流式处理API实现 | ✅ 已存在 |
| **gtcrn_streaming_impl.c** | 底层流式实现（DPGRNN、GTConvBlock流式函数） | ✅ 已创建 |
| **stream_conv.h** | 流式卷积头文件 | ✅ 已创建 |
| **stream_conv.c** | 流式卷积实现 | ✅ 已创建 |

### 2. 模型文件

| 文件 | 用途 | 状态 |
|------|------|------|
| **gtcrn_model.h** | 模型定义和接口 | ✅ 已更新（添加流式接口） |
| **gtcrn_model.c** | 模型实现 | ✅ 已存在 |
| **gtcrn_modules.h** | 模块定义（ERB、SFE、TRA） | ✅ 已存在 |
| **gtcrn_modules.c** | 模块实现（含TRA流式支持） | ✅ 已更新 |

### 3. 基础层文件

| 文件 | 用途 | 状态 |
|------|------|------|
| **GRU.h** | GRU实现头文件 | ✅ 已存在 |
| **GRU.c** | GRU实现 | ✅ 已存在 |
| **conv2d.h** | 卷积操作头文件 | ✅ 已存在 |
| **conv2d.c** | 卷积操作实现 | ✅ 已存在 |
| **nn_layers.h** | 神经网络层头文件 | ✅ 已存在 |
| **nn_layers.c** | 神经网络层实现 | ✅ 已存在 |
| **batchnorm2d.h** | BatchNorm头文件 | ✅ 已存在 |
| **batchnorm2d.c** | BatchNorm实现 | ✅ 已存在 |
| **layernorm.h** | LayerNorm头文件 | ✅ 已存在 |
| **layernorm.c** | LayerNorm实现 | ✅ 已存在 |

### 4. 示例程序

| 文件 | 用途 | 状态 |
|------|------|------|
| **example_realtime_denoise.c** | 完整使用示例 | ✅ 已存在 |

## 使用方式

### 方式1: 使用高级API（推荐）

**主要文件**: `gtcrn_streaming.h` + `gtcrn_streaming.c`

```c
#include "gtcrn_streaming.h"

int main() {
    // 1. 创建模型
    GTCRN* model = gtcrn_create();

    // 2. 创建流式处理器
    GTCRNStreaming* stream = gtcrn_streaming_create(model, 48000, 768);

    // 3. 处理音频块
    float input[768], output[768];
    gtcrn_streaming_process_chunk(stream, input, output);

    // 4. 清理
    gtcrn_streaming_free(stream);
    gtcrn_free(model);

    return 0;
}
```

### 方式2: 使用底层API（高级用户）

**主要文件**: `gtcrn_streaming_impl.c` 中的函数

```c
#include "gtcrn_model.h"

int main() {
    GTCRN* model = gtcrn_create();

    // 手动管理缓存
    float* conv_cache = calloc(cache_size, sizeof(float));
    float* tra_cache = calloc(tra_size, sizeof(float));
    float* inter_cache = calloc(inter_size, sizeof(float));

    // 处理单帧
    float spec_input[769 * 1 * 2];
    float spec_output[769 * 1 * 2];
    gtcrn_forward_stream(spec_input, spec_output,
                         conv_cache, tra_cache, inter_cache,
                         1, 769, model);

    // 清理
    free(conv_cache);
    free(tra_cache);
    free(inter_cache);
    gtcrn_free(model);

    return 0;
}
```

## 编译命令

### 完整编译（推荐）

```bash
cd Unit_C

gcc -o realtime_denoise \
    example_realtime_denoise.c \
    gtcrn_streaming.c \
    gtcrn_streaming_impl.c \
    stream_conv.c \
    gtcrn_model.c \
    gtcrn_modules.c \
    GRU.c \
    conv2d.c \
    nn_layers.c \
    batchnorm2d.c \
    layernorm.c \
    -lm -O2

./realtime_denoise input.wav output.wav weights/
```

### 编译为库

```bash
# 编译所有对象文件
gcc -c -O2 gtcrn_streaming.c -o gtcrn_streaming.o
gcc -c -O2 gtcrn_streaming_impl.c -o gtcrn_streaming_impl.o
gcc -c -O2 stream_conv.c -o stream_conv.o
gcc -c -O2 gtcrn_model.c -o gtcrn_model.o
gcc -c -O2 gtcrn_modules.c -o gtcrn_modules.o
gcc -c -O2 GRU.c -o GRU.o
gcc -c -O2 conv2d.c -o conv2d.o
gcc -c -O2 nn_layers.c -o nn_layers.o
gcc -c -O2 batchnorm2d.c -o batchnorm2d.o
gcc -c -O2 layernorm.c -o layernorm.o

# 创建静态库
ar rcs libgtcrn.a \
    gtcrn_streaming.o \
    gtcrn_streaming_impl.o \
    stream_conv.o \
    gtcrn_model.o \
    gtcrn_modules.o \
    GRU.o \
    conv2d.o \
    nn_layers.o \
    batchnorm2d.o \
    layernorm.o

# 使用库
gcc -o my_app my_app.c -L. -lgtcrn -lm
```

## 文件依赖关系

```
example_realtime_denoise.c
    ↓
gtcrn_streaming.h/c (推荐使用)
    ↓
gtcrn_streaming_impl.c (底层实现)
    ├─ dpgrnn_forward_stream()
    ├─ gtconvblock_forward_stream()
    └─ gtcrn_forward_stream()
    ↓
stream_conv.h/c (流式卷积)
    ├─ stream_conv2d_forward()
    └─ stream_conv_transpose2d_forward()
    ↓
gtcrn_model.h/c (模型定义)
    ↓
gtcrn_modules.h/c (模块实现)
    └─ tra_forward_stream() (TRA流式支持)
    ↓
GRU.h/c, conv2d.h/c, nn_layers.h/c (基础层)
```

## 最终推荐

**使用这些文件即可实现完整的实时降噪**:

### 必需文件（按优先级）:
1. ✅ `gtcrn_streaming.h` - 主要API接口
2. ✅ `gtcrn_streaming.c` - 主要API实现
3. ✅ `gtcrn_streaming_impl.c` - 底层流式实现
4. ✅ `stream_conv.h/c` - 流式卷积
5. ✅ `gtcrn_model.h/c` - 模型定义
6. ✅ `gtcrn_modules.h/c` - 模块实现（含TRA流式）
7. ✅ `GRU.h/c` - GRU实现
8. ✅ `conv2d.h/c` - 卷积操作
9. ✅ `nn_layers.h/c` - 神经网络层
10. ✅ `batchnorm2d.h/c` - BatchNorm
11. ✅ `layernorm.h/c` - LayerNorm

### 示例文件:
- ✅ `example_realtime_denoise.c` - 完整使用示例

### 文档文件:
- ✅ `REALTIME_FINAL_STATUS.md` - 最终状态说明
- ✅ `STREAMING_IMPLEMENTATION_STATUS.md` - 详细技术文档
- ✅ `TRA_FIX_SUMMARY.md` - TRA修复总结

## 总结

**所有文件已准备就绪！**

- ✅ 核心实现: 11个C/H文件
- ✅ 示例程序: 1个
- ✅ 文档: 3个

**直接使用 `gtcrn_streaming.h` 的API即可实现实时降噪处理！**
