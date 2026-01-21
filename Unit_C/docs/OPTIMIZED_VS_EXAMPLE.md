# gtcrn_streaming_optimized.c vs example_realtime_denoise.c

## 关键区别

### example_realtime_denoise.c
- **性质**: 示例程序（应用层）
- **作用**: 展示如何使用API
- **包含**:
  - WAV文件读写
  - 命令行参数处理
  - 性能统计
  - 完整的使用流程
- **依赖**: 调用 `gtcrn_streaming.h` 的API

### gtcrn_streaming_optimized.c
- **性质**: 核心实现（库层）
- **作用**: 提供优化的流式处理实现
- **包含**:
  - 优化的单帧处理函数
  - 状态缓存管理
  - DPGRNN流式处理
  - 内存优化
- **被调用**: 被应用程序使用

## 层次关系

```
应用层:
  example_realtime_denoise.c (示例程序)
      ↓ 调用

API层:
  gtcrn_streaming.h (接口定义)
      ↓ 实现

实现层:
  gtcrn_streaming_optimized.c (优化实现) ⭐ 最优
  gtcrn_streaming.c (基础实现)
      ↓ 调用

底层:
  gtcrn_streaming_impl.c (流式函数)
  stream_conv.c (流式卷积)
  gtcrn_model.c (模型)
  ...
```

## 实际使用场景

### 场景1: 你想直接运行程序处理音频文件

**使用**: `example_realtime_denoise.c`

```bash
# 编译
gcc -o denoise example_realtime_denoise.c \
    gtcrn_streaming.c gtcrn_streaming_optimized.c \
    ... (其他文件) -lm -O3

# 运行
./denoise input.wav output.wav weights/
```

**优点**:
- ✅ 开箱即用
- ✅ 包含完整功能（文件I/O、统计等）
- ✅ 适合快速测试

### 场景2: 你想集成到自己的项目

**使用**: `gtcrn_streaming_optimized.c` 提供的函数

```c
// 你的项目代码
#include "gtcrn_streaming.h"

void my_audio_callback(float* input, float* output, int size) {
    // 使用优化的处理函数
    gtcrn_streaming_process_chunk_optimized(stream, input, output);
}
```

**优点**:
- ✅ 灵活集成
- ✅ 最优性能
- ✅ 可自定义

## 性能对比

| 实现 | 类型 | 性能 | 用途 |
|------|------|------|------|
| **gtcrn_streaming_optimized.c** | 库实现 | ⚡ 最优 (~20ms) | 集成到项目 |
| gtcrn_streaming.c | 库实现 | 良好 (~32ms) | 集成到项目 |
| example_realtime_denoise.c | 应用程序 | 取决于使用的库 | 独立运行 |

## 代码对比

### example_realtime_denoise.c 的核心代码

```c
// 创建流式处理器
GTCRNStreaming* stream = gtcrn_streaming_create(model, 48000, 768);

// 处理音频块
for (int chunk = 0; chunk < total_chunks; chunk++) {
    gtcrn_streaming_process_chunk(  // 使用基础版本
        stream,
        input_audio->data + processed,
        output_audio->data + processed
    );
    processed += chunk_size;
}
```

### 如果修改为使用优化版本

```c
// 创建流式处理器（相同）
GTCRNStreaming* stream = gtcrn_streaming_create(model, 48000, 768);

// 处理音频块（使用优化版本）
for (int chunk = 0; chunk < total_chunks; chunk++) {
    gtcrn_streaming_process_chunk_optimized(  // ⭐ 使用优化版本
        stream,
        input_audio->data + processed,
        output_audio->data + processed
    );
    processed += chunk_size;
}
```

## 最终建议

### 如果你想要：

#### 1. 快速测试和验证
**使用**: `example_realtime_denoise.c`（修改为调用优化版本）

```c
// 在 example_realtime_denoise.c 中修改：
// 将 gtcrn_streaming_process_chunk()
// 改为 gtcrn_streaming_process_chunk_optimized()
```

#### 2. 集成到自己的项目
**使用**: 直接调用 `gtcrn_streaming_optimized.c` 的函数

```c
#include "gtcrn_streaming.h"

// 在你的代码中
gtcrn_streaming_process_chunk_optimized(stream, input, output);
```

#### 3. 最佳实践
**编译时包含所有文件**:

```bash
gcc -o your_app your_app.c \
    gtcrn_streaming.c \
    gtcrn_streaming_optimized.c \  # ⭐ 包含优化实现
    gtcrn_streaming_impl.c \
    stream_conv.c \
    gtcrn_model.c \
    gtcrn_modules.c \
    GRU.c conv2d.c nn_layers.c \
    batchnorm2d.c layernorm.c \
    -lm -O3
```

## 总结

| 文件 | 用途 | 何时使用 |
|------|------|---------|
| **gtcrn_streaming_optimized.c** | 优化的核心实现 | ⭐ 总是包含（获得最佳性能） |
| gtcrn_streaming.c | 基础实现和缓存管理 | 总是包含（提供基础功能） |
| example_realtime_denoise.c | 示例应用程序 | 学习、测试、快速原型 |

**最优方案**:
- 编译时包含 `gtcrn_streaming_optimized.c`
- 代码中调用 `gtcrn_streaming_process_chunk_optimized()`
- 参考 `example_realtime_denoise.c` 的使用方式

**这样你既能获得最优性能，又能快速上手！** 🚀
