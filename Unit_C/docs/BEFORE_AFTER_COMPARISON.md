# 修复前后代码对比

## 问题 1: Skip Connections 内存管理

### ❌ 修复前 (gtcrn_streaming_optimized.c)

```c
static int encoder_forward_streaming(
    const Tensor* input,
    Tensor* output,
    Tensor** skip_connections,  // ❌ 指向局部变量的指针
    Encoder* encoder,
    GTCRNStreaming* stream
) {
    // ❌ 分配局部缓冲区
    Tensor layer1_out = {
        .data = (float*)malloc(B * 16 * T * 193 * sizeof(float)),
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    // ❌ 设置指针指向局部变量
    if (skip_connections) skip_connections[0] = &layer1_out;

    // ... 处理

    // ❌ 释放内存 - skip_connections[0] 现在是悬空指针！
    free(layer1_out.data);

    return 0;
}

static int decoder_forward_streaming(
    const Tensor* input,
    Tensor** skip_connections,  // ❌ 指向已释放内存的指针
    Tensor* output,
    Decoder* decoder,
    GTCRNStreaming* stream
) {
    // ❌ 访问已释放的内存 - 段错误！
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer1_in.data[i] = input->data[i] + skip_connections[4]->data[i];
    }
}
```

### ✅ 修复后 (gtcrn_streaming_optimized_FIXED.c)

```c
// 在 gtcrn_streaming.h 中添加
typedef struct {
    float* data;
    int size;
} SkipBuffer;

typedef struct {
    // ... 其他字段
    SkipBuffer skip_buffers[5];  // ✅ 持久化内存
} GTCRNStreaming;

// 在 gtcrn_streaming.c 中初始化
GTCRNStreaming* gtcrn_streaming_create(...) {
    // ✅ 分配持久化 skip buffers
    for (int i = 0; i < 5; i++) {
        stream->skip_buffers[i].data = (float*)calloc(skip_sizes[i], sizeof(float));
    }
}

// 在 gtcrn_streaming_optimized_FIXED.c 中使用
static int encoder_forward_streaming(
    const Tensor* input,
    Tensor* output,
    GTCRNStreaming* stream,  // ✅ 传入 stream
    Encoder* encoder
) {
    // ✅ 使用持久化内存
    Tensor layer1_out = {
        .data = stream->skip_buffers[0].data,  // ✅ 持久化内存
        .shape = {.batch = B, .channels = 16, .height = T, .width = 193}
    };

    // ... 处理

    // ✅ 不释放 - 使用持久化 buffers
    return 0;
}

static int decoder_forward_streaming(
    const Tensor* input,
    GTCRNStreaming* stream,  // ✅ 传入 stream
    Tensor* output,
    Decoder* decoder
) {
    // ✅ 访问有效的持久化内存
    for (int i = 0; i < B * 16 * T * 97; i++) {
        layer1_in.data[i] = input->data[i] + stream->skip_buffers[4].data[i];
    }
}
```

---

## 问题 2: DPGRNN 缓存使用 static 变量

### ❌ 修复前 (gtcrn_streaming_optimized.c)

```c
static int dpgrnn_forward_streaming_wrapper(
    const Tensor* input,
    Tensor* output,
    DPGRNN* dpgrnn,
    DPGRNNCache* cache
) {
    // ❌ 使用 static 变量 - 所有实例共享！
    static float* persistent_inter_cache = NULL;
    static int cache_size = 0;

    int required_size = B * F * hidden_size;
    if (persistent_inter_cache == NULL || cache_size != required_size) {
        if (persistent_inter_cache) {
            free(persistent_inter_cache);
        }
        persistent_inter_cache = (float*)calloc(required_size, sizeof(float));
        cache_size = required_size;
    }

    // ❌ 使用全局 static 缓存
    dpgrnn_forward_stream(input, output, persistent_inter_cache, dpgrnn);

    return 0;
}
```

**问题**:
- ❌ 只能有一个流式处理器实例
- ❌ 多个实例会互相干扰
- ❌ 多线程不安全

### ✅ 修复后

```c
// 在 gtcrn_streaming.h 中添加
typedef struct {
    GRUCache* inter_gru_g1_cache;
    GRUCache* inter_gru_g2_cache;

    // ✅ 每个实例独立的缓存
    float* inter_cache_buffer;
    int inter_cache_size;
} DPGRNNCache;

// 在 gtcrn_streaming.c 中分配
DPGRNNCache* dpgrnn_cache_create(int hidden_size, int batch_size, int freq_bins) {
    DPGRNNCache* cache = (DPGRNNCache*)malloc(sizeof(DPGRNNCache));

    // ✅ 分配实例独立的缓存
    cache->inter_cache_size = batch_size * freq_bins * hidden_size;
    cache->inter_cache_buffer = (float*)calloc(cache->inter_cache_size, sizeof(float));

    return cache;
}

// 在 gtcrn_streaming_optimized_FIXED.c 中使用
static int dpgrnn_forward_streaming_wrapper(
    const Tensor* input,
    Tensor* output,
    DPGRNN* dpgrnn,
    DPGRNNCache* cache
) {
    // ✅ 使用实例独立的缓存
    dpgrnn_forward_stream(input, output, cache->inter_cache_buffer, dpgrnn);

    return 0;
}
```

**效果**:
- ✅ 支持多个流式处理器实例
- ✅ 每个实例有独立的缓存
- ✅ 线程安全（实例级别）

---

## 问题 3: 函数签名变化

### 修复前

```c
// gtcrn_streaming.h
DPGRNNCache* dpgrnn_cache_create(int hidden_size);

// gtcrn_streaming.c
GTCRNStreaming* gtcrn_streaming_create(...) {
    stream->dpgrnn1_cache = dpgrnn_cache_create(16);
    stream->dpgrnn2_cache = dpgrnn_cache_create(16);
}
```

### 修复后

```c
// gtcrn_streaming.h
DPGRNNCache* dpgrnn_cache_create(int hidden_size, int batch_size, int freq_bins);

// gtcrn_streaming.c
GTCRNStreaming* gtcrn_streaming_create(...) {
    // ✅ 传入 batch_size 和 freq_bins
    stream->dpgrnn1_cache = dpgrnn_cache_create(16, 1, 97);
    stream->dpgrnn2_cache = dpgrnn_cache_create(16, 1, 97);
}
```

---

## 内存生命周期对比

### ❌ 修复前的内存生命周期

```
gtcrn_streaming_process_frame_optimized() {
    ├─ encoder_forward_streaming() {
    │   ├─ malloc(layer1_out)        ──┐
    │   ├─ skip[0] = &layer1_out       │ 局部作用域
    │   ├─ ... 处理 ...                │
    │   └─ free(layer1_out)          ──┘ ← 内存释放
    │   }
    │
    │   ↓ skip[0] 现在是悬空指针！
    │
    ├─ decoder_forward_streaming() {
    │   └─ 访问 skip[0]->data        ← ❌ 段错误！
    │   }
    }
```

### ✅ 修复后的内存生命周期

```
gtcrn_streaming_create() {
    └─ malloc(skip_buffers[0..4])    ──┐
}                                       │
                                        │ 持久化内存
gtcrn_streaming_process_frame_optimized() {
    ├─ encoder_forward_streaming() {   │
    │   ├─ 使用 skip_buffers[0]       │ ← ✅ 有效
    │   └─ 不释放                      │
    │   }                               │
    │                                   │
    ├─ decoder_forward_streaming() {   │
    │   └─ 访问 skip_buffers[0]       │ ← ✅ 有效
    │   }                               │
}                                       │
                                        │
gtcrn_streaming_free() {                │
    └─ free(skip_buffers[0..4])      ──┘
}
```

---

## 编译命令对比

### ❌ 修复前（会崩溃）

```bash
gcc -o denoise example_realtime_denoise.c \
    gtcrn_streaming_optimized.c \      # ❌ 有 bug 的版本
    gtcrn_streaming.c \
    gtcrn_streaming_impl.c \
    gtcrn_model.c gtcrn_modules.c stream_conv.c \
    stft.c weight_loader.c GRU.c conv2d.c \
    batchnorm2d.c nn_layers.c layernorm.c -lm -O2
```

### ✅ 修复后（安全）

```bash
gcc -o denoise example_realtime_denoise.c \
    gtcrn_streaming_optimized_FIXED.c \  # ✅ 修复后的版本
    gtcrn_streaming.c \
    gtcrn_streaming_impl.c \
    gtcrn_model.c gtcrn_modules.c stream_conv.c \
    stft.c weight_loader.c GRU.c conv2d.c \
    batchnorm2d.c nn_layers.c layernorm.c -lm -O2
```

---

## 运行结果对比

### ❌ 修复前

```
$ ./denoise input.wav output.wav weights/

Step 1: Loading audio...
Reading WAV: input.wav
  Sample rate: 48000 Hz
  Samples: 96000
  Duration: 2.00 seconds

Step 2: Creating GTCRN model...
...

Step 5: Processing audio...
Processing 125 chunks...
Segmentation fault (core dumped)    ← ❌ 崩溃！
```

### ✅ 修复后

```
$ ./denoise input.wav output.wav weights/

Step 1: Loading audio...
Reading WAV: input.wav
  Sample rate: 48000 Hz
  Samples: 96000
  Duration: 2.00 seconds

Step 2: Creating GTCRN model...
...

Step 5: Processing audio...
Processing 125 chunks...
  Progress: 100.0% (125/125 chunks)

Processing complete!
  Audio duration: 2.00 seconds
  Processing time: 0.10 seconds
  Real-time factor: 0.050 (20.0x faster than real-time)
  Frames processed: 125
  Average latency: 0.80 ms
  Total latency: 32.00 ms

Step 6: Saving enhanced audio...
Wrote WAV: output.wav
  Samples: 96000
  Duration: 2.00 seconds

Done!                                ← ✅ 成功！
```

---

## 总结

| 方面 | 修复前 | 修复后 |
|------|--------|--------|
| **Skip Connections** | ❌ 悬空指针 | ✅ 持久化内存 |
| **DPGRNN 缓存** | ❌ static 变量 | ✅ 实例缓存 |
| **多实例支持** | ❌ 不支持 | ✅ 支持 |
| **线程安全** | ❌ 不安全 | ✅ 实例级安全 |
| **段错误风险** | ❌ 高 | ✅ 无 |
| **内存泄漏** | ⚠️ 可能 | ✅ 无 |
| **可用性** | ❌ 不可用 | ✅ 可用 |

---

**关键要点**:
1. 必须使用 `gtcrn_streaming_optimized_FIXED.c`
2. 必须使用修改后的 `gtcrn_streaming.h` 和 `gtcrn_streaming.c`
3. 所有内存管理问题已修复
4. 现在可以安全使用实时降噪功能
