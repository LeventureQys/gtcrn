/// <file>main.c</file>
/// <summary>GTCRN演示应用</summary>
/// <author>李文轩</author>
/// <remarks>使用GTCRN C库进行语音增强演示。用法: gtcrn_demo &lt;weights_file&gt; &lt;input_wav&gt; &lt;output_wav&gt;</remarks>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gtcrn_model.h"
#include "wav_io.h"

static void print_usage(const char* program) {
    printf("GTCRN语音增强演示\n\n");
    printf("用法: %s <weights_file> <input_wav> <output_wav>\n\n", program);
    printf("参数:\n");
    printf("  weights_file  二进制权重文件路径(.bin)\n");
    printf("  input_wav     输入含噪WAV文件路径(16kHz单声道)\n");
    printf("  output_wav    输出增强WAV文件路径\n\n");
    printf("示例:\n");
    printf("  %s weights/gtcrn_weights.bin noisy.wav enhanced.wav\n", program);
}

static double get_time_ms(void) {
#ifdef _WIN32
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
#endif
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    /* Set console output to UTF-8 for Chinese characters */
    SetConsoleOutputCP(65001);
#endif
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char* weights_path = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];

    printf("=== GTCRN语音增强演示 ===\n\n");

    /* 步骤1: 创建模型 */
    printf("[1/5] 创建GTCRN模型...\n");
    gtcrn_t* model = gtcrn_create();
    if (!model) {
        fprintf(stderr, "错误: 创建GTCRN模型失败\n");
        return 1;
    }
    printf("  模型创建成功\n");
    printf("  工作空间大小: %.2f KB\n\n", gtcrn_get_workspace_size() / 1024.0);

    /* 步骤2: 加载权重 */
    printf("[2/5] 从 %s 加载权重...\n", weights_path);
    gtcrn_status_t status = gtcrn_load_weights(model, weights_path);
    if (status != GTCRN_OK) {
        fprintf(stderr, "错误: 加载权重失败(status=%d)\n", status);
        gtcrn_destroy(model);
        return 1;
    }
    printf("  权重加载成功\n\n");

    /* 步骤3: 读取输入WAV */
    printf("[3/5] 读取输入WAV: %s...\n", input_path);
    wav_info_t wav_info;
    float* audio_in = NULL;
    int num_samples = wav_read(input_path, &wav_info, &audio_in);
    if (num_samples <= 0) {
        fprintf(stderr, "错误: 读取输入WAV文件失败\n");
        gtcrn_destroy(model);
        return 1;
    }
    printf("  采样率: %d Hz\n", wav_info.sample_rate);
    printf("  通道数: %d\n", wav_info.num_channels);
    printf("  时长: %.2f 秒 (%d 采样点)\n\n",
           (float)num_samples / wav_info.sample_rate, num_samples);

    if (wav_info.sample_rate != 16000) {
        fprintf(stderr, "警告: 输入采样率为 %d Hz,期望16000 Hz\n",
                wav_info.sample_rate);
    }

    /* 步骤4: 处理音频 */
    printf("[4/5] 处理音频...\n");
    float* audio_out = (float*)malloc(num_samples * sizeof(float));
    if (!audio_out) {
        fprintf(stderr, "错误: 内存分配失败\n");
        free(audio_in);
        gtcrn_destroy(model);
        return 1;
    }

    double start_time = get_time_ms();

    int output_len = 0;
    status = gtcrn_process(model, audio_in, num_samples, audio_out, &output_len);

    double end_time = get_time_ms();
    double processing_time = end_time - start_time;

    if (status != GTCRN_OK) {
        fprintf(stderr, "错误: 处理失败(status=%d)\n", status);
        free(audio_in);
        free(audio_out);
        gtcrn_destroy(model);
        return 1;
    }

    double audio_duration_ms = (double)num_samples / wav_info.sample_rate * 1000.0;
    double rtf = processing_time / audio_duration_ms;

    printf("  处理时间: %.2f ms\n", processing_time);
    printf("  音频时长: %.2f ms\n", audio_duration_ms);
    printf("  实时因子(RTF): %.4f\n", rtf);
    if (rtf < 1.0) {
        printf("  状态: 实时(比实时快%.1fx)\n\n", 1.0 / rtf);
    } else {
        printf("  状态: 非实时(比实时慢%.1fx)\n\n", rtf);
    }

    /* 步骤5: 写入输出WAV */
    printf("[5/5] 写入输出WAV: %s...\n", output_path);
    if (wav_write(output_path, audio_out, output_len, wav_info.sample_rate) != 0) {
        fprintf(stderr, "错误: 写入输出WAV文件失败\n");
        free(audio_in);
        free(audio_out);
        gtcrn_destroy(model);
        return 1;
    }
    printf("  输出写入成功\n\n");

    /* 清理 */
    free(audio_in);
    free(audio_out);
    gtcrn_destroy(model);

    printf("=== 增强完成 ===\n");
    printf("输入:  %s\n", input_path);
    printf("输出: %s\n", output_path);

    return 0;
}
