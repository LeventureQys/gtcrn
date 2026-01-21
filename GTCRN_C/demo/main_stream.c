/// <file>main_stream.c</file>
/// <summary>GTCRN流式演示应用</summary>
/// <author>李文轩</author>
/// <remarks>使用GTCRN C库进行实时流式语音增强演示。逐帧处理音频(每帧256个采样点/16ms)。用法: gtcrn_stream_demo &lt;weights_file&gt; &lt;input_wav&gt; &lt;output_wav&gt;</remarks>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gtcrn_model.h"
#include "wav_io.h"

/* 帧大小: 256个采样点 = 16kHz时16ms */
#define FRAME_SIZE 256

static void print_usage(const char* program) {
    printf("GTCRN流式语音增强演示\n\n");
    printf("用法: %s <weights_file> <input_wav> <output_wav>\n\n", program);
    printf("参数:\n");
    printf("  weights_file  二进制权重文件路径(.bin)\n");
    printf("  input_wav     输入含噪WAV文件路径(16kHz单声道)\n");
    printf("  output_wav    输出增强WAV文件路径\n\n");
    printf("特性:\n");
    printf("  - 逐帧处理(每帧256个采样点/16ms)\n");
    printf("  - 适用于实时应用\n");
    printf("  - 低延迟: 单帧延迟\n\n");
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

    printf("=== GTCRN流式语音增强演示 ===\n\n");

    /* 步骤1: 创建模型 */
    printf("[1/6] 创建GTCRN模型...\n");
    gtcrn_t* model = gtcrn_create();
    if (!model) {
        fprintf(stderr, "错误: 创建GTCRN模型失败\n");
        return 1;
    }
    printf("  模型创建成功\n");
    printf("  工作空间大小: %.2f KB\n\n", gtcrn_get_workspace_size() / 1024.0);

    /* 步骤2: 加载权重 */
    printf("[2/6] 从 %s 加载权重...\n", weights_path);
    gtcrn_status_t status = gtcrn_load_weights(model, weights_path);
    if (status != GTCRN_OK) {
        fprintf(stderr, "错误: 加载权重失败(status=%d)\n", status);
        gtcrn_destroy(model);
        return 1;
    }
    printf("  权重加载成功\n\n");

    /* 步骤3: 读取输入WAV */
    printf("[3/6] 读取输入WAV: %s...\n", input_path);
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

    /* 步骤4: 准备流式处理 */
    printf("[4/6] 准备流式推理...\n");

    /* 计算完整帧数 */
    int num_frames = num_samples / FRAME_SIZE;
    int processed_samples = num_frames * FRAME_SIZE;

    printf("  帧大小: %d 采样点(%.1f ms)\n", FRAME_SIZE, FRAME_SIZE * 1000.0 / 16000.0);
    printf("  总帧数: %d\n", num_frames);
    printf("  处理采样点: %d / %d\n\n", processed_samples, num_samples);

    /* 分配输出缓冲区 */
    float* audio_out = (float*)malloc(processed_samples * sizeof(float));
    if (!audio_out) {
        fprintf(stderr, "错误: 内存分配失败\n");
        free(audio_in);
        gtcrn_destroy(model);
        return 1;
    }

    /* 流式处理前重置模型状态 */
    gtcrn_reset_state(model);

    /* 步骤5: 逐帧处理音频 */
    printf("[5/6] 处理音频(流式模式)...\n");

    double start_time = get_time_ms();
    double total_frame_time = 0.0;
    double max_frame_time = 0.0;
    double min_frame_time = 1e9;

    for (int i = 0; i < num_frames; i++) {
        const float* input_frame = audio_in + i * FRAME_SIZE;
        float* output_frame = audio_out + i * FRAME_SIZE;

        double frame_start = get_time_ms();

        status = gtcrn_process_frame(model, input_frame, output_frame);

        double frame_end = get_time_ms();
        double frame_time = frame_end - frame_start;

        if (status != GTCRN_OK) {
            fprintf(stderr, "错误: 帧 %d 处理失败(status=%d)\n", i, status);
            free(audio_in);
            free(audio_out);
            gtcrn_destroy(model);
            return 1;
        }

        total_frame_time += frame_time;
        if (frame_time > max_frame_time) max_frame_time = frame_time;
        if (frame_time < min_frame_time) min_frame_time = frame_time;

        /* 每100帧显示进度 */
        if ((i + 1) % 100 == 0 || i == num_frames - 1) {
            printf("\r  进度: %d / %d 帧(%.1f%%)",
                   i + 1, num_frames, (i + 1) * 100.0 / num_frames);
            fflush(stdout);
        }
    }
    printf("\n\n");

    double end_time = get_time_ms();
    double total_time = end_time - start_time;

    /* 计算统计信息 */
    double audio_duration_ms = (double)processed_samples / wav_info.sample_rate * 1000.0;
    double frame_duration_ms = FRAME_SIZE * 1000.0 / 16000.0;  /* 16ms */
    double avg_frame_time = total_frame_time / num_frames;
    double rtf = total_time / audio_duration_ms;

    printf("  流式统计:\n");
    printf("  ---------------------\n");
    printf("  总处理时间: %.2f ms\n", total_time);
    printf("  音频时长: %.2f ms\n", audio_duration_ms);
    printf("  帧时长: %.2f ms\n", frame_duration_ms);
    printf("  平均帧时间: %.3f ms\n", avg_frame_time);
    printf("  最小帧时间: %.3f ms\n", min_frame_time);
    printf("  最大帧时间: %.3f ms\n", max_frame_time);
    printf("  实时因子(RTF): %.4f\n", rtf);

    if (avg_frame_time < frame_duration_ms) {
        printf("  状态: 可实时处理\n");
        printf("  余量: 每帧%.2f ms(%.1f%%余量)\n\n",
               frame_duration_ms - avg_frame_time,
               (1.0 - avg_frame_time / frame_duration_ms) * 100.0);
    } else {
        printf("  状态: 非实时\n");
        printf("  超时: 每帧%.2f ms\n\n", avg_frame_time - frame_duration_ms);
    }

    /* 步骤6: 写入输出WAV */
    printf("[6/6] 写入输出WAV: %s...\n", output_path);
    if (wav_write(output_path, audio_out, processed_samples, wav_info.sample_rate) != 0) {
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

    printf("=== 流式增强完成 ===\n");
    printf("输入:  %s\n", input_path);
    printf("输出: %s\n", output_path);
    printf("模式:   流式(逐帧)\n");

    return 0;
}
