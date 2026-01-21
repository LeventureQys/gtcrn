/// <file>wav_io.c</file>
/// <summary>简单WAV文件I/O实现</summary>
/// <author>李文轩</author>

#include "wav_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* WAV文件结构 */
#pragma pack(push, 1)
typedef struct {
    char riff[4];           /* "RIFF" */
    uint32_t file_size;     /* 文件大小 - 8 */
    char wave[4];           /* "WAVE" */
} wav_riff_header_t;

typedef struct {
    char id[4];             /* 块ID */
    uint32_t size;          /* 块大小 */
} wav_chunk_header_t;

typedef struct {
    uint16_t audio_format;  /* 1 = PCM, 3 = IEEE float */
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
} wav_fmt_chunk_t;
#pragma pack(pop)

int wav_read(const char* filepath, wav_info_t* info, float** data) {
    FILE* fp = fopen(filepath, "rb");
    if (!fp) {
        fprintf(stderr, "错误: 无法打开文件: %s\n", filepath);
        return -1;
    }

    /* 读取RIFF头 */
    wav_riff_header_t riff;
    if (fread(&riff, sizeof(riff), 1, fp) != 1) {
        fprintf(stderr, "错误: 无法读取RIFF头\n");
        fclose(fp);
        return -1;
    }

    if (memcmp(riff.riff, "RIFF", 4) != 0 || memcmp(riff.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "错误: 不是有效的WAV文件\n");
        fclose(fp);
        return -1;
    }

    /* 查找fmt和data块 */
    wav_fmt_chunk_t fmt;
    memset(&fmt, 0, sizeof(fmt));
    uint32_t data_size = 0;
    long data_offset = 0;

    while (1) {
        wav_chunk_header_t chunk;
        if (fread(&chunk, sizeof(chunk), 1, fp) != 1) break;

        if (memcmp(chunk.id, "fmt ", 4) == 0) {
            if (fread(&fmt, sizeof(fmt), 1, fp) != 1) {
                fclose(fp);
                return -1;
            }
            /* 跳过额外格式字节(如有) */
            if (chunk.size > sizeof(fmt)) {
                fseek(fp, chunk.size - sizeof(fmt), SEEK_CUR);
            }
        } else if (memcmp(chunk.id, "data", 4) == 0) {
            data_size = chunk.size;
            data_offset = ftell(fp);
            break;
        } else {
            /* 跳过未知块 */
            fseek(fp, chunk.size, SEEK_CUR);
        }
    }

    if (data_size == 0 || fmt.sample_rate == 0) {
        fprintf(stderr, "错误: 无效的WAV格式\n");
        fclose(fp);
        return -1;
    }

    /* 填充信息 */
    info->sample_rate = fmt.sample_rate;
    info->num_channels = fmt.num_channels;
    info->bits_per_sample = fmt.bits_per_sample;
    info->num_samples = data_size / (fmt.bits_per_sample / 8) / fmt.num_channels;

    /* 分配输出缓冲区 */
    *data = (float*)malloc(info->num_samples * sizeof(float));
    if (!*data) {
        fclose(fp);
        return -1;
    }

    /* 读取音频数据 */
    fseek(fp, data_offset, SEEK_SET);

    if (fmt.audio_format == 1) {
        /* PCM整数 */
        if (fmt.bits_per_sample == 16) {
            int16_t* raw = (int16_t*)malloc(info->num_samples * fmt.num_channels * sizeof(int16_t));
            if (!raw) {
                free(*data);
                fclose(fp);
                return -1;
            }
            fread(raw, sizeof(int16_t), info->num_samples * fmt.num_channels, fp);

            /* 转换为浮点数,如果是立体声则取第一通道 */
            for (uint32_t i = 0; i < info->num_samples; i++) {
                (*data)[i] = raw[i * fmt.num_channels] / 32768.0f;
            }
            free(raw);
        } else if (fmt.bits_per_sample == 32) {
            int32_t* raw = (int32_t*)malloc(info->num_samples * fmt.num_channels * sizeof(int32_t));
            if (!raw) {
                free(*data);
                fclose(fp);
                return -1;
            }
            fread(raw, sizeof(int32_t), info->num_samples * fmt.num_channels, fp);

            for (uint32_t i = 0; i < info->num_samples; i++) {
                (*data)[i] = raw[i * fmt.num_channels] / 2147483648.0f;
            }
            free(raw);
        } else {
            fprintf(stderr, "错误: 不支持的位深度: %d\n", fmt.bits_per_sample);
            free(*data);
            fclose(fp);
            return -1;
        }
    } else if (fmt.audio_format == 3) {
        /* IEEE浮点 */
        float* raw = (float*)malloc(info->num_samples * fmt.num_channels * sizeof(float));
        if (!raw) {
            free(*data);
            fclose(fp);
            return -1;
        }
        fread(raw, sizeof(float), info->num_samples * fmt.num_channels, fp);

        for (uint32_t i = 0; i < info->num_samples; i++) {
            (*data)[i] = raw[i * fmt.num_channels];
        }
        free(raw);
    } else {
        fprintf(stderr, "错误: 不支持的音频格式: %d\n", fmt.audio_format);
        free(*data);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return (int)info->num_samples;
}

int wav_write(const char* filepath, const float* data, int num_samples, int sample_rate) {
    FILE* fp = fopen(filepath, "wb");
    if (!fp) {
        fprintf(stderr, "错误: 无法创建文件: %s\n", filepath);
        return -1;
    }

    /* 准备头 */
    uint32_t data_size = num_samples * sizeof(int16_t);
    uint32_t file_size = 36 + data_size;

    wav_riff_header_t riff = {
        .riff = {'R', 'I', 'F', 'F'},
        .file_size = file_size,
        .wave = {'W', 'A', 'V', 'E'}
    };

    wav_chunk_header_t fmt_header = {
        .id = {'f', 'm', 't', ' '},
        .size = 16
    };

    wav_fmt_chunk_t fmt = {
        .audio_format = 1,  /* PCM */
        .num_channels = 1,
        .sample_rate = (uint32_t)sample_rate,
        .byte_rate = (uint32_t)(sample_rate * 2),
        .block_align = 2,
        .bits_per_sample = 16
    };

    wav_chunk_header_t data_header = {
        .id = {'d', 'a', 't', 'a'},
        .size = data_size
    };

    /* 写入头 */
    fwrite(&riff, sizeof(riff), 1, fp);
    fwrite(&fmt_header, sizeof(fmt_header), 1, fp);
    fwrite(&fmt, sizeof(fmt), 1, fp);
    fwrite(&data_header, sizeof(data_header), 1, fp);

    /* 转换并写入采样 */
    int16_t* raw = (int16_t*)malloc(num_samples * sizeof(int16_t));
    if (!raw) {
        fclose(fp);
        return -1;
    }

    for (int i = 0; i < num_samples; i++) {
        float sample = data[i];
        /* 限幅 */
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        raw[i] = (int16_t)(sample * 32767.0f);
    }

    fwrite(raw, sizeof(int16_t), num_samples, fp);
    free(raw);
    fclose(fp);

    return 0;
}
