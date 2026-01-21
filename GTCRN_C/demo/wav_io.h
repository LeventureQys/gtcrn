/// <file>wav_io.h</file>
/// <summary>GTCRN演示的简单WAV文件I/O</summary>
/// <author>李文轩</author>

#ifndef WAV_IO_H
#define WAV_IO_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// <summary>WAV文件头结构</summary>
typedef struct {
    uint32_t sample_rate;
    uint16_t num_channels;
    uint16_t bits_per_sample;
    uint32_t num_samples;
} wav_info_t;

/// <summary>读取WAV文件</summary>
/// <remarks>从WAV文件中读取音频数据并转换为浮点格式。支持PCM 16位/32位整数和IEEE 32位浮点格式。如果是立体声,只读取第一个通道。函数内部会分配内存,调用者必须调用free(*data)释放。如果是立体声,只返回第一个通道的数据。输出数据为浮点格式,PCM整数会自动归一化到[-1.0, 1.0]。</remarks>
/// <param name="filepath">WAV文件的完整路径,文件必须存在且格式正确</param>
/// <param name="info">输出参数,函数返回后包含WAV文件信息(采样率、通道数等)</param>
/// <param name="data">输出参数,函数会分配内存并返回音频数据指针,调用者必须使用free()释放此内存,数据格式为32位浮点,范围通常在[-1.0, 1.0]</param>
/// <returns>成功返回读取的采样点数(单声道),失败返回-1(文件不存在/无法打开/不是有效的WAV文件格式/不支持的音频格式/不支持的位深度/内存分配失败)</returns>
int wav_read(const char* filepath, wav_info_t* info, float** data);

/// <summary>写入WAV文件</summary>
/// <remarks>将浮点音频数据写入WAV文件,自动转换为16位PCM格式。输出为单声道、16位PCM格式。输出格式固定为: 16位PCM、单声道。输入数据会被限幅到[-1.0, 1.0]范围。浮点值会转换为16位整数: int16_t(value * 32767.0)。</remarks>
/// <param name="filepath">输出WAV文件的完整路径,如果文件已存在会被覆盖</param>
/// <param name="data">输入音频数据数组,长度为num_samples,数据应为32位浮点,范围建议在[-1.0, 1.0],超出范围的值会被限幅到[-1.0, 1.0]</param>
/// <param name="num_samples">采样点数,必须大于0</param>
/// <param name="sample_rate">采样率(Hz),通常为16000或44100</param>
/// <returns>成功返回0,失败返回-1(无法创建输出文件/路径无效或权限不足/内存分配失败/写入失败)</returns>
int wav_write(const char* filepath, const float* data, int num_samples, int sample_rate);

#ifdef __cplusplus
}
#endif

#endif /* WAV_IO_H */
