/// <file>gtcrn_fft.h</file>
/// <summary>GTCRN FFT/STFT实现</summary>
/// <author>江月希 李文轩</author>
/// <remarks>音频处理的纯C语言FFT和STFT实现</remarks>

#ifndef GTCRN_FFT_H
#define GTCRN_FFT_H

#include "gtcrn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// <summary>FFT计划结构体</summary>
typedef struct {
    int n;                      /* FFT大小 */
    gtcrn_float* cos_table;     /* 余弦查找表 */
    gtcrn_float* sin_table;     /* 正弦查找表 */
    int* bit_reverse;           /* 位反转索引 */
} gtcrn_fft_plan_t;

/// <summary>STFT配置</summary>
typedef struct {
    int n_fft;                  /* FFT大小 */
    int hop_length;             /* 跳跃长度 */
    int win_length;             /* 窗长度 */
    gtcrn_float* window;        /* 窗函数(平方根汉宁窗) */
    gtcrn_fft_plan_t* fft_plan; /* FFT计划 */
    gtcrn_float* fft_buffer;    /* FFT临时缓冲区 */
} gtcrn_stft_t;

/// <summary>创建FFT计划</summary>
/// <remarks>为指定大小的FFT分配并初始化所有必要的查找表和索引。FFT计划可以重复使用以提高计算效率。n必须是2的幂,否则返回NULL。使用完毕后必须调用gtcrn_fft_plan_destroy()释放。同一个计划可以用于多次FFT计算。</remarks>
/// <param name="n">FFT大小,必须是2的幂(如256, 512, 1024等)</param>
/// <returns>成功返回FFT计划指针,失败返回NULL(参数无效或内存分配失败)</returns>
gtcrn_fft_plan_t* gtcrn_fft_plan_create(int n);

/// <summary>销毁FFT计划</summary>
/// <remarks>释放FFT计划占用的所有内存,包括查找表和索引数组。重复调用是安全的(传入NULL)。销毁后计划指针无效,不应再次使用。</remarks>
/// <param name="plan">FFT计划指针,可为NULL(安全处理)</param>
void gtcrn_fft_plan_destroy(gtcrn_fft_plan_t* plan);

/// <summary>计算正向FFT(原地操作)</summary>
/// <remarks>对输入的时域信号进行快速傅里叶变换,将结果直接写回输入缓冲区。输入为实部和虚部数组,输出为频域复数表示。这是原地操作,输入数据会被覆盖。real和imag数组长度必须等于plan->n。对于实数输入,通常imag初始化为全零。</remarks>
/// <param name="plan">FFT计划,必须已通过gtcrn_fft_plan_create()创建</param>
/// <param name="real">实部数组(输入/输出),长度必须为plan->n</param>
/// <param name="imag">虚部数组(输入/输出),长度必须为plan->n</param>
void gtcrn_fft_forward(gtcrn_fft_plan_t* plan,
                       gtcrn_float* real, gtcrn_float* imag);

/// <summary>计算逆FFT(原地操作)</summary>
/// <remarks>对输入的频域信号进行逆快速傅里叶变换,将结果直接写回输入缓冲区。输入为频域复数,输出为时域信号。这是原地操作,输入数据会被覆盖。输出结果需要除以n才能得到正确的幅度。real和imag数组长度必须等于plan->n。</remarks>
/// <param name="plan">FFT计划,必须已通过gtcrn_fft_plan_create()创建</param>
/// <param name="real">实部数组(输入/输出),长度必须为plan->n</param>
/// <param name="imag">虚部数组(输入/输出),长度必须为plan->n</param>
void gtcrn_fft_inverse(gtcrn_fft_plan_t* plan,
                       gtcrn_float* real, gtcrn_float* imag);

/// <summary>创建STFT处理器</summary>
/// <remarks>分配并初始化短时傅里叶变换处理器,包括窗函数、FFT计划和临时缓冲区。STFT用于将时域信号转换为时频域表示。n_fft必须是2的幂。通常hop_length = n_fft/2,win_length = n_fft。使用完毕后必须调用gtcrn_stft_destroy()释放。</remarks>
/// <param name="n_fft">FFT大小,必须是2的幂,通常为512</param>
/// <param name="hop_length">帧之间的跳跃长度(采样点数),通常为n_fft/2(256)</param>
/// <param name="win_length">窗函数长度(采样点数),通常等于n_fft</param>
/// <returns>成功返回STFT处理器指针,失败返回NULL(参数无效或内存分配失败)</returns>
gtcrn_stft_t* gtcrn_stft_create(int n_fft, int hop_length, int win_length);

/// <summary>销毁STFT处理器</summary>
/// <remarks>释放STFT处理器占用的所有内存,包括窗函数、FFT计划和缓冲区。重复调用是安全的(传入NULL)。销毁后处理器指针无效,不应再次使用。</remarks>
/// <param name="stft">STFT处理器指针,可为NULL(安全处理)</param>
void gtcrn_stft_destroy(gtcrn_stft_t* stft);

/// <summary>计算整个信号的STFT</summary>
/// <remarks>对完整的时域信号进行短时傅里叶变换,将信号分解为时频域表示。输出为二维数组,每行代表一帧的频域数据。输出数组大小: n_frames = gtcrn_stft_num_frames(signal_len, n_fft, hop_length)。每帧输出n_fft/2+1个频率bin(对称性只保留一半)。输出按行主序存储,访问第i帧第j个频率: spec_real[i * (n_fft/2+1) + j]。</remarks>
/// <param name="stft">STFT处理器,必须已通过gtcrn_stft_create()创建</param>
/// <param name="signal">输入音频信号数组,长度为signal_len</param>
/// <param name="signal_len">信号长度(采样点数),必须大于0</param>
/// <param name="spec_real">输出实部数组,必须预分配n_frames * (n_fft/2+1)个元素,按行主序存储: [frame0_freq0, frame0_freq1, ..., frame1_freq0, ...]</param>
/// <param name="spec_imag">输出虚部数组,大小与spec_real相同</param>
/// <returns>成功返回处理的帧数(n_frames),失败返回-1(参数无效)</returns>
int gtcrn_stft_forward(gtcrn_stft_t* stft,
                       const gtcrn_float* signal, int signal_len,
                       gtcrn_float* spec_real, gtcrn_float* spec_imag);

/// <summary>计算单帧STFT</summary>
/// <remarks>对单个音频帧进行短时傅里叶变换,用于流式处理场景。输入为一帧的时域数据,输出为该帧的频域表示。frame数组长度必须等于stft->win_length。输出包含n_fft/2+1个频率bin。适合流式处理,每次处理一帧。</remarks>
/// <param name="stft">STFT处理器,必须已通过gtcrn_stft_create()创建</param>
/// <param name="frame">输入帧数组,必须包含win_length个采样点</param>
/// <param name="spec_real">输出实部数组,必须预分配n_fft/2+1个元素</param>
/// <param name="spec_imag">输出虚部数组,必须预分配n_fft/2+1个元素</param>
void gtcrn_stft_frame(gtcrn_stft_t* stft,
                      const gtcrn_float* frame,
                      gtcrn_float* spec_real, gtcrn_float* spec_imag);

/// <summary>计算逆STFT(完整信号)</summary>
/// <remarks>将时频域表示转换回时域信号,使用重叠相加方法重建完整信号。输出长度可能因帧对齐与原始信号略有差异。使用重叠相加方法重建,确保时域连续性。stft参数必须与正向STFT时一致。</remarks>
/// <param name="stft">STFT处理器,必须与STFT时使用相同的参数</param>
/// <param name="spec_real">输入实部数组,形状为(n_frames, n_fft/2+1),按行主序</param>
/// <param name="spec_imag">输入虚部数组,形状与spec_real相同</param>
/// <param name="n_frames">帧数,必须与STFT时的帧数一致</param>
/// <param name="signal">输出信号数组,必须预分配足够空间</param>
/// <param name="signal_len">输出参数,函数返回后存储实际信号长度</param>
void gtcrn_istft(gtcrn_stft_t* stft,
                 const gtcrn_float* spec_real, const gtcrn_float* spec_imag,
                 int n_frames,
                 gtcrn_float* signal, int signal_len);

/// <summary>单帧逆STFT(重叠相加)</summary>
/// <remarks>将单帧的频域表示转换回时域,输出到重叠相加缓冲区。用于流式处理场景,需要外部维护重叠相加缓冲区。output缓冲区需要外部初始化为零或保留上次的结果。输出长度为n_fft,但只有前hop_length个采样点是新的。适合流式处理,需要外部管理重叠相加逻辑。</remarks>
/// <param name="stft">STFT处理器,必须与STFT时使用相同的参数</param>
/// <param name="spec_real">输入实部数组,长度为n_fft/2+1</param>
/// <param name="spec_imag">输入虚部数组,长度为n_fft/2+1</param>
/// <param name="output">输出缓冲区,长度至少为n_fft个采样点,函数会将重建的时域帧累加到此缓冲区(重叠相加)</param>
void gtcrn_istft_frame(gtcrn_stft_t* stft,
                       const gtcrn_float* spec_real, const gtcrn_float* spec_imag,
                       gtcrn_float* output);

/// <summary>计算给定信号长度的STFT帧数</summary>
/// <remarks>根据信号长度、FFT大小和跳跃长度计算STFT会产生多少帧。用于预先分配输出缓冲区。计算公式: n_frames = ceil((signal_len - n_fft) / hop_length) + 1。如果signal_len < n_fft,返回1。</remarks>
/// <param name="signal_len">信号长度(采样点数)</param>
/// <param name="n_fft">FFT大小</param>
/// <param name="hop_length">跳跃长度(采样点数)</param>
/// <returns>计算的帧数</returns>
int gtcrn_stft_num_frames(int signal_len, int n_fft, int hop_length);

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_FFT_H */
