#ifndef STFT_16K_H
#define STFT_16K_H

#include <stdint.h>

/**
 * STFT (Short-Time Fourier Transform) for GTCRN - 16kHz Version
 *
 * Configuration for 16kHz audio:
 * - FFT size: 512 (was 1536 for 48kHz)
 * - Hop length: 256 (was 768 for 48kHz)
 * - Window: Hann^0.5
 * - Output: (freq_bins=257, time_frames, 2) complex spectrum
 */

typedef struct {
    int n_fft;          // FFT size (512)
    int hop_length;     // Hop length (256)
    int win_length;     // Window length (512)
    int sample_rate;    // Sample rate (16000)

    float* window;      // Window function (Hann^0.5)
    float* fft_buffer;  // FFT working buffer
    float* ifft_buffer; // iFFT working buffer

    // For overlap-add in iSTFT
    float* overlap_buffer;
    int overlap_size;
} STFTParams_16k;

/**
 * Create STFT parameters for 16kHz
 *
 * @param n_fft         FFT size (default: 512)
 * @param hop_length    Hop length (default: 256)
 * @param sample_rate   Sample rate (default: 16000)
 * @return              STFT parameters structure
 */
STFTParams_16k* stft_16k_create(int n_fft, int hop_length, int sample_rate);

/**
 * Free STFT parameters
 */
void stft_16k_free(STFTParams_16k* params);

/**
 * Perform STFT on audio signal
 *
 * @param audio         Input audio signal (num_samples,)
 * @param num_samples   Number of audio samples
 * @param spec_real     Output real part (freq_bins, time_frames)
 * @param spec_imag     Output imaginary part (freq_bins, time_frames)
 * @param params        STFT parameters
 * @return              Number of time frames
 */
int stft_16k_forward(
    const float* audio,
    int num_samples,
    float* spec_real,
    float* spec_imag,
    STFTParams_16k* params
);

/**
 * Perform inverse STFT (iSTFT)
 *
 * @param spec_real     Input real part (freq_bins, time_frames)
 * @param spec_imag     Input imaginary part (freq_bins, time_frames)
 * @param time_frames   Number of time frames
 * @param audio         Output audio signal (num_samples,)
 * @param params        STFT parameters
 * @return              Number of output samples
 */
int istft_16k_forward(
    const float* spec_real,
    const float* spec_imag,
    int time_frames,
    float* audio,
    STFTParams_16k* params
);

/**
 * Compute number of frames from audio length
 */
int stft_16k_num_frames(int num_samples, int n_fft, int hop_length);

/**
 * Compute audio length from number of frames
 */
int istft_16k_num_samples(int num_frames, int hop_length);

#endif // STFT_16K_H
