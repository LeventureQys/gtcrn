#ifndef STFT_H
#define STFT_H

#include <stdint.h>

/**
 * STFT (Short-Time Fourier Transform) for GTCRN
 *
 * Configuration for 48kHz audio:
 * - FFT size: 1536
 * - Hop length: 768
 * - Window: Hann^0.5
 * - Output: (freq_bins=769, time_frames, 2) complex spectrum
 */

typedef struct {
    int n_fft;          // FFT size (1536)
    int hop_length;     // Hop length (768)
    int win_length;     // Window length (1536)
    int sample_rate;    // Sample rate (48000)

    float* window;      // Window function (Hann^0.5)
    float* fft_buffer;  // FFT working buffer
    float* ifft_buffer; // iFFT working buffer

    // For overlap-add in iSTFT
    float* overlap_buffer;
    int overlap_size;
} STFTParams;

/**
 * Create STFT parameters
 *
 * @param n_fft         FFT size (default: 1536)
 * @param hop_length    Hop length (default: 768)
 * @param sample_rate   Sample rate (default: 48000)
 * @return              STFT parameters structure
 */
STFTParams* stft_create(int n_fft, int hop_length, int sample_rate);

/**
 * Free STFT parameters
 */
void stft_free(STFTParams* params);

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
int stft_forward(
    const float* audio,
    int num_samples,
    float* spec_real,
    float* spec_imag,
    STFTParams* params
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
int istft_forward(
    const float* spec_real,
    const float* spec_imag,
    int time_frames,
    float* audio,
    STFTParams* params
);

/**
 * Compute number of frames from audio length
 */
int stft_num_frames(int num_samples, int n_fft, int hop_length);

/**
 * Compute audio length from number of frames
 */
int istft_num_samples(int num_frames, int hop_length);

#endif // STFT_H
