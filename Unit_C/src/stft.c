#include "stft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// FFT Implementation (Cooley-Tukey Radix-2)
// ============================================================================

/**
 * Bit-reversal permutation for FFT
 */
static void bit_reverse(float* real, float* imag, int n) {
    int j = 0;
    for (int i = 0; i < n - 1; i++) {
        if (i < j && j < n) {  // 添加边界检查
            // Swap real parts
            float temp = real[i];
            real[i] = real[j];
            real[j] = temp;

            // Swap imaginary parts
            temp = imag[i];
            imag[i] = imag[j];
            imag[j] = temp;
        }

        int k = n / 2;
        while (k > 0 && k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
        
        // 确保 j 不会超出范围
        if (j >= n) {
            j = n - 1;
        }
    }
}

/**
 * Check if n is a power of 2
 */
static int is_power_of_2(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/**
 * In-place FFT (Cooley-Tukey algorithm)
 * Input/Output: real[n], imag[n]
 * Note: n must be a power of 2 for this implementation
 */
static void fft_inplace(float* real, float* imag, int n) {
    // 验证 n 是 2 的幂
    if (!is_power_of_2(n)) {
        fprintf(stderr, "Error: FFT requires n to be a power of 2, but n=%d\n", n);
        return;
    }
    
    // Bit-reversal permutation
    bit_reverse(real, imag, n);

    // FFT computation
    for (int len = 2; len <= n; len *= 2) {
        float angle = -2.0f * M_PI / len;
        float wlen_real = cosf(angle);
        float wlen_imag = sinf(angle);

        for (int i = 0; i < n; i += len) {
            float w_real = 1.0f;
            float w_imag = 0.0f;

            for (int j = 0; j < len / 2; j++) {
                int idx1 = i + j;
                int idx2 = i + j + len / 2;

                // 边界检查，防止数组越界
                if (idx1 >= n || idx2 >= n || idx1 < 0 || idx2 < 0) {
                    fprintf(stderr, "Error: Array index out of bounds in FFT: idx1=%d, idx2=%d, n=%d, i=%d, j=%d, len=%d\n", 
                            idx1, idx2, n, i, j, len);
                    return;
                }

                float u_real = real[idx1];
                float u_imag = imag[idx1];

                float t_real = w_real * real[idx2] - w_imag * imag[idx2];
                float t_imag = w_real * imag[idx2] + w_imag * real[idx2];

                real[idx1] = u_real + t_real;
                imag[idx1] = u_imag + t_imag;

                real[idx2] = u_real - t_real;
                imag[idx2] = u_imag - t_imag;

                // Update twiddle factor
                float w_real_new = w_real * wlen_real - w_imag * wlen_imag;
                float w_imag_new = w_real * wlen_imag + w_imag * wlen_real;
                w_real = w_real_new;
                w_imag = w_imag_new;
            }
        }
    }
}

/**
 * In-place IFFT (Inverse FFT)
 */
static void ifft_inplace(float* real, float* imag, int n) {
    // Conjugate input
    for (int i = 0; i < n; i++) {
        imag[i] = -imag[i];
    }

    // Perform FFT
    fft_inplace(real, imag, n);

    // Conjugate output and scale
    float scale = 1.0f / n;
    for (int i = 0; i < n; i++) {
        real[i] *= scale;
        imag[i] *= -scale;
    }
}

// ============================================================================
// Window Functions
// ============================================================================

/**
 * Generate Hann window: w[n] = 0.5 * (1 - cos(2*pi*n/(N-1)))
 * For GTCRN, we use sqrt(Hann) for perfect reconstruction
 */
static void generate_hann_window(float* window, int length, int use_sqrt) {
    for (int i = 0; i < length; i++) {
        float w = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (length - 1)));
        window[i] = use_sqrt ? sqrtf(w) : w;
    }
}

// ============================================================================
// STFT Implementation
// ============================================================================

STFTParams* stft_create(int n_fft, int hop_length, int sample_rate) {
    STFTParams* params = (STFTParams*)malloc(sizeof(STFTParams));
    if (!params) return NULL;

    params->n_fft = n_fft;
    params->hop_length = hop_length;
    params->win_length = n_fft;
    params->sample_rate = sample_rate;

    // Allocate window (Hann^0.5 for perfect reconstruction)
    params->window = (float*)malloc(n_fft * sizeof(float));
    generate_hann_window(params->window, n_fft, 1);  // use_sqrt=1

    // Allocate FFT buffers
    params->fft_buffer = (float*)malloc(n_fft * 2 * sizeof(float));
    params->ifft_buffer = (float*)malloc(n_fft * 2 * sizeof(float));

    // Allocate overlap buffer for iSTFT
    params->overlap_size = n_fft;
    params->overlap_buffer = (float*)calloc(n_fft, sizeof(float));

    printf("STFT created: n_fft=%d, hop=%d, sr=%d\n", n_fft, hop_length, sample_rate);

    return params;
}

void stft_free(STFTParams* params) {
    if (params) {
        free(params->window);
        free(params->fft_buffer);
        free(params->ifft_buffer);
        free(params->overlap_buffer);
        free(params);
    }
}

int stft_num_frames(int num_samples, int n_fft, int hop_length) {
    // Number of frames = ceil((num_samples - n_fft) / hop_length) + 1
    if (num_samples < n_fft) return 0;
    return (num_samples - n_fft) / hop_length + 1;
}

int istft_num_samples(int num_frames, int hop_length) {
    return num_frames * hop_length;
}

int stft_forward(
    const float* audio,
    int num_samples,
    float* spec_real,
    float* spec_imag,
    STFTParams* params
) {
    int n_fft = params->n_fft;
    int hop_length = params->hop_length;
    int freq_bins = n_fft / 2 + 1;

    // 验证 n_fft 是 2 的幂（FFT 算法要求）
    if (!is_power_of_2(n_fft)) {
        fprintf(stderr, "Error: n_fft must be a power of 2, but n_fft=%d\n", n_fft);
        fprintf(stderr, "       Please use a power of 2 value (e.g., 512, 1024, 2048)\n");
        return 0;
    }

    // Calculate number of frames
    int num_frames = stft_num_frames(num_samples, n_fft, hop_length);
    if (num_frames <= 0) {
        fprintf(stderr, "Error: Audio too short for STFT\n");
        return 0;
    }

    // Allocate temporary buffers for FFT
    float* fft_real = (float*)malloc(n_fft * sizeof(float));
    float* fft_imag = (float*)malloc(n_fft * sizeof(float));
    
    if (!fft_real || !fft_imag) {
        fprintf(stderr, "Error: Failed to allocate memory for FFT buffers\n");
        if (fft_real) free(fft_real);
        if (fft_imag) free(fft_imag);
        return 0;
    }

    // Process each frame
    for (int frame = 0; frame < num_frames; frame++) {
        int start = frame * hop_length;

        // Extract and window the frame
        for (int i = 0; i < n_fft; i++) {
            int idx = start + i;
            if (idx < num_samples) {
                fft_real[i] = audio[idx] * params->window[i];
            } else {
                fft_real[i] = 0.0f;  // Zero padding
            }
            fft_imag[i] = 0.0f;
        }

        // Perform FFT
        fft_inplace(fft_real, fft_imag, n_fft);

        // Store only positive frequencies (0 to n_fft/2)
        for (int f = 0; f < freq_bins; f++) {
            int spec_idx = f * num_frames + frame;
            // 边界检查（虽然理论上不应该越界，但为了安全）
            if (spec_idx >= 0) {
                spec_real[spec_idx] = fft_real[f];
                spec_imag[spec_idx] = fft_imag[f];
            }
        }
    }

    free(fft_real);
    free(fft_imag);

    return num_frames;
}

int istft_forward(
    const float* spec_real,
    const float* spec_imag,
    int time_frames,
    float* audio,
    STFTParams* params
) {
    int n_fft = params->n_fft;
    int hop_length = params->hop_length;
    int freq_bins = n_fft / 2 + 1;

    // Calculate output length
    int num_samples = istft_num_samples(time_frames, hop_length);

    // Initialize output audio
    memset(audio, 0, num_samples * sizeof(float));

    // Reset overlap buffer
    memset(params->overlap_buffer, 0, params->overlap_size * sizeof(float));

    // Allocate temporary buffers for IFFT
    float* ifft_real = (float*)malloc(n_fft * sizeof(float));
    float* ifft_imag = (float*)malloc(n_fft * sizeof(float));
    float* frame_audio = (float*)malloc(n_fft * sizeof(float));

    // Process each frame
    for (int frame = 0; frame < time_frames; frame++) {
        // Load spectrum for this frame
        for (int f = 0; f < freq_bins; f++) {
            ifft_real[f] = spec_real[f * time_frames + frame];
            ifft_imag[f] = spec_imag[f * time_frames + frame];
        }

        // Mirror for negative frequencies (Hermitian symmetry)
        for (int f = freq_bins; f < n_fft; f++) {
            int mirror_idx = n_fft - f;
            ifft_real[f] = ifft_real[mirror_idx];
            ifft_imag[f] = -ifft_imag[mirror_idx];  // Conjugate
        }

        // Perform IFFT
        ifft_inplace(ifft_real, ifft_imag, n_fft);

        // Apply window
        for (int i = 0; i < n_fft; i++) {
            frame_audio[i] = ifft_real[i] * params->window[i];
        }

        // Overlap-add
        int start = frame * hop_length;
        for (int i = 0; i < n_fft && (start + i) < num_samples; i++) {
            audio[start + i] += frame_audio[i];
        }
    }

    free(ifft_real);
    free(ifft_imag);
    free(frame_audio);

    return num_samples;
}

// ============================================================================
// Streaming STFT (for real-time processing)
// ============================================================================

typedef struct {
    STFTParams* params;
    float* input_buffer;   // Circular buffer for input audio
    int buffer_size;
    int write_pos;
    int read_pos;
    int samples_available;
} StreamSTFT;

/**
 * Create streaming STFT processor
 */
StreamSTFT* stream_stft_create(int n_fft, int hop_length, int sample_rate) {
    StreamSTFT* stream = (StreamSTFT*)malloc(sizeof(StreamSTFT));
    if (!stream) return NULL;

    stream->params = stft_create(n_fft, hop_length, sample_rate);
    stream->buffer_size = n_fft * 2;  // Double buffer for safety
    stream->input_buffer = (float*)calloc(stream->buffer_size, sizeof(float));
    stream->write_pos = 0;
    stream->read_pos = 0;
    stream->samples_available = 0;

    return stream;
}

/**
 * Free streaming STFT processor
 */
void stream_stft_free(StreamSTFT* stream) {
    if (stream) {
        stft_free(stream->params);
        free(stream->input_buffer);
        free(stream);
    }
}

/**
 * Process one frame of audio in streaming mode
 * Returns 1 if a frame is ready, 0 otherwise
 */
int stream_stft_process_frame(
    StreamSTFT* stream,
    const float* input_chunk,
    int chunk_size,
    float* spec_real,
    float* spec_imag
) {
    // Add input to buffer
    for (int i = 0; i < chunk_size; i++) {
        stream->input_buffer[stream->write_pos] = input_chunk[i];
        stream->write_pos = (stream->write_pos + 1) % stream->buffer_size;
        stream->samples_available++;
    }

    // Check if we have enough samples for a frame
    int n_fft = stream->params->n_fft;
    int hop_length = stream->params->hop_length;

    if (stream->samples_available >= n_fft) {
        // Extract frame from circular buffer
        float* frame = (float*)malloc(n_fft * sizeof(float));
        for (int i = 0; i < n_fft; i++) {
            int idx = (stream->read_pos + i) % stream->buffer_size;
            frame[i] = stream->input_buffer[idx];
        }

        // Perform STFT on this frame
        float* fft_real = (float*)malloc(n_fft * sizeof(float));
        float* fft_imag = (float*)malloc(n_fft * sizeof(float));

        // Apply window
        for (int i = 0; i < n_fft; i++) {
            fft_real[i] = frame[i] * stream->params->window[i];
            fft_imag[i] = 0.0f;
        }

        // FFT
        fft_inplace(fft_real, fft_imag, n_fft);

        // Copy output (only positive frequencies)
        int freq_bins = n_fft / 2 + 1;
        memcpy(spec_real, fft_real, freq_bins * sizeof(float));
        memcpy(spec_imag, fft_imag, freq_bins * sizeof(float));

        // Advance read position
        stream->read_pos = (stream->read_pos + hop_length) % stream->buffer_size;
        stream->samples_available -= hop_length;

        free(frame);
        free(fft_real);
        free(fft_imag);

        return 1;  // Frame ready
    }

    return 0;  // Not enough samples yet
}
