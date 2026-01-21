/// <file>test_stft_istft_roundtrip.c</file>
/// <summary>Test STFT->ISTFT roundtrip</summary>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gtcrn_model.h"

#define N_FFT 512
#define WIN_LEN 512
#define HOP_LEN 256
#define N_FREQ 257

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    printf("=== STFT->ISTFT Roundtrip Test ===\n\n");

    /* Create STFT */
    gtcrn_stft_t* stft = gtcrn_stft_create(N_FFT, HOP_LEN, WIN_LEN);
    if (!stft) {
        fprintf(stderr, "Error: Failed to create STFT\n");
        return 1;
    }

    /* Create test signal */
    float* signal = (float*)malloc(N_FFT * sizeof(float));
    float* spec_real = (float*)malloc(N_FREQ * sizeof(float));
    float* spec_imag = (float*)malloc(N_FREQ * sizeof(float));
    float* reconstructed = (float*)malloc(N_FFT * sizeof(float));

    for (int i = 0; i < N_FFT; i++) {
        signal[i] = sin(2.0 * M_PI * 10.0 * i / N_FFT);
    }

    printf("Test 1: Direct FFT/IFFT with window (simulating STFT->ISTFT)\n");
    printf("------------------------------------------------------------\n");

    /* Compute original windowed signal's energy */
    double orig_windowed_energy = 0;
    for (int i = 0; i < WIN_LEN; i++) {
        double val = signal[i] * stft->window[i];
        orig_windowed_energy += val * val;
    }
    printf("Original windowed signal energy: %.6f\n", orig_windowed_energy);

    /* STFT: window -> FFT -> take positive freqs */
    gtcrn_stft_frame(stft, signal, spec_real, spec_imag);

    /* Check spectrum energy */
    double pos_spec_energy = 0;
    for (int i = 0; i < N_FREQ; i++) {
        pos_spec_energy += spec_real[i] * spec_real[i] + spec_imag[i] * spec_imag[i];
    }
    printf("Positive frequency spectrum energy: %.6f\n", pos_spec_energy);

    /* ISTFT: reconstruct full spectrum -> IFFT -> window */
    gtcrn_istft_frame(stft, spec_real, spec_imag, reconstructed);

    /* Check reconstructed energy */
    double recon_energy = 0;
    for (int i = 0; i < WIN_LEN; i++) {
        recon_energy += reconstructed[i] * reconstructed[i];
    }
    printf("Reconstructed signal energy: %.6f\n", recon_energy);

    /* Expected: recon = original * window^2 */
    /* So recon_energy should be sum(original^2 * window^4) */
    double expected_energy = 0;
    for (int i = 0; i < WIN_LEN; i++) {
        double val = signal[i] * stft->window[i] * stft->window[i];
        expected_energy += val * val;
    }
    printf("Expected energy (orig*window^4): %.6f\n", expected_energy);

    printf("\nEnergy ratios:\n");
    printf("  Recon/OrigWindowed: %.6f\n", recon_energy / orig_windowed_energy);
    printf("  Recon/Expected: %.6f\n", recon_energy / expected_energy);

    /* Sample comparison */
    printf("\n=== Sample comparison ===\n");
    printf("Idx | Orig*Win    | Orig*Win^2  | Reconstructed | Diff from Win^2\n");
    printf("----+-------------+-------------+---------------+----------------\n");
    for (int i = 0; i < 10; i++) {
        double orig_win = signal[i] * stft->window[i];
        double orig_win2 = signal[i] * stft->window[i] * stft->window[i];
        double diff = reconstructed[i] - orig_win2;
        printf("%3d | %11.6f | %11.6f | %13.6f | %.6f\n",
               i, orig_win, orig_win2, reconstructed[i], diff);
    }

    /* Now test: what if we manually reconstruct full spectrum correctly? */
    printf("\n\nTest 2: Manual full spectrum reconstruction\n");
    printf("-------------------------------------------\n");

    /* Manually build full spectrum from positive freqs */
    float* full_real = (float*)malloc(N_FFT * sizeof(float));
    float* full_imag = (float*)malloc(N_FFT * sizeof(float));

    /* Copy positive frequencies */
    for (int i = 0; i < N_FREQ; i++) {
        full_real[i] = spec_real[i];
        full_imag[i] = spec_imag[i];
    }

    /* Reconstruct negative frequencies (conjugate symmetric) */
    for (int i = 1; i < N_FFT / 2; i++) {
        full_real[N_FFT - i] = full_real[i];
        full_imag[N_FFT - i] = -full_imag[i];
    }

    /* Full spectrum energy */
    double full_spec_energy = 0;
    for (int i = 0; i < N_FFT; i++) {
        full_spec_energy += full_real[i] * full_real[i] + full_imag[i] * full_imag[i];
    }
    printf("Full spectrum energy: %.6f\n", full_spec_energy);

    /* Relationship: full = 2 * positive - DC^2 - Nyquist^2 */
    double dc_energy = spec_real[0] * spec_real[0] + spec_imag[0] * spec_imag[0];
    double nyq_energy = spec_real[N_FREQ-1] * spec_real[N_FREQ-1] + spec_imag[N_FREQ-1] * spec_imag[N_FREQ-1];
    double expected_full = 2 * pos_spec_energy - dc_energy - nyq_energy;
    printf("Expected full (2*pos - DC - Nyq): %.6f\n", expected_full);

    /* IFFT */
    gtcrn_fft_inverse(stft->fft_plan, full_real, full_imag);

    /* After IFFT, we have the original windowed signal */
    double ifft_energy = 0;
    for (int i = 0; i < N_FFT; i++) {
        ifft_energy += full_real[i] * full_real[i];
    }
    printf("After IFFT energy: %.6f\n", ifft_energy);
    printf("Ratio to orig_windowed: %.6f\n", ifft_energy / orig_windowed_energy);

    /* The IFFT output should match original * window (before second windowing) */
    printf("\n=== IFFT output vs original*window ===\n");
    printf("Idx | Orig*Win    | IFFT Output | Diff\n");
    printf("----+-------------+-------------+--------\n");
    for (int i = 0; i < 10; i++) {
        double orig_win = signal[i] * stft->window[i];
        printf("%3d | %11.6f | %11.6f | %.6f\n",
               i, orig_win, full_real[i], full_real[i] - orig_win);
    }

    /* Clean up */
    free(signal);
    free(spec_real);
    free(spec_imag);
    free(reconstructed);
    free(full_real);
    free(full_imag);
    gtcrn_stft_destroy(stft);

    return 0;
}
