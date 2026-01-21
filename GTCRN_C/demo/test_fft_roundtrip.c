/// <file>test_fft_roundtrip.c</file>
/// <summary>Test FFT roundtrip (FFT->IFFT) to verify correctness</summary>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "gtcrn_model.h"

#define N_FFT 512

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
#endif

    printf("=== FFT Roundtrip Test ===\n\n");

    /* Create FFT plan */
    gtcrn_fft_plan_t* plan = gtcrn_fft_plan_create(N_FFT);
    if (!plan) {
        fprintf(stderr, "Error: Failed to create FFT plan\n");
        return 1;
    }

    /* Create test signal (simple sinusoid) */
    float* real = (float*)malloc(N_FFT * sizeof(float));
    float* imag = (float*)malloc(N_FFT * sizeof(float));
    float* original = (float*)malloc(N_FFT * sizeof(float));

    for (int i = 0; i < N_FFT; i++) {
        original[i] = sin(2.0 * M_PI * 10.0 * i / N_FFT);  /* 10 cycles */
        real[i] = original[i];
        imag[i] = 0.0f;
    }

    printf("Original signal:\n");
    double orig_energy = 0;
    for (int i = 0; i < N_FFT; i++) {
        orig_energy += original[i] * original[i];
    }
    printf("  Energy: %.6f\n", orig_energy);
    printf("  First 5: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           original[0], original[1], original[2], original[3], original[4]);

    /* Forward FFT */
    gtcrn_fft_forward(plan, real, imag);

    printf("\nAfter FFT:\n");
    double spec_energy = 0;
    for (int i = 0; i < N_FFT; i++) {
        spec_energy += real[i] * real[i] + imag[i] * imag[i];
    }
    printf("  Spectrum energy: %.6f\n", spec_energy);
    printf("  First 5 real: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           real[0], real[1], real[2], real[3], real[4]);

    /* Inverse FFT */
    gtcrn_fft_inverse(plan, real, imag);

    printf("\nAfter IFFT:\n");
    double recon_energy = 0;
    for (int i = 0; i < N_FFT; i++) {
        recon_energy += real[i] * real[i];
    }
    printf("  Reconstructed energy: %.6f\n", recon_energy);
    printf("  First 5: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           real[0], real[1], real[2], real[3], real[4]);

    /* Compare */
    printf("\n=== Comparison ===\n");
    printf("Energy ratio (recon/orig): %.6f\n", recon_energy / orig_energy);

    double error = 0;
    for (int i = 0; i < N_FFT; i++) {
        double diff = real[i] - original[i];
        error += diff * diff;
    }
    printf("Error energy: %.10f\n", error);

    /* Sample comparison */
    printf("\n=== Sample comparison ===\n");
    printf("Idx | Original    | Reconstructed | Diff\n");
    printf("----+-------------+---------------+--------\n");
    for (int i = 0; i < 10; i++) {
        printf("%3d | %11.6f | %13.6f | %.6f\n",
               i, original[i], real[i], real[i] - original[i]);
    }

    /* Clean up */
    free(real);
    free(imag);
    free(original);
    gtcrn_fft_plan_destroy(plan);

    return 0;
}
