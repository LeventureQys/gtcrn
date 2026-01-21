/**
 * example_realtime_denoise.c - Complete example of real-time audio denoising
 *
 * This example demonstrates:
 * 1. Loading GTCRN model weights
 * 2. Creating streaming processor
 * 3. Processing audio file in real-time chunks
 * 4. Saving enhanced audio
 *
 * Compile:
 *   gcc -o denoise example_realtime_denoise.c \
 *       gtcrn_streaming_optimized.c gtcrn_streaming.c gtcrn_streaming_impl.c \
 *       gtcrn_model.c gtcrn_modules.c stream_conv.c stft.c weight_loader.c \
 *       GRU.c conv2d.c batchnorm2d.c nn_layers.c layernorm.c -lm -O2
 *
 * Usage:
 *   ./denoise
 * 
 * Note: 参数路径已写死在代码中，可直接执行
 *       如需修改路径，请编辑 main 函数中的路径常量
 */

#include "gtcrn_model.h"
#include "gtcrn_streaming.h"
#include "weight_loader.h"
#include "stft.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ============================================================================
// WAV File I/O (Simplified)
// ============================================================================

typedef struct {
    int sample_rate;
    int num_channels;
    int num_samples;
    float* data;
} AudioData;

/**
 * Read WAV file (simplified, assumes 16-bit PCM)
 */
AudioData* read_wav(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return NULL;
    }

    // Read WAV header (simplified)
    char header[44];
    fread(header, 1, 44, fp);

    // Extract sample rate and data size
    int sample_rate = *(int*)(header + 24);
    int data_size = *(int*)(header + 40);
    int num_samples = data_size / 2;  // 16-bit = 2 bytes per sample

    printf("Reading WAV: %s\n", filename);
    printf("  Sample rate: %d Hz\n", sample_rate);
    printf("  Samples: %d\n", num_samples);
    printf("  Duration: %.2f seconds\n", (float)num_samples / sample_rate);

    // Read audio data
    short* pcm_data = (short*)malloc(num_samples * sizeof(short));
    fread(pcm_data, sizeof(short), num_samples, fp);
    fclose(fp);

    // Convert to float [-1, 1]
    AudioData* audio = (AudioData*)malloc(sizeof(AudioData));
    audio->sample_rate = sample_rate;
    audio->num_channels = 1;
    audio->num_samples = num_samples;
    audio->data = (float*)malloc(num_samples * sizeof(float));

    for (int i = 0; i < num_samples; i++) {
        audio->data[i] = pcm_data[i] / 32768.0f;
    }

    free(pcm_data);
    return audio;
}

/**
 * Write WAV file (simplified, 16-bit PCM)
 */
int write_wav(const char* filename, AudioData* audio) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create %s\n", filename);
        return -1;
    }

    // Write WAV header
    int data_size = audio->num_samples * 2;  // 16-bit
    int file_size = 36 + data_size;

    fwrite("RIFF", 1, 4, fp);
    fwrite(&file_size, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);
    fwrite("fmt ", 1, 4, fp);

    int fmt_size = 16;
    short audio_format = 1;  // PCM
    short num_channels = 1;
    int sample_rate = audio->sample_rate;
    int byte_rate = sample_rate * 2;
    short block_align = 2;
    short bits_per_sample = 16;

    fwrite(&fmt_size, 4, 1, fp);
    fwrite(&audio_format, 2, 1, fp);
    fwrite(&num_channels, 2, 1, fp);
    fwrite(&sample_rate, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    fwrite(&bits_per_sample, 2, 1, fp);

    fwrite("data", 1, 4, fp);
    fwrite(&data_size, 4, 1, fp);

    // Convert float to 16-bit PCM and write
    for (int i = 0; i < audio->num_samples; i++) {
        float sample = audio->data[i];
        // Clamp to [-1, 1]
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;

        short pcm_sample = (short)(sample * 32767.0f);
        fwrite(&pcm_sample, 2, 1, fp);
    }

    fclose(fp);

    printf("Wrote WAV: %s\n", filename);
    printf("  Samples: %d\n", audio->num_samples);
    printf("  Duration: %.2f seconds\n", (float)audio->num_samples / audio->sample_rate);

    return 0;
}

void free_audio(AudioData* audio) {
    if (audio) {
        free(audio->data);
        free(audio);
    }
}

// ============================================================================
// Main Program
// ============================================================================

int main(void) {
    // 写死的参数路径，可根据需要修改
    const char* input_file = "D://WorkShop//gtcrn-project//Unit_C//Properties//Test_Wavs//input_16khz//00071_6168521e947324b2c17b8c2ee641b294_16k.wav";
    const char* output_file = "D://WorkShop//gtcrn-project//Unit_C//Properties//Test_Wavs//output_16khz//output.wav";
    const char* weights_dir = "D://WorkShop//gtcrn-project//Unit_C//Properties//weight//model_trained_on_dns3.tar";

    printf("\n");
    printf("=================================================================\n");
    printf("GTCRN Real-Time Audio Denoising\n");
    printf("=================================================================\n\n");

    // Step 1: Load audio
    printf("Step 1: Loading audio...\n");
    AudioData* input_audio = read_wav(input_file);
    if (!input_audio) {
        fprintf(stderr, "Failed to load audio\n");
        return 1;
    }
    printf("\n");

    // Step 2: Create GTCRN model
    printf("Step 2: Creating GTCRN model...\n");
    GTCRN* model = gtcrn_create();  // 修复: 使用正确的函数名
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        free_audio(input_audio);
        return 1;
    }
    printf("\n");

    // Step 3: Load weights
    printf("Step 3: Loading model weights...\n");
    if (load_gtcrn_weights(model, weights_dir) != 0) {
        fprintf(stderr, "Warning: Failed to load weights, using random initialization\n");
        fprintf(stderr, "         For actual denoising, you need to export weights from PyTorch\n");
        fprintf(stderr, "         See export_weights.py for instructions\n");
    }
    printf("\n");

    // Step 4: Create streaming processor
    printf("Step 4: Creating streaming processor...\n");
    int chunk_size = 768;  // 16ms @ 48kHz
    GTCRNStreaming* stream = gtcrn_streaming_create(model, input_audio->sample_rate, chunk_size);
    if (!stream) {
        fprintf(stderr, "Failed to create streaming processor\n");
        gtcrn_free(model);
        free_audio(input_audio);
        return 1;
    }
    printf("\n");

    // Step 5: Process audio
    printf("Step 5: Processing audio...\n");
    AudioData* output_audio = (AudioData*)malloc(sizeof(AudioData));
    output_audio->sample_rate = input_audio->sample_rate;
    output_audio->num_channels = 1;
    output_audio->num_samples = input_audio->num_samples;
    output_audio->data = (float*)calloc(output_audio->num_samples, sizeof(float));

    clock_t start_time = clock();

    // Process in chunks (simulating real-time)
    int processed = 0;
    int total_chunks = (input_audio->num_samples + chunk_size - 1) / chunk_size;

    printf("Processing %d chunks...\n", total_chunks);

    for (int chunk = 0; chunk < total_chunks; chunk++) {
        int remaining = input_audio->num_samples - processed;
        int current_chunk = remaining < chunk_size ? remaining : chunk_size;

        // Process chunk
        gtcrn_streaming_process_chunk_optimized(
            stream,
            input_audio->data + processed,
            output_audio->data + processed
        );

        processed += current_chunk;

        // Print progress
        if ((chunk + 1) % 10 == 0 || chunk == total_chunks - 1) {
            float progress = (float)(chunk + 1) / total_chunks * 100;
            printf("  Progress: %.1f%% (%d/%d chunks)\r", progress, chunk + 1, total_chunks);
            fflush(stdout);
        }
    }

    printf("\n");

    clock_t end_time = clock();
    float processing_time = (float)(end_time - start_time) / CLOCKS_PER_SEC;
    float audio_duration = (float)input_audio->num_samples / input_audio->sample_rate;
    float rtf = processing_time / audio_duration;

    printf("\n");
    printf("Processing complete!\n");
    printf("  Audio duration: %.2f seconds\n", audio_duration);
    printf("  Processing time: %.2f seconds\n", processing_time);
    printf("  Real-time factor: %.3f (%.1fx %s)\n",
           rtf, 1.0f / rtf, rtf < 1.0f ? "faster than real-time" : "slower than real-time");

    // Get statistics
    int frames_processed;
    float avg_latency;
    gtcrn_streaming_get_stats(stream, &frames_processed, &avg_latency);
    printf("  Frames processed: %d\n", frames_processed);
    printf("  Average latency: %.2f ms\n", avg_latency);
    printf("  Total latency: %.2f ms\n", gtcrn_streaming_get_latency_ms(stream));
    printf("\n");

    // Step 6: Save output
    printf("Step 6: Saving enhanced audio...\n");
    if (write_wav(output_file, output_audio) != 0) {
        fprintf(stderr, "Failed to write output\n");
    }
    printf("\n");

    // Cleanup
    printf("Cleaning up...\n");
    gtcrn_streaming_free(stream);
    gtcrn_free(model);
    free_audio(input_audio);
    free_audio(output_audio);

    printf("\n");
    printf("=================================================================\n");
    printf("Done!\n");
    printf("=================================================================\n\n");

    printf("Next steps:\n");
    printf("  1. Export weights from PyTorch model using export_weights.py\n");
    printf("  2. Run this program with actual weights for real denoising\n");
    printf("  3. Compare input and output audio files\n\n");

    return 0;
}
