"""
Verify frame indexing between Python (center=True) and C streaming.
"""
import os
import numpy as np
import torch
import soundfile as sf

def main():
    print("=== Verify Frame Indexing ===\n")

    gtcrn_dir = os.path.dirname(os.path.abspath(__file__))
    noisy_path = os.path.join(gtcrn_dir, 'test_wavs/noisy_16k/00027_1_keyboard_loud_snr+5dB_noisy.wav')
    audio, sr = sf.read(noisy_path)

    n_fft = 512
    hop_length = 256
    win_length = 512
    window = torch.sqrt(torch.hann_window(win_length))

    # Python: pad 256 zeros at beginning, then use torch.stft with center=True (default)
    audio_padded = np.concatenate([np.zeros(256), audio])
    audio_tensor = torch.from_numpy(audio_padded.astype(np.float32))

    spec = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length,
                      win_length=win_length, window=window, return_complex=False)
    print(f"Python torch.stft (center=True default) shape: {spec.shape}")

    # Simulate C streaming:
    # C initializes stft_input_buffer = zeros(256)
    # C frame 1 (g_stream_frame_count=1):
    #   stft_window = [stft_input_buffer, input_frame[0]] = [zeros(256), audio[0:256]]
    #   then stft_input_buffer = audio[0:256]
    # C frame 2 (g_stream_frame_count=2):
    #   stft_window = [stft_input_buffer, input_frame[1]] = [audio[0:256], audio[256:512]]
    #   then stft_input_buffer = audio[256:512]
    # ...

    # But Python with center=True adds n_fft//2=256 zeros on EACH side!
    # So Python frame 0 uses: [zeros(256), padded_audio[0:256]] = [zeros(256), zeros(256)]
    # Python frame 1 uses: [padded_audio[0:256], padded_audio[256:512]] = [zeros(256), audio[0:256]]
    # Python frame 2 uses: [padded_audio[256:512], padded_audio[512:768]] = [audio[0:256], audio[256:512]]

    # Compare:
    # C frame 1 window = [zeros, audio[0:256]] = Python frame 1 window ✓
    # C frame 2 window = [audio[0:256], audio[256:512]] = Python frame 2 window ✓

    # So C frame N (1-indexed) = Python frame N (0-indexed starting from frame 0 which is all zeros)
    # But the debug_decoder.py uses target_frame=5 which is Python frame 5 (0-indexed)
    # And C Frame 6 (1-indexed) should match Python frame 5 if we count C starting from 1

    # Wait, let me re-check the C code:
    # g_stream_frame_count is incremented BEFORE processing, so:
    # - First call: g_stream_frame_count becomes 1
    # - Second call: g_stream_frame_count becomes 2
    # etc.

    # So C Frame 6 means the 6th call, which should correspond to:
    # stft_window = [audio[4*256:5*256], audio[5*256:6*256]] = [audio[1024:1280], audio[1280:1536]]

    # For Python with padded audio and center=True:
    # Frame N uses audio_padded with center padding...

    # Actually, let me just trace it exactly
    print("\n=== Tracing exact frames ===\n")

    # What center=True does: pads the signal with n_fft//2 on each side
    # So effective_audio = [zeros(256), audio_padded, zeros(256)]
    # = [zeros(256), zeros(256), audio, zeros(256)]
    # = [zeros(512), audio, zeros(256)]

    # Frame i uses indices [i*hop_length : i*hop_length + n_fft]
    # Frame 0: [0:512] from effective_audio = [zeros(512)]
    # Frame 1: [256:768] = [zeros(256), audio[0:256]]
    # Frame 2: [512:1024] = [audio[0:512]]
    # ...

    # Hmm, this is getting complex. Let me just print what frames actually contain

    # Simulate C streaming manually
    def manual_stft_frame(frame_512):
        windowed = frame_512 * window.numpy()
        fft_result = np.fft.rfft(windowed)
        return fft_result.real.sum(), fft_result.imag.sum()

    stft_input_buffer = np.zeros(256)
    print("C streaming simulation (1-indexed frame count):")
    for c_frame in range(1, 8):
        input_frame_idx = c_frame - 1  # 0-indexed input frame
        input_frame = audio[input_frame_idx * 256:(input_frame_idx + 1) * 256]
        stft_window = np.concatenate([stft_input_buffer, input_frame])
        stft_input_buffer = input_frame.copy()

        real_sum, imag_sum = manual_stft_frame(stft_window)
        print(f"  C Frame {c_frame}: real_sum={real_sum:.4f}, imag_sum={imag_sum:.4f}")

    print("\nPython torch.stft (center=True, audio_padded = [zeros(256), audio]):")
    for py_frame in range(7):
        real_sum = spec[:, py_frame, 0].sum().item()
        imag_sum = spec[:, py_frame, 1].sum().item()
        print(f"  Py Frame {py_frame}: real_sum={real_sum:.4f}, imag_sum={imag_sum:.4f}")

    print("\n=== Frame Mapping ===")
    print("C Frame 1 should match Py Frame 1 (both: [zeros, audio[0:256]])")
    print("C Frame 2 should match Py Frame 2 (both: [audio[0:256], audio[256:512]])")
    print("...")
    print("C Frame 6 should match Py Frame 6")

    print("\nVerification:")
    print(f"  C Frame 6: real_sum={manual_stft_frame(audio[1024:1536])[0]:.4f}")
    print(f"  Py Frame 6: real_sum={spec[:, 6, 0].sum().item():.4f}")

if __name__ == "__main__":
    main()
