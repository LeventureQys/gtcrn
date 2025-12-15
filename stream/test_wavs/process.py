import os
import librosa
import soundfile as sf
from tqdm import tqdm

SRC_DIR = "noisy"     # 原始 wav 文件夹
DST_DIR = "noisy_16k"     # 输出文件夹
TARGET_SR = 16000

os.makedirs(DST_DIR, exist_ok=True)

def is_wav(fname):
    return fname.lower().endswith(".wav")

wav_files = []
for root, _, files in os.walk(SRC_DIR):
    for f in files:
        if is_wav(f):
            wav_files.append(os.path.join(root, f))

print(f"Found {len(wav_files)} wav files")

for wav_path in tqdm(wav_files):
    # 读取音频（librosa 会自动转成 float32）
    audio, sr = librosa.load(wav_path, sr=None, mono=False)

    # 多声道 → 单声道
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # 重采样
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # 输出路径（保持文件名）
    out_path = os.path.join(DST_DIR, os.path.basename(wav_path))
    sf.write(out_path, audio, TARGET_SR)

print("Done.")
