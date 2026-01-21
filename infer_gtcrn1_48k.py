"""
Inference script for GTCRN1 48kHz model
"""
import os
import torch
import torchaudio
import argparse
from gtcrn1 import GTCRN
import numpy as np
from tqdm import tqdm


class GTCRN1Inference:
    """GTCRN1 inference wrapper for 48kHz audio"""
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # STFT parameters for 48kHz
        self.sample_rate = 48000
        self.n_fft = 1536
        self.hop_length = 768
        self.window = torch.hann_window(self.n_fft).pow(0.5).to(self.device)

        # Load model
        self.model = GTCRN().to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

        print(f"Model loaded from: {checkpoint_path}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e3:.2f}K")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def process_audio(self, audio_path, output_path=None):
        """
        Process a single audio file

        Args:
            audio_path: Path to input audio file
            output_path: Path to save enhanced audio (optional)

        Returns:
            enhanced_audio: Enhanced audio tensor
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate:
            print(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Move to device
        audio = audio.to(self.device)

        # Process
        with torch.no_grad():
            enhanced_audio = self.enhance(audio)

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            torchaudio.save(output_path, enhanced_audio.cpu(), self.sample_rate)
            print(f"Saved enhanced audio to: {output_path}")

        return enhanced_audio

    def enhance(self, audio):
        """
        Enhance audio using GTCRN1

        Args:
            audio: Input audio tensor (1, samples)

        Returns:
            enhanced_audio: Enhanced audio tensor (1, samples)
        """
        original_length = audio.shape[1]

        # Pad to ensure divisible by hop_length
        pad_length = (self.hop_length - (original_length % self.hop_length)) % self.hop_length
        if pad_length > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_length))

        # STFT
        noisy_spec = torch.stft(audio.squeeze(0), self.n_fft, self.hop_length,
                               self.n_fft, self.window, return_complex=False)

        # Add batch dimension
        noisy_spec = noisy_spec.unsqueeze(0)

        # Enhance
        enhanced_spec = self.model(noisy_spec)

        # Remove batch dimension
        enhanced_spec = enhanced_spec.squeeze(0)

        # ISTFT
        enhanced_audio = torch.istft(enhanced_spec, self.n_fft, self.hop_length,
                                     self.n_fft, self.window, return_complex=False)

        # Remove padding
        enhanced_audio = enhanced_audio[:original_length]

        return enhanced_audio.unsqueeze(0)

    def process_folder(self, input_folder, output_folder):
        """
        Process all audio files in a folder

        Args:
            input_folder: Path to input folder
            output_folder: Path to output folder
        """
        os.makedirs(output_folder, exist_ok=True)

        # Get all audio files
        audio_files = [f for f in os.listdir(input_folder)
                      if f.endswith(('.wav', '.flac', '.mp3'))]

        print(f"Found {len(audio_files)} audio files")

        # Process each file
        for audio_file in tqdm(audio_files, desc="Processing"):
            input_path = os.path.join(input_folder, audio_file)
            output_path = os.path.join(output_folder, audio_file.replace('.mp3', '.wav').replace('.flac', '.wav'))

            try:
                self.process_audio(input_path, output_path)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    def process_stream(self, audio_stream, chunk_size=48000, overlap=4800):
        """
        Process audio stream in chunks with overlap for real-time applications

        Args:
            audio_stream: Input audio tensor (1, total_samples)
            chunk_size: Size of each processing chunk (default: 1 second at 48kHz)
            overlap: Overlap between chunks (default: 0.1 second)

        Returns:
            enhanced_stream: Enhanced audio tensor
        """
        total_length = audio_stream.shape[1]
        hop_size = chunk_size - overlap

        enhanced_chunks = []

        for start in range(0, total_length, hop_size):
            end = min(start + chunk_size, total_length)
            chunk = audio_stream[:, start:end]

            # Pad last chunk if needed
            if chunk.shape[1] < chunk_size:
                pad_length = chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_length))

            # Process chunk
            with torch.no_grad():
                enhanced_chunk = self.enhance(chunk)

            # Handle overlap
            if start == 0:
                # First chunk: keep all
                enhanced_chunks.append(enhanced_chunk[:, :hop_size])
            elif end >= total_length:
                # Last chunk: keep from overlap to end
                actual_length = total_length - start
                enhanced_chunks.append(enhanced_chunk[:, overlap:overlap + actual_length - overlap])
            else:
                # Middle chunks: keep from overlap to hop_size
                enhanced_chunks.append(enhanced_chunk[:, overlap:hop_size])

        # Concatenate all chunks
        enhanced_stream = torch.cat(enhanced_chunks, dim=1)

        return enhanced_stream


def main():
    parser = argparse.ArgumentParser(description='GTCRN1 48kHz Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or folder')
    parser.add_argument('--output', type=str, required=True, help='Output audio file or folder')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--stream', action='store_true', help='Use streaming mode for long audio files')
    parser.add_argument('--chunk_size', type=int, default=48000, help='Chunk size for streaming mode (samples)')
    parser.add_argument('--overlap', type=int, default=4800, help='Overlap size for streaming mode (samples)')
    args = parser.parse_args()

    # Initialize inference
    inferencer = GTCRN1Inference(args.checkpoint, device=args.device)

    # Process
    if os.path.isfile(args.input):
        # Single file
        if args.stream:
            # Load full audio
            audio, sr = torchaudio.load(args.input)
            if sr != inferencer.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, inferencer.sample_rate)
                audio = resampler(audio)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            audio = audio.to(inferencer.device)

            # Process in streaming mode
            print(f"Processing in streaming mode (chunk_size={args.chunk_size}, overlap={args.overlap})")
            enhanced_audio = inferencer.process_stream(audio, args.chunk_size, args.overlap)

            # Save
            os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
            torchaudio.save(args.output, enhanced_audio.cpu(), inferencer.sample_rate)
            print(f"Saved enhanced audio to: {args.output}")
        else:
            # Normal mode
            inferencer.process_audio(args.input, args.output)
    elif os.path.isdir(args.input):
        # Folder
        inferencer.process_folder(args.input, args.output)
    else:
        print(f"Error: {args.input} is not a valid file or folder")


if __name__ == '__main__':
    main()
