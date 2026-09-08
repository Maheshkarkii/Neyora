import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import argparse
from src.inference.transcribe import NepaliTranscriber

def main():
    parser = argparse.ArgumentParser(description="Transcribe Nepali Audio using NepaliASR CTC model.")
    parser.add_argument("--audio", type=str, default="data/raw/asr/audio/nep_spk_01_0000.wav", help="Path to audio .wav file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_nepali_asr.pt", help="Path to ASR checkpoint")
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}")
        return

    transcriber = NepaliTranscriber(checkpoint_path=args.checkpoint)
    print(f"Transcribing audio: {args.audio}")
    result = transcriber.transcribe_file(args.audio)
    print(f"Predicted Nepali Transcription: {result}")

if __name__ == '__main__':
    main()
