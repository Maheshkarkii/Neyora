import os
import sys
import glob
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.utils.logger import get_logger

logger = get_logger("batch_translate")

def main():
    parser = argparse.ArgumentParser(description="Batch Nepali Voice Translation CLI")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .wav audio files")
    parser.add_argument("--output", type=str, default="results/batch_translation_results.csv", help="Path to save output CSV")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    parser.add_argument("--asr_checkpoint", type=str, default="checkpoints/best_nepali_asr.pt", help="Path to ASR checkpoint")
    parser.add_argument("--translation_checkpoint", type=str, default="checkpoints/best_lstm_attention.pt", help="Path to NMT checkpoint")

    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' not found.")
        sys.exit(1)

    audio_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")) + glob.glob(os.path.join(args.input_dir, "**/*.wav"), recursive=True))
    if not audio_files:
        print(f"No .wav audio files found in '{args.input_dir}'.")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio files in '{args.input_dir}'.")
    print("Loading models...")

    translator = NepaliVoiceTranslator(
        asr_checkpoint=args.asr_checkpoint,
        translation_checkpoint=args.translation_checkpoint,
        device=args.device
    )

    records = []
    for idx, audio_path in enumerate(audio_files, 1):
        print(f"[{idx}/{len(audio_files)}] Processing: {os.path.basename(audio_path)}")
        try:
            res = translator.translate_audio(audio_path)
            records.append({
                "audio_path": audio_path,
                "transcription": res["transcription"],
                "translation": res["translation"],
                "audio_duration_sec": res["audio_duration_sec"],
                "asr_latency_sec": res["asr_latency_sec"],
                "translation_latency_sec": res["translation_latency_sec"],
                "total_latency_sec": res["total_latency_sec"],
                "real_time_factor": res["real_time_factor"],
                "asr_confidence": res["asr_confidence"]
            })
        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            records.append({
                "audio_path": audio_path,
                "transcription": f"ERROR: {e}",
                "translation": "",
                "audio_duration_sec": 0.0,
                "asr_latency_sec": 0.0,
                "translation_latency_sec": 0.0,
                "total_latency_sec": 0.0,
                "real_time_factor": 0.0,
                "asr_confidence": 0.0
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nBatch processing complete! Saved {len(df)} results to '{args.output}'.")

if __name__ == "__main__":
    main()
