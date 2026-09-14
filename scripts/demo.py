import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.voice_translator import NepaliVoiceTranslator

def main():
    parser = argparse.ArgumentParser(description="Nepali Voice Translator Interactive Portfolio Demo")
    parser.add_argument("--audio", type=str, default="data/raw/asr/audio/nep_spk_07_019.wav", help="Audio file path")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('cpu' or 'cuda')")
    parser.add_argument("--beam", action="store_true", help="Use Beam Search decoding instead of Greedy")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' does not exist.")
        sys.exit(1)

    decoder_mode = "beam_search" if args.beam else "greedy"
    translator = NepaliVoiceTranslator(
        device=args.device,
        asr_decoder_type=decoder_mode,
        translation_decoder_type=decoder_mode
    )

    result = translator.translate_audio(args.audio)

    print("\n" + "=" * 44)
    print("NEPALI VOICE TRANSLATOR")
    print("=" * 44)
    print(f"\nInput:\n{args.audio}")
    print(f"\nNepali transcription:\n{result['transcription']}")
    print(f"\nEnglish translation:\n{result['translation']}")
    print(f"\nASR latency:\n{result['asr_latency_sec']:.4f} s")
    print(f"\nTranslation latency:\n{result['translation_latency_sec']:.4f} s")
    print(f"\nTotal latency:\n{result['total_latency_sec']:.4f} s (RTF: {result['real_time_factor']:.4f})")
    print("=" * 44 + "\n")

if __name__ == "__main__":
    main()
