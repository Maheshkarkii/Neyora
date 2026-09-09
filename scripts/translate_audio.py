import os
import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.evaluation.attention_visualization import print_text_attention_alignment, plot_attention_matrix

def main():
    parser = argparse.ArgumentParser(description="End-to-End Nepali Voice Translator CLI (Phase 5)")
    parser.add_argument("--audio", type=str, required=True, help="Path to input Nepali audio .wav file")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--asr_checkpoint", type=str, default="checkpoints/best_nepali_asr.pt", help="Path to ASR checkpoint")
    parser.add_argument("--translation_checkpoint", type=str, default="checkpoints/best_lstm_attention.pt", help="Path to NMT checkpoint")
    parser.add_argument("--diagnostic", action="store_true", help="Print detailed diagnostic token breakdown and attention")
    parser.add_argument("--save_plot", type=str, default=None, help="Optional path to save attention heatmap image")
    parser.add_argument("--max_len", type=int, default=30, help="Maximum translation output tokens")

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file '{args.audio}' does not exist.")
        sys.exit(1)

    print("=" * 60)
    print("      🎙️  Nepali Voice Translator (End-to-End)  🇬🇧")
    print("=" * 60)
    print("Loading models...")

    translator = NepaliVoiceTranslator(
        asr_checkpoint=args.asr_checkpoint,
        translation_checkpoint=args.translation_checkpoint,
        device=args.device
    )

    print("\nProcessing Audio...")
    result = translator.translate_audio(
        audio_input=args.audio,
        max_len=args.max_len,
        diagnostic=args.diagnostic or bool(args.save_plot)
    )

    print("\n" + "=" * 60)
    print(f"🎙️  Input Audio  : {args.audio}")
    print(f"⏱️  Audio Duration: {result['audio_duration_sec']:.2f} s")
    print(f"⚡ Latency Breakdown:")
    print(f"    - ASR Inference        : {result['asr_latency_sec']:.4f} s")
    print(f"    - Translation Inference: {result['translation_latency_sec']:.4f} s")
    print(f"    - Total Time           : {result['total_latency_sec']:.4f} s")
    print(f"    - Real-Time Factor(RTF): {result['real_time_factor']:.4f}")
    print(f"📊 Diagnostic Confidence  : {result['asr_confidence']:.4f}")
    print(f"   ℹ️  Note: {result['confidence_note']}")
    print("-" * 60)
    print("🇳🇵 Nepali transcription:")
    print(f"   {result['transcription']}")
    print("\n🇬🇧 English translation:")
    print(f"   {result['translation']}")
    print("=" * 60)

    if args.diagnostic and "diagnostic" in result:
        diag = result["diagnostic"]
        print("\n--- Diagnostic Details ---")
        print(f"Source tokens: {diag['src_tokens']}")
        print(f"Target tokens: {diag['tgt_tokens']}")
        if diag["attention_matrix"] is not None:
            print_text_attention_alignment(diag["attention_matrix"], diag["src_tokens"], diag["tgt_tokens"])

    if args.save_plot and "diagnostic" in result and result["diagnostic"]["attention_matrix"] is not None:
        os.makedirs(os.path.dirname(args.save_plot) or ".", exist_ok=True)
        plot_attention_matrix(
            result["diagnostic"]["attention_matrix"],
            result["diagnostic"]["src_tokens"],
            result["diagnostic"]["tgt_tokens"],
            save_path=args.save_plot
        )
        print(f"\nAttention heatmap saved to: {args.save_plot}")

if __name__ == "__main__":
    main()
