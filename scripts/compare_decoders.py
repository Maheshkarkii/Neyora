import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.evaluation.end_to_end import EndToEndEvaluator
from scripts.evaluate_end_to_end import get_end_to_end_test_dataset
from src.evaluation.asr_metrics import compute_cer, compute_wer
from src.evaluation.metrics import calculate_sentence_bleu

def main():
    parser = argparse.ArgumentParser(description="Compare Greedy vs Beam Search Decoding")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/decoder_comparison.json")
    args = parser.parse_args()

    print("=" * 70)
    print("      🔍  COMPARATIVE BENCHMARK: GREEDY vs BEAM SEARCH DECODING  🔍")
    print("=" * 70)

    dataset = get_end_to_end_test_dataset()
    print(f"Loaded {len(dataset)} evaluation samples.")

    configurations = [
        {"name": "Greedy (Baseline)", "asr_decoder": "greedy", "nmt_decoder": "greedy", "beam_width": 1},
        {"name": "Beam Search (k=3)", "asr_decoder": "beam_search", "nmt_decoder": "beam_search", "beam_width": 3},
        {"name": "Beam Search (k=5)", "asr_decoder": "beam_search", "nmt_decoder": "beam_search", "beam_width": 5},
    ]

    results_table = []

    for cfg in configurations:
        print(f"\nEvaluating: {cfg['name']}...")
        translator = NepaliVoiceTranslator(
            device=args.device,
            asr_decoder_type=cfg["asr_decoder"],
            translation_decoder_type=cfg["nmt_decoder"],
            beam_width=cfg["beam_width"]
        )

        latencies = []
        cers = []
        wers = []
        bleus = []

        for item in dataset:
            start = time.perf_counter()
            res = translator.translate_audio(
                item["audio_path"],
                asr_decoder_type=cfg["asr_decoder"],
                translation_decoder_type=cfg["nmt_decoder"]
            )
            lat = time.perf_counter() - start
            latencies.append(lat)

            cer = compute_cer(item["ground_truth_nepali"], res["transcription"])
            wer = compute_wer(item["ground_truth_nepali"], res["transcription"])
            cers.append(cer)
            wers.append(wer)

            gt_tokens = item["ground_truth_english"].strip().lower().split()
            pred_tokens = res["translation"].strip().lower().split()
            bleu = calculate_sentence_bleu(gt_tokens, pred_tokens) * 100.0
            bleus.append(bleu)

        row = {
            "Configuration": cfg["name"],
            "ASR Decoder": cfg["asr_decoder"],
            "NMT Decoder": cfg["nmt_decoder"],
            "Beam Width": cfg["beam_width"],
            "Mean CER": round(float(np.mean(cers)), 4),
            "Mean WER": round(float(np.mean(wers)), 4),
            "Mean BLEU (%)": round(float(np.mean(bleus)), 2),
            "Mean Latency (s)": round(float(np.mean(latencies)), 4),
            "P95 Latency (s)": round(float(np.percentile(latencies, 95)), 4),
        }
        results_table.append(row)

    print("\n" + "=" * 70)
    print("                    🏆  DECODER COMPARISON RESULTS  🏆")
    print("=" * 70)
    df_res = pd.DataFrame(results_table)
    print(df_res[["Configuration", "Mean CER", "Mean WER", "Mean BLEU (%)", "Mean Latency (s)"]].to_string(index=False))
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"Decoder comparison saved to '{args.output_json}'.")

if __name__ == "__main__":
    main()
