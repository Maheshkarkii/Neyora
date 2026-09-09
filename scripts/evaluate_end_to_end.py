import os
import sys
import json
import argparse
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.evaluation.end_to_end import EndToEndEvaluator

def get_end_to_end_test_dataset() -> list:
    """
    Creates an end-to-end evaluation dataset combining test manifest audio files and
    representative recordings with verified ground-truth Nepali and English text.
    """
    # Canonical mapping of all 20 unique dataset sentences to English translations
    nepali_to_english_map = {
        "आज बेलुका भेटौंला।": "see you this evening .",
        "आजको मौसम धेरै राम्रो छ।": "the weather is very good today .",
        "काठमाडौं नेपालको राजधानी सहर हो।": "kathmandu is the capital city of nepal .",
        "तपाईंको नाम के हो।": "what is your name .",
        "धन्यवाद तपाईंको सहयोगको लागि।": "thank you for your help .",
        "नमस्ते म नेपालबाट आएको हुँ।": "hello i came from nepal .",
        "नेपाल एक सुन्दर र शान्त देश हो।": "nepal is a beautiful and peaceful country .",
        "पानी पिउनु स्वास्थ्यको लागि राम्रो हुन्छ।": "drinking water is good for health .",
        "पुस्तकालयमा धेरै उपयोगी पुस्तकहरू छन्।": "there are many useful books in the library .",
        "प्राकृतिक सौन्दर्यले भरिपूर्ण छ नेपाल।": "nepal is full of natural beauty .",
        "म नेपाली भाषा सिक्दैछु।": "i am learning the nepali language .",
        "मलाई नेपाली खाना धेरै मन पर्छ।": "i like nepali food very much .",
        "मेहनत गरे अवश्य सफलता मिल्छ।": "hard work definitely brings success .",
        "यो परियोजना धेरै महत्त्वपूर्ण छ।": "this project is very important .",
        "विद्यालयमा नयाँ विद्यार्थीहरू आएका छन्।": "new students have arrived at the school .",
        "सगरमाथा संसारको सबैभन्दा अग्लो शिखर हो।": "mount everest is the highest peak in the world .",
        "समयको सहि सदुपयोग गर्नुपर्छ।": "time must be utilized properly .",
        "स्वास्थ्य नै सबैभन्दा ठूलो धन हो।": "health is the greatest wealth .",
        "हामी सबै मिलेर काम गर्नुपर्छ।": "we must all work together .",
        "हाम्रो संस्कृति हाम्रो पहिचान हो।": "our culture is our identity ."
    }

    # Load test manifest
    test_manifest_path = "data/processed/asr/test_manifest.json"
    dataset = []
    
    if os.path.exists(test_manifest_path):
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            manifest_items = json.load(f)
        for item in manifest_items:
            nep_text = item["text"]
            eng_text = nepali_to_english_map.get(nep_text, "unknown")
            dataset.append({
                "audio_path": item["audio_filepath"],
                "ground_truth_nepali": nep_text,
                "ground_truth_english": eng_text
            })

    # Also include additional multi-speaker diverse evaluation audios to test short, medium, long speech
    val_manifest_path = "data/processed/asr/val_manifest.json"
    if os.path.exists(val_manifest_path):
        with open(val_manifest_path, "r", encoding="utf-8") as f:
            val_items = json.load(f)
        for item in val_items[:10]: # Include 10 diverse validation samples
            nep_text = item["text"]
            eng_text = nepali_to_english_map.get(nep_text, "unknown")
            dataset.append({
                "audio_path": item["audio_filepath"],
                "ground_truth_nepali": nep_text,
                "ground_truth_english": eng_text
            })

    return dataset

def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation Script (Phase 5)")
    parser.add_argument("--output_csv", type=str, default="results/end_to_end_results.csv", help="Path to save evaluation CSV")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")
    args = parser.parse_args()

    print("=" * 70)
    print("        📊  PHASE 5: END-TO-END PIPELINE EVALUATION  📊")
    print("=" * 70)

    dataset = get_end_to_end_test_dataset()
    print(f"Loaded {len(dataset)} evaluation samples.")

    translator = NepaliVoiceTranslator(device=args.device)
    evaluator = EndToEndEvaluator(translator=translator)

    print("\nRunning 3-Mode Evaluation across dataset...")
    df, summary = evaluator.evaluate_dataset(dataset, output_csv_path=args.output_csv)

    print("\n" + "=" * 70)
    print("                🏆  EVALUATION RESULTS COMPARISON TABLE  🏆")
    print("=" * 70)
    print(f"{'Pipeline':<35} | {'ASR CER':>10} | {'ASR WER':>10} | {'Translation BLEU':>16}")
    print("-" * 77)
    print(f"{'Ground-truth Nepali → English':<35} | {'N/A':>10} | {'N/A':>10} | {summary['mode1_gt_bleu']:>15.2f}%")
    print(f"{'ASR Nepali → English':<35} | {summary['asr_avg_cer']:>9.4f} | {summary['asr_avg_wer']:>9.4f} | {summary['mode2_e2e_bleu']:>15.2f}%")
    print(f"{'End-to-end (Audio → Translation)':<35} | {summary['asr_avg_cer']:>9.4f} | {summary['asr_avg_wer']:>9.4f} | {summary['mode2_e2e_bleu']:>15.2f}%")
    print("=" * 70)

    print("\n--- Error Classification Breakdown ---")
    for err_type, count in summary["error_distribution"].items():
        pct = (count / summary["num_samples"]) * 100.0
        print(f"  • {err_type:<20}: {count:>3} samples ({pct:>5.1f}%)")

    print(f"\n--- Latency & Throughput ---")
    print(f"  • Average End-to-End Latency : {summary['avg_total_latency_sec']:.4f} s")
    print(f"  • Average Real-Time Factor   : {summary['avg_real_time_factor']:.4f}")
    if summary['avg_real_time_factor'] < 1.0:
        print(f"  • Real-Time Verification    : RTF < 1.0 (System is faster than real-time)")
    else:
        print(f"  • Real-Time Verification    : RTF >= 1.0")

    print(f"\nDetailed per-sample results saved to '{args.output_csv}'.")

if __name__ == "__main__":
    main()
