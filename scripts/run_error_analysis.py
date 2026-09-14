import os
import sys
import pandas as pd
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.error_analyzer import run_systematic_error_analysis

def main():
    e2e_results_path = "results/end_to_end_results.csv"
    if not os.path.exists(e2e_results_path):
        print(f"Error: {e2e_results_path} not found. Please run 'python scripts/evaluate_end_to_end.py' first.")
        sys.exit(1)

    df_eval = pd.read_csv(e2e_results_path)
    print("=" * 70)
    print("           🔬  PHASE 6: SYSTEMATIC ERROR ANALYSIS  🔬")
    print("=" * 70)

    df_analysis, summary = run_systematic_error_analysis(df_eval, output_csv_path="results/error_analysis.csv")

    print("\n--- ASR Error Breakdown ---")
    for err, cnt in summary["asr_error_distribution"].items():
        pct = (cnt / summary["total_samples"]) * 100.0
        print(f"  • {err:<25}: {cnt:>3} ({pct:>5.1f}%)")

    print("\n--- Translation Error Breakdown ---")
    for err, cnt in summary["nmt_error_distribution"].items():
        pct = (cnt / summary["total_samples"]) * 100.0
        print(f"  • {err:<25}: {cnt:>3} ({pct:>5.1f}%)")

    print("\n--- Stratified Performance by Audio Duration ---")
    for dur, metrics in summary["performance_by_duration"].get("cer", {}).items():
        cer = summary["performance_by_duration"]["cer"][dur]
        wer = summary["performance_by_duration"]["wer"][dur]
        print(f"  • Duration: {dur:<18} -> CER: {cer:.4f}, WER: {wer:.4f}")

    print("\n--- Stratified Performance by Transcript Length ---")
    for length, metrics in summary["performance_by_length"].get("cer", {}).items():
        cer = summary["performance_by_length"]["cer"][length]
        wer = summary["performance_by_length"]["wer"][length]
        print(f"  • Sentence Length: {length:<20} -> CER: {cer:.4f}, WER: {wer:.4f}")

    print("\nDetailed breakdown saved to 'results/error_analysis.csv'.")

if __name__ == "__main__":
    main()
