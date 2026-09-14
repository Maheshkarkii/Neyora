import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.voice_translator import NepaliVoiceTranslator
from scripts.evaluate_end_to_end import get_end_to_end_test_dataset
from src.evaluation.asr_metrics import compute_cer, compute_wer
from src.evaluation.metrics import calculate_sentence_bleu

def evaluate_condition(translator, dataset, condition_name, modifier_fn=None):
    cers, wers, bleus, latencies = [], [], [], []

    for item in dataset:
        audio_path = item["audio_path"]
        if modifier_fn:
            waveform, sr = translator.asr_engine.preprocessor.load_audio(audio_path)
            mod_wave = modifier_fn(waveform)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, mod_wave.squeeze(0).cpu().numpy(), sr)
                test_audio = f.name
        else:
            test_audio = audio_path

        start = time.perf_counter()
        res = translator.translate_audio(test_audio)
        lat = time.perf_counter() - start
        latencies.append(lat)

        if modifier_fn:
            try:
                os.remove(test_audio)
            except Exception:
                pass

        pred_nep = res["transcription"]
        pred_eng = res["translation"]

        cers.append(compute_cer(item["ground_truth_nepali"], pred_nep))
        wers.append(compute_wer(item["ground_truth_nepali"], pred_nep))

        gt_tokens = item["ground_truth_english"].strip().lower().split()
        pred_tokens = pred_eng.strip().lower().split()
        bleus.append(calculate_sentence_bleu(gt_tokens, pred_tokens) * 100.0)

    return {
        "Condition": condition_name,
        "Mean CER": round(float(np.mean(cers)), 4),
        "Mean WER": round(float(np.mean(wers)), 4),
        "Mean BLEU (%)": round(float(np.mean(bleus)), 2),
        "Mean Latency (s)": round(float(np.mean(latencies)), 4),
    }

def main():
    parser = argparse.ArgumentParser(description="Robustness Evaluation Suite")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/robustness_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("           🛡️  PHASE 6: ROBUSTNESS & STRESS EVALUATION  🛡️")
    print("=" * 70)

    dataset = get_end_to_end_test_dataset()
    translator = NepaliVoiceTranslator(device=args.device)

    results = []

    # 1. Clean speech (Baseline)
    print("\n[1/5] Testing Clean Speech...")
    results.append(evaluate_condition(translator, dataset, "1. Clean Speech (Baseline)"))

    # 2. Additive Background Noise (Low, sigma=0.01)
    print("[2/5] Testing Low Background Noise (sigma=0.01)...")
    results.append(evaluate_condition(
        translator, dataset, "2. Low Noise (sigma=0.01)",
        lambda w: translator.asr_engine.preprocessor.augment_waveform(w, noise_level=0.01)
    ))

    # 3. Additive Background Noise (Medium, sigma=0.05)
    print("[3/5] Testing Medium Background Noise (sigma=0.05)...")
    results.append(evaluate_condition(
        translator, dataset, "3. Medium Noise (sigma=0.05)",
        lambda w: translator.asr_engine.preprocessor.augment_waveform(w, noise_level=0.05)
    ))

    # 4. Low-Volume Speech (-6 dB)
    print("[4/5] Testing Low Volume Speech (-6 dB)...")
    results.append(evaluate_condition(
        translator, dataset, "4. Low Volume (-6 dB)",
        lambda w: translator.asr_engine.preprocessor.augment_waveform(w, noise_level=0.0, gain_db=-6.0)
    ))

    # 5. High-Volume Speech (+3 dB)
    print("[5/5] Testing High Volume Speech (+3 dB)...")
    results.append(evaluate_condition(
        translator, dataset, "5. High Volume (+3 dB)",
        lambda w: translator.asr_engine.preprocessor.augment_waveform(w, noise_level=0.0, gain_db=3.0)
    ))

    print("\n" + "=" * 70)
    print("                    🏆  ROBUSTNESS BENCHMARK RESULTS  🏆")
    print("=" * 70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("=" * 70)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Robustness results saved to '{args.output_json}'.")

if __name__ == "__main__":
    main()
