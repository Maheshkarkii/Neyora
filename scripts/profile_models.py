import os
import sys
import json
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.utils.model_profiler import profile_system_parameters, benchmark_inference_latency
from scripts.evaluate_end_to_end import get_end_to_end_test_dataset

def main():
    parser = argparse.ArgumentParser(description="Model Parameters and Inference Profiling CLI")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default="results/model_profile.json")
    args = parser.parse_args()

    print("=" * 70)
    print("        📊  PHASE 6: MODEL PROFILING & PARAMETER ANALYSIS  📊")
    print("=" * 70)

    translator = NepaliVoiceTranslator(device=args.device)

    # 1. Parameter Analysis
    param_profile = profile_system_parameters(translator)

    print("\n--- Model Architecture & Parameter Counts ---")
    asr_p = param_profile["asr_model"]
    print(f"ASR Model (CNN-BiLSTM-CTC):")
    print(f"  • Total Parameters        : {asr_p['total_params']:,}")
    print(f"  • Trainable Parameters    : {asr_p['trainable_params']:,}")
    print(f"  • Non-trainable Parameters: {asr_p['non_trainable_params']:,}")
    print(f"  • Checkpoint Disk Size    : {asr_p['disk_size_mb']:.2f} MB")

    nmt_p = param_profile["translation_model"]
    print(f"\nTranslation Model (Seq2Seq-Attention):")
    print(f"  • Total Parameters        : {nmt_p['total_params']:,}")
    print(f"  • Trainable Parameters    : {nmt_p['trainable_params']:,}")
    print(f"  • Non-trainable Parameters: {nmt_p['non_trainable_params']:,}")
    print(f"  • Checkpoint Disk Size    : {nmt_p['disk_size_mb']:.2f} MB")

    sys_p = param_profile["combined_system"]
    print(f"\nCombined Pipeline:")
    print(f"  • Total Parameters        : {sys_p['total_params']:,}")
    print(f"  • Total Disk Size         : {sys_p['total_disk_size_mb']:.2f} MB")

    # 2. Multi-sample Inference Latency Benchmarking
    dataset = get_end_to_end_test_dataset()
    audio_paths = [item["audio_path"] for item in dataset[:10]]
    
    print("\nBenchmarking multi-run inference latency across 10 samples (5 runs each)...")
    latency_profile = benchmark_inference_latency(translator, audio_paths, num_runs=5, warmup_runs=2)

    tot_lat = latency_profile["total_latency_sec"]
    print("\n--- Latency Breakdown Statistics ---")
    print(f"  • Mean Pipeline Latency   : {tot_lat['mean']:.4f} s (Std: {tot_lat['std']:.4f} s)")
    print(f"  • Median Pipeline Latency : {tot_lat['median']:.4f} s")
    print(f"  • P95 Pipeline Latency    : {tot_lat['p95']:.4f} s")
    print(f"  • Min / Max Latency       : {tot_lat['min']:.4f} s / {tot_lat['max']:.4f} s")
    print(f"  • Mean ASR Latency        : {latency_profile['asr_latency_sec']['mean']:.4f} s")
    print(f"  • Mean Translation Latency: {latency_profile['translation_latency_sec']['mean']:.4f} s")
    print(f"  • Mean Real-Time Factor   : {latency_profile['real_time_factor']['mean']:.4f} (Real-time: {latency_profile['real_time_factor']['is_realtime']})")

    combined_data = {
        "parameters": param_profile,
        "latency_profile": latency_profile
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)

    print(f"\nFull profiling results saved to '{args.output_json}'.")

if __name__ == "__main__":
    main()
