import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional
from src.pipeline.voice_translator import NepaliVoiceTranslator

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total, trainable, and non-trainable parameters of a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable
    }

def get_checkpoint_size_mb(path: str) -> float:
    """Returns file size in Megabytes."""
    if os.path.exists(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    return 0.0

def profile_system_parameters(translator: NepaliVoiceTranslator) -> Dict[str, Any]:
    """Analyzes parameter counts and disk footprint for all subsystems."""
    asr_stats = count_parameters(translator.asr_engine.model)
    nmt_stats = count_parameters(translator.translation_engine.model)

    asr_ckpt_mb = get_checkpoint_size_mb(translator.asr_checkpoint)
    nmt_ckpt_mb = get_checkpoint_size_mb(translator.translation_checkpoint)

    total_params = asr_stats["total_params"] + nmt_stats["total_params"]
    trainable_params = asr_stats["trainable_params"] + nmt_stats["trainable_params"]

    return {
        "asr_model": {
            **asr_stats,
            "checkpoint_path": translator.asr_checkpoint,
            "disk_size_mb": asr_ckpt_mb
        },
        "translation_model": {
            **nmt_stats,
            "checkpoint_path": translator.translation_checkpoint,
            "disk_size_mb": nmt_ckpt_mb
        },
        "combined_system": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "total_disk_size_mb": round(asr_ckpt_mb + nmt_ckpt_mb, 2)
        }
    }

def benchmark_inference_latency(
    translator: NepaliVoiceTranslator,
    audio_paths: List[str],
    num_runs: int = 5,
    warmup_runs: int = 2
) -> Dict[str, Any]:
    """
    Measures multi-run latency statistics (mean, median, p95, min, max, std)
    and Real-Time Factor (RTF) across multiple audio samples.
    """
    # Warmup
    if audio_paths:
        for _ in range(warmup_runs):
            translator.translate_audio(audio_paths[0])

    all_latencies = []
    all_asr_latencies = []
    all_nmt_latencies = []
    all_rtfs = []

    for path in audio_paths:
        for _ in range(num_runs):
            res = translator.translate_audio(path)
            all_latencies.append(res["total_latency_sec"])
            all_asr_latencies.append(res["asr_latency_sec"])
            all_nmt_latencies.append(res["translation_latency_sec"])
            all_rtfs.append(res["real_time_factor"])

    lat_arr = np.array(all_latencies)
    asr_arr = np.array(all_asr_latencies)
    nmt_arr = np.array(all_nmt_latencies)
    rtf_arr = np.array(all_rtfs)

    return {
        "num_samples_evaluated": len(audio_paths),
        "total_executions": len(lat_arr),
        "total_latency_sec": {
            "mean": round(float(np.mean(lat_arr)), 4),
            "median": round(float(np.median(lat_arr)), 4),
            "std": round(float(np.std(lat_arr)), 4),
            "min": round(float(np.min(lat_arr)), 4),
            "max": round(float(np.max(lat_arr)), 4),
            "p95": round(float(np.percentile(lat_arr, 95)), 4)
        },
        "asr_latency_sec": {
            "mean": round(float(np.mean(asr_arr)), 4),
            "median": round(float(np.median(asr_arr)), 4)
        },
        "translation_latency_sec": {
            "mean": round(float(np.mean(nmt_arr)), 4),
            "median": round(float(np.median(nmt_arr)), 4)
        },
        "real_time_factor": {
            "mean": round(float(np.mean(rtf_arr)), 4),
            "median": round(float(np.median(rtf_arr)), 4),
            "is_realtime": bool(np.mean(rtf_arr) < 1.0)
        }
    }
