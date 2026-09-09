import os
import io
import sys
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.evaluation.asr_metrics import compute_cer, compute_wer, compute_corpus_cer, compute_corpus_wer
from src.evaluation.metrics import calculate_sentence_bleu
from src.utils.logger import get_logger

logger = get_logger("end_to_end_evaluation")

class EndToEndEvaluator:
    """
    Evaluator for comparing 3 translation modes:
    - Mode 1: Ground Truth Nepali Text -> Translation Model -> English (Independent Translation)
    - Mode 2: Audio -> ASR Prediction -> Translation Model -> English (Cascaded Real Pipeline)
    - Mode 3: End-to-End Evaluation (Mode 2 predictions vs Ground Truth English reference)
    """
    def __init__(self, translator: Optional[NepaliVoiceTranslator] = None):
        self.translator = translator or NepaliVoiceTranslator()

    def evaluate_sample(
        self,
        audio_path: str,
        ground_truth_nepali: str,
        ground_truth_english: str
    ) -> Dict[str, Any]:
        """Runs evaluation for a single audio sample across all 3 modes."""
        # Mode 1: Direct translation from Ground-truth Nepali
        mode1_res = self.translator.translate_text(ground_truth_nepali)
        mode1_predicted_english = mode1_res["translation"]

        # Mode 2 & 3: Audio -> ASR -> Translation
        e2e_res = self.translator.translate_audio(audio_path, diagnostic=True)
        predicted_nepali = e2e_res["transcription"]
        predicted_english = e2e_res["translation"]

        # Metric computations
        cer = compute_cer(ground_truth_nepali, predicted_nepali)
        wer = compute_wer(ground_truth_nepali, predicted_nepali)

        gt_tokens = ground_truth_english.strip().lower().split()
        mode1_tokens = mode1_predicted_english.strip().lower().split()
        pred_tokens = predicted_english.strip().lower().split()

        mode1_bleu = calculate_sentence_bleu(gt_tokens, mode1_tokens) * 100.0
        e2e_bleu = calculate_sentence_bleu(gt_tokens, pred_tokens) * 100.0

        # Error diagnosis
        diag = self.translator.analyze_error_propagation(
            ground_truth_nepali=ground_truth_nepali,
            predicted_nepali=predicted_nepali,
            ground_truth_english=ground_truth_english,
            predicted_english=predicted_english,
            standalone_english_from_gt=mode1_predicted_english
        )

        return {
            "audio_path": audio_path,
            "ground_truth_nepali": ground_truth_nepali,
            "predicted_nepali": predicted_nepali,
            "ground_truth_english": ground_truth_english,
            "mode1_english": mode1_predicted_english,
            "predicted_english": predicted_english,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "mode1_bleu": round(mode1_bleu, 2),
            "e2e_bleu": round(e2e_bleu, 2),
            "error_classification": diag["classification"],
            "error_reason": diag["reason"],
            "audio_duration_sec": e2e_res["audio_duration_sec"],
            "asr_latency_sec": e2e_res["asr_latency_sec"],
            "translation_latency_sec": e2e_res["translation_latency_sec"],
            "total_latency_sec": e2e_res["total_latency_sec"],
            "real_time_factor": e2e_res["real_time_factor"],
            "asr_confidence": e2e_res["asr_confidence"]
        }

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, str]],
        output_csv_path: str = "results/end_to_end_results.csv"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluates a complete test dataset across all 3 modes and generates full metrics.
        """
        results = []
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

        for i, item in enumerate(dataset):
            audio_path = item["audio_path"]
            gt_nep = item["ground_truth_nepali"]
            gt_eng = item["ground_truth_english"]
            logger.info(f"Evaluating sample {i+1}/{len(dataset)}: {os.path.basename(audio_path)}")
            sample_eval = self.evaluate_sample(audio_path, gt_nep, gt_eng)
            results.append(sample_eval)

        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False, encoding="utf-8")
        logger.info(f"Evaluation results saved to {output_csv_path}")

        # Summary Metrics
        avg_cer = float(df["cer"].mean())
        avg_wer = float(df["wer"].mean())
        avg_mode1_bleu = float(df["mode1_bleu"].mean())
        avg_e2e_bleu = float(df["e2e_bleu"].mean())
        avg_rtf = float(df["real_time_factor"].mean())
        avg_latency = float(df["total_latency_sec"].mean())

        error_counts = df["error_classification"].value_counts().to_dict()

        summary = {
            "num_samples": len(df),
            "asr_avg_cer": round(avg_cer, 4),
            "asr_avg_wer": round(avg_wer, 4),
            "mode1_gt_bleu": round(avg_mode1_bleu, 2),
            "mode2_e2e_bleu": round(avg_e2e_bleu, 2),
            "avg_real_time_factor": round(avg_rtf, 4),
            "avg_total_latency_sec": round(avg_latency, 4),
            "error_distribution": error_counts
        }

        return df, summary
