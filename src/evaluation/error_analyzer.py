import os
import io
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import Counter
from src.evaluation.asr_metrics import compute_cer, compute_wer
from src.evaluation.metrics import calculate_sentence_bleu
from src.utils.logger import get_logger

logger = get_logger("error_analyzer")

def classify_asr_error(ground_truth: str, hypothesis: str) -> Dict[str, Any]:
    """
    Classifies ASR errors into character/word substitutions, insertions, and deletions.
    """
    ref_words = ground_truth.strip().split()
    hyp_words = hypothesis.strip().split()
    ref_chars = list(ground_truth.strip().replace(" ", ""))
    hyp_chars = list(hypothesis.strip().replace(" ", ""))

    # Word-level counts
    ref_w_set = Counter(ref_words)
    hyp_w_set = Counter(hyp_words)

    word_deletions = sum(max(0, count - hyp_w_set[w]) for w, count in ref_w_set.items())
    word_insertions = sum(max(0, count - ref_w_set[w]) for w, count in hyp_w_set.items())
    word_substitutions = min(word_deletions, word_insertions)
    net_deletions = word_deletions - word_substitutions
    net_insertions = word_insertions - word_substitutions

    # Character-level counts
    ref_c_set = Counter(ref_chars)
    hyp_c_set = Counter(hyp_chars)
    char_deletions = sum(max(0, count - hyp_c_set[c]) for c, count in ref_c_set.items())
    char_insertions = sum(max(0, count - ref_c_set[c]) for c, count in hyp_c_set.items())
    char_substitutions = min(char_deletions, char_insertions)

    primary_error = "None (Exact Match)"
    if ground_truth.strip() != hypothesis.strip():
        if word_substitutions > 0:
            primary_error = "word_substitution"
        elif net_deletions > 0:
            primary_error = "word_deletion"
        elif net_insertions > 0:
            primary_error = "word_insertion"
        elif char_substitutions > 0:
            primary_error = "char_substitution"
        elif char_deletions > 0:
            primary_error = "char_deletion"
        elif char_insertions > 0:
            primary_error = "char_insertion"

    return {
        "primary_asr_error": primary_error,
        "char_substitutions": char_substitutions,
        "char_deletions": char_deletions,
        "char_insertions": char_insertions,
        "word_substitutions": word_substitutions,
        "word_deletions": net_deletions,
        "word_insertions": net_insertions
    }

def classify_translation_error(
    ground_truth_eng: str,
    predicted_eng: str,
    src_nepali: str
) -> Dict[str, Any]:
    """
    Classifies translation errors:
    wrong word, missing word, extra word, word-order error, grammar error,
    unknown-word problem, long-sentence failure.
    """
    ref_tokens = ground_truth_eng.strip().lower().split()
    hyp_tokens = predicted_eng.strip().lower().split()

    if not hyp_tokens or hyp_tokens == ["."]:
        return {"primary_nmt_error": "missing_translation"}

    if "<unk>" in hyp_tokens or "<UNK>" in hyp_tokens:
        return {"primary_nmt_error": "unknown_word_problem"}

    if len(src_nepali.split()) >= 8 and len(hyp_tokens) < len(ref_tokens) // 2:
        return {"primary_nmt_error": "long_sentence_failure"}

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)

    missing_words = sum(max(0, count - hyp_counts[w]) for w, count in ref_counts.items())
    extra_words = sum(max(0, count - ref_counts[w]) for w, count in hyp_counts.items())

    # Check repetitive token loops
    most_common_token, max_freq = hyp_counts.most_common(1)[0]
    if max_freq >= 3 and len(hyp_tokens) >= 3 and (max_freq / len(hyp_tokens)) >= 0.5:
        return {"primary_nmt_error": "repetitive_token_loop"}

    if missing_words > 0 and extra_words > 0:
        return {"primary_nmt_error": "wrong_word"}
    elif missing_words > 0:
        return {"primary_nmt_error": "missing_word"}
    elif extra_words > 0:
        return {"primary_nmt_error": "extra_word"}

    if set(ref_tokens) == set(hyp_tokens) and ref_tokens != hyp_tokens:
        return {"primary_nmt_error": "word_order_error"}

    if ground_truth_eng.strip().lower() == predicted_eng.strip().lower():
        return {"primary_nmt_error": "None (Exact Match)"}

    return {"primary_nmt_error": "grammar_error"}


def run_systematic_error_analysis(
    evaluation_df: pd.DataFrame,
    output_csv_path: str = "results/error_analysis.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs full error classification and stratified breakdown by:
    audio duration, speaker, transcript length, and speech speed.
    """
    analyzed_rows = []

    for _, row in evaluation_df.iterrows():
        gt_nep = str(row["ground_truth_nepali"])
        pred_nep = str(row["predicted_nepali"])
        gt_eng = str(row["ground_truth_english"])
        pred_eng = str(row["predicted_english"])
        duration = float(row.get("audio_duration_sec", 1.0))
        audio_path = str(row["audio_path"])

        # Extract speaker id if available in audio filename
        speaker = "unknown"
        filename = os.path.basename(audio_path)
        if "spk_" in filename:
            parts = filename.split("_")
            for i, p in enumerate(parts):
                if p == "spk" and i + 1 < len(parts):
                    speaker = f"spk_{parts[i+1]}"

        # Speech speed
        nep_word_count = len(gt_nep.split())
        words_per_sec = nep_word_count / max(0.1, duration)

        duration_bin = "short (<2s)" if duration < 2.0 else ("medium (2-5s)" if duration <= 5.0 else "long (>5s)")
        length_bin = "short (<4 words)" if nep_word_count < 4 else ("medium (4-7 words)" if nep_word_count <= 7 else "long (>7 words)")

        asr_err = classify_asr_error(gt_nep, pred_nep)
        nmt_err = classify_translation_error(gt_eng, pred_eng, gt_nep)

        combined_entry = {
            "audio_path": audio_path,
            "speaker": speaker,
            "duration_sec": duration,
            "duration_bin": duration_bin,
            "transcript_words": nep_word_count,
            "length_bin": length_bin,
            "speech_speed_wps": round(words_per_sec, 2),
            "ground_truth_nepali": gt_nep,
            "predicted_nepali": pred_nep,
            "ground_truth_english": gt_eng,
            "predicted_english": pred_eng,
            "cer": row.get("cer", compute_cer(gt_nep, pred_nep)),
            "wer": row.get("wer", compute_wer(gt_nep, pred_nep)),
            "e2e_bleu": row.get("e2e_bleu", 0.0),
            **asr_err,
            **nmt_err
        }
        analyzed_rows.append(combined_entry)

    df_res = pd.DataFrame(analyzed_rows)
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    df_res.to_csv(output_csv_path, index=False, encoding="utf-8")

    # Aggregate summaries
    summary = {
        "total_samples": len(df_res),
        "asr_error_distribution": df_res["primary_asr_error"].value_counts().to_dict(),
        "nmt_error_distribution": df_res["primary_nmt_error"].value_counts().to_dict(),
        "performance_by_duration": df_res.groupby("duration_bin")[["cer", "wer", "e2e_bleu"]].mean().to_dict(),
        "performance_by_length": df_res.groupby("length_bin")[["cer", "wer", "e2e_bleu"]].mean().to_dict(),
        "performance_by_speaker": df_res.groupby("speaker")[["cer", "wer", "e2e_bleu"]].mean().to_dict()
    }

    return df_res, summary
