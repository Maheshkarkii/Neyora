import os
import pytest
import numpy as np
import soundfile as sf
import pandas as pd
from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.evaluation.end_to_end import EndToEndEvaluator

@pytest.fixture(scope="module")
def evaluator():
    translator = NepaliVoiceTranslator(device="cpu")
    return EndToEndEvaluator(translator=translator)

@pytest.fixture
def synthetic_dataset(tmp_path):
    dataset = []
    sr = 16000
    sentences = [
        ("हामी सबै मिलेर काम गर्नुपर्छ।", "we must all work together ."),
        ("धन्यवाद तपाईंको सहयोगको लागि।", "thank you for your help .")
    ]
    for idx, (nep, eng) in enumerate(sentences):
        audio_p = str(tmp_path / f"synth_{idx}.wav")
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        waveform = 0.3 * np.sin(2 * np.pi * 300 * (idx + 1) * t)
        sf.write(audio_p, waveform, sr)
        dataset.append({
            "audio_path": audio_p,
            "ground_truth_nepali": nep,
            "ground_truth_english": eng
        })
    return dataset

def test_evaluate_sample(evaluator, synthetic_dataset):
    sample = synthetic_dataset[0]
    res = evaluator.evaluate_sample(
        audio_path=sample["audio_path"],
        ground_truth_nepali=sample["ground_truth_nepali"],
        ground_truth_english=sample["ground_truth_english"]
    )
    assert "mode1_english" in res
    assert "predicted_nepali" in res
    assert "predicted_english" in res
    assert "cer" in res
    assert "wer" in res
    assert "mode1_bleu" in res
    assert "e2e_bleu" in res
    assert "error_classification" in res

def test_evaluate_dataset(evaluator, synthetic_dataset, tmp_path):
    csv_out = str(tmp_path / "test_results.csv")
    df, summary = evaluator.evaluate_dataset(synthetic_dataset, output_csv_path=csv_out)

    assert os.path.exists(csv_out)
    assert len(df) == len(synthetic_dataset)
    assert "asr_avg_cer" in summary
    assert "mode1_gt_bleu" in summary
    assert "mode2_e2e_bleu" in summary
    assert summary["num_samples"] == len(synthetic_dataset)
