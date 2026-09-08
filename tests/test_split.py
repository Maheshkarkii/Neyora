import pytest
import os
import pandas as pd
from src.data.split import create_asr_splits, create_translation_splits

def test_asr_speaker_disjoint_split(tmp_path):
    out_dir = str(tmp_path / "meta")
    records = [
        {"audio_path": "a.wav", "transcription": "म", "speaker_id": "spk_1", "duration": 1.0},
        {"audio_path": "b.wav", "transcription": "क", "speaker_id": "spk_1", "duration": 1.2},
        {"audio_path": "c.wav", "transcription": "ख", "speaker_id": "spk_2", "duration": 1.5},
        {"audio_path": "d.wav", "transcription": "ग", "speaker_id": "spk_3", "duration": 2.0},
        {"audio_path": "e.wav", "transcription": "घ", "speaker_id": "spk_4", "duration": 2.2},
    ]
    splits = create_asr_splits(records, output_dir=out_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    train_spks = set(splits["train"]["speaker_id"])
    test_spks = set(splits["test"]["speaker_id"])
    # Zero speaker overlap
    assert len(train_spks.intersection(test_spks)) == 0
    assert os.path.exists(os.path.join(out_dir, "asr_train.csv"))

def test_translation_splits_deduplication(tmp_path):
    out_dir = str(tmp_path / "meta_nmt")
    pairs = [
        {"nepali_text": "नमस्ते", "english_text": "hello"},
        {"nepali_text": "नमस्ते", "english_text": "hello"}, # Duplicate
        {"nepali_text": "धन्यवाद", "english_text": "thank you"},
        {"nepali_text": "काठमाडौं", "english_text": "kathmandu"}
    ]
    splits = create_translation_splits(pairs, output_dir=out_dir, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42)
    total_split_rows = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total_split_rows == 3 # Deduplicated from 4 to 3
