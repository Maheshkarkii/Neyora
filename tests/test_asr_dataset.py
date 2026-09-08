import pytest
import os
import torch
import pandas as pd
from src.data.asr_dataset import ASRDataset
from src.data.preprocessing import AudioPreprocessor
from src.data.vocabulary import ASRVocabulary
from src.data.tokenizer import ASRTokenizer
from scripts.prepare_asr_data import synthesize_acoustic_sample

def test_asr_dataset_item(tmp_path):
    wav_path = str(tmp_path / "audio.wav")
    synthesize_acoustic_sample(wav_path, duration=1.5, sample_rate=16000)

    csv_path = str(tmp_path / "asr.csv")
    df = pd.DataFrame([{
        "audio_path": wav_path,
        "transcription": "म कलेज जान्छु।",
        "speaker_id": "spk_1",
        "duration": 1.5
    }])
    df.to_csv(csv_path, index=False)

    vocab = ASRVocabulary()
    vocab.build_from_texts(["म कलेज जान्छु।"])
    tokenizer = ASRTokenizer(vocab)
    preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)

    dataset = ASRDataset(csv_path, preprocessor, tokenizer)
    assert len(dataset) == 1

    item = dataset[0]
    assert "mel_spec" in item
    assert "tokens" in item
    assert item["mel_spec"].shape[0] == 80 # [n_mels, time]
    assert item["mel_spec"].dtype == torch.float32
    assert item["tokens"].dtype == torch.long
