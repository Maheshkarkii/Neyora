import pytest
import torch
import pandas as pd
from src.data.translation_dataset import TranslationDataset
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer

def test_translation_dataset_item(tmp_path):
    csv_path = str(tmp_path / "nmt.csv")
    df = pd.DataFrame([{
        "nepali_text": "म कलेज जान्छु ।",
        "english_text": "i go to college ."
    }])
    df.to_csv(csv_path, index=False)

    src_vocab = TranslationVocabulary()
    tgt_vocab = TranslationVocabulary()
    src_vocab.build_from_texts(["म कलेज जान्छु ।"])
    tgt_vocab.build_from_texts(["i go to college ."])

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    dataset = TranslationDataset(csv_path, src_tok, tgt_tok)
    assert len(dataset) == 1

    item = dataset[0]
    assert item["src_ids"].ndim == 1
    assert item["tgt_ids"].ndim == 1
    assert item["src_ids"][0] == src_vocab.sos_idx
    assert item["src_ids"][-1] == src_vocab.eos_idx
    assert item["tgt_ids"][0] == tgt_vocab.sos_idx
    assert item["tgt_ids"][-1] == tgt_vocab.eos_idx
