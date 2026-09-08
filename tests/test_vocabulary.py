import pytest
from src.data.vocabulary import ASRVocabulary, TranslationVocabulary

def test_asr_vocabulary_special_tokens():
    vocab = ASRVocabulary()
    assert vocab.blank_idx == 0
    assert vocab.pad_idx == 1
    assert vocab.unk_idx == 2
    assert vocab.space_idx == 3
    assert len(vocab) == 4

def test_asr_vocabulary_devanagari_build():
    vocab = ASRVocabulary()
    sample_texts = ["म आज कलेज जाँदै छु।", "नेपाल एक सुन्दर देश हो।"]
    vocab.build_from_texts(sample_texts)
    assert "म" in vocab.char2idx
    assert "क" in vocab.char2idx
    assert "।" in vocab.char2idx
    assert len(vocab) > 4

def test_translation_vocabulary_special_tokens():
    vocab = TranslationVocabulary()
    assert vocab.pad_idx == 0
    assert vocab.unk_idx == 1
    assert vocab.sos_idx == 2
    assert vocab.eos_idx == 3
    assert len(vocab) == 4

def test_translation_vocabulary_build():
    vocab = TranslationVocabulary()
    texts = ["i go to college", "nepal is beautiful country"]
    vocab.build_from_texts(texts, min_freq=1, max_size=100)
    assert "college" in vocab.token2idx
    assert "nepal" in vocab.token2idx
