import pytest
from src.data.vocabulary import ASRVocabulary, TranslationVocabulary
from src.data.tokenizer import ASRTokenizer, TranslationTokenizer

def test_asr_tokenizer_roundtrip():
    vocab = ASRVocabulary()
    test_phrase = "म आज कलेज जाँदै छु।"
    vocab.build_from_texts([test_phrase])
    tokenizer = ASRTokenizer(vocab)

    encoded = tokenizer.encode(test_phrase)
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    decoded = tokenizer.decode(encoded, remove_special=True)
    assert "म" in decoded
    assert "कलेज" in decoded
    assert "छु।" in decoded

def test_translation_tokenizer_special_tokens():
    vocab = TranslationVocabulary()
    texts = ["i go to college", "hello world"]
    vocab.build_from_texts(texts)
    tokenizer = TranslationTokenizer(vocab, is_nepali=False)

    encoded = tokenizer.encode("i go to college", add_special_tokens=True)
    assert encoded[0] == vocab.sos_idx
    assert encoded[-1] == vocab.eos_idx

    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert decoded == "i go to college"

def test_nepali_translation_tokenizer():
    vocab = TranslationVocabulary()
    texts = ["म कलेज जान्छु ।", "आज मौसम राम्रो छ ।"]
    vocab.build_from_texts(texts)
    tokenizer = TranslationTokenizer(vocab, is_nepali=True)

    encoded = tokenizer.encode("म कलेज जान्छु ।", add_special_tokens=True)
    assert encoded[0] == vocab.sos_idx
    assert encoded[-1] == vocab.eos_idx
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    assert "म" in decoded
    assert "कलेज" in decoded
