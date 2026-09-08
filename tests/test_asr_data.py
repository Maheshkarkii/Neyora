import pytest
import torch
import os
from src.asr.text_cleaner import DevanagariTextCleaner
from src.asr.tokenizer import ASRTokenizer
from src.asr.audio_transforms import AudioPreprocessor
from src.asr.download_openslr54 import generate_dev_audio_sample
from src.asr.dataset import NepaliASRDataset, ASRCollateFn

def test_devanagari_cleaner():
    cleaner = DevanagariTextCleaner()
    raw = "  नमस्ते !  तपाईँको नाम के हो??  "
    cleaned = cleaner.normalize(raw)
    assert "नमस्ते" in cleaned
    assert "नाम" in cleaned
    assert cleaner.is_valid_nepali(cleaned)
    assert not cleaner.is_valid_nepali("Hello world 12345")

def test_asr_tokenizer():
    tokenizer = ASRTokenizer()
    sample_texts = ["नमस्ते नेपाल", "काठमाडौं राजधानी हो"]
    tokenizer.build_vocab_from_texts(sample_texts)
    
    assert tokenizer.blank_id == 0
    assert tokenizer.pad_id == 1
    assert tokenizer.unk_id == 2
    
    encoded = tokenizer.encode("नमस्ते नेपाल")
    decoded = tokenizer.decode(encoded)
    assert "नमस्ते" in decoded
    assert "नेपाल" in decoded

def test_audio_preprocessor(tmp_path):
    wav_path = str(tmp_path / "test.wav")
    generate_dev_audio_sample(wav_path, duration_sec=1.0, sample_rate=16000)
    
    preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)
    waveform, sr = preprocessor.load_audio(wav_path)
    assert sr == 16000
    assert waveform.ndim == 2
    assert waveform.shape[0] == 1
    
    mel_spec = preprocessor.extract_features(waveform)
    assert mel_spec.ndim == 2
    assert mel_spec.shape[0] == 80
    assert mel_spec.shape[1] > 0

def test_asr_collate_fn():
    collate_fn = ASRCollateFn(pad_id=1)
    sample_batch = [
        {"mel_spec": torch.randn(80, 100), "tokens": torch.tensor([4, 5, 6]), "text": "नम", "duration": 1.0, "audio_path": "a.wav"},
        {"mel_spec": torch.randn(80, 150), "tokens": torch.tensor([7, 8, 9, 10, 11]), "text": "नेपाल", "duration": 1.5, "audio_path": "b.wav"}
    ]
    padded_specs, padded_targets, in_lens, tgt_lens, texts = collate_fn(sample_batch)
    
    assert padded_specs.shape == (2, 80, 150)
    assert padded_targets.shape == (2, 5)
    assert torch.equal(in_lens, torch.tensor([100, 150]))
    assert torch.equal(tgt_lens, torch.tensor([3, 5]))
    assert padded_targets[0, 3].item() == 1
