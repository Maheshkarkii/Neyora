import pytest
import torch
from src.data.preprocessing import NepaliTextCleaner, EnglishTextCleaner, AudioPreprocessor, validate_audio_file, validate_parallel_pair
from scripts.prepare_asr_data import synthesize_acoustic_sample

def test_nepali_cleaner():
    cleaner = NepaliTextCleaner()
    raw = "  म आज  कलेज जाँदै छु!!  "
    cleaned = cleaner.normalize(raw)
    assert "कलेज" in cleaned
    assert "जाँदै" in cleaned
    assert cleaner.is_valid_nepali(cleaned)
    assert not cleaner.is_valid_nepali("Just english text 123")

def test_english_cleaner():
    cleaner = EnglishTextCleaner()
    raw = "  I'M GOING to   college!  "
    cleaned = cleaner.normalize(raw)
    assert "i'm going to college !" in cleaned

def test_audio_preprocessor_mel_shape(tmp_path):
    wav_path = str(tmp_path / "sample.wav")
    synthesize_acoustic_sample(wav_path, duration=1.0, sample_rate=16000)
    
    preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)
    waveform, sr = preprocessor.load_audio(wav_path)
    assert sr == 16000
    assert waveform.shape[0] == 1 # Mono
    
    mel_spec = preprocessor.extract_mel_spectrogram(waveform)
    assert mel_spec.shape[0] == 80 # [n_mels, time]
    assert mel_spec.ndim == 2

def test_parallel_pair_validation():
    is_valid, _ = validate_parallel_pair("म कलेज जान्छु ।", "I go to college .")
    assert is_valid
    is_invalid, reason = validate_parallel_pair("", "Empty")
    assert not is_invalid
