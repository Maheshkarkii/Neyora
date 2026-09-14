import os
import pytest
import torch
import numpy as np
import soundfile as sf
import tempfile
from src.pipeline.voice_translator import NepaliVoiceTranslator
from src.utils.model_profiler import profile_system_parameters, benchmark_inference_latency

@pytest.fixture(scope="module")
def translator():
    return NepaliVoiceTranslator(device="cpu")

@pytest.fixture
def sample_audio(tmp_path):
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)
    file_path = str(tmp_path / "test_tone.wav")
    sf.write(file_path, waveform, sr)
    return file_path

@pytest.fixture
def long_sample_audio(tmp_path):
    sr = 16000
    # 12 seconds audio
    t = np.linspace(0, 12.0, int(sr * 12.0), endpoint=False, dtype=np.float32)
    waveform = 0.4 * np.sin(2 * np.pi * 440 * t)
    file_path = str(tmp_path / "long_tone.wav")
    sf.write(file_path, waveform, sr)
    return file_path

@pytest.fixture
def silence_audio(tmp_path):
    sr = 16000
    waveform = np.zeros(sr, dtype=np.float32)
    file_path = str(tmp_path / "silence.wav")
    sf.write(file_path, waveform, sr)
    return file_path

def test_translator_initialization(translator):
    assert translator.asr_engine is not None
    assert translator.translation_engine is not None
    assert translator.device == torch.device("cpu")

def test_translate_text_mode1(translator):
    text = "हामी सबै मिलेर काम गर्नुपर्छ।"
    res = translator.translate_text(text)
    assert "translation" in res
    assert isinstance(res["translation"], str)
    assert len(res["translation"]) > 0
    assert "latency_sec" in res
    assert res["latency_sec"] > 0

def test_translate_text_beam_search(translator):
    text = "हामी सबै मिलेर काम गर्नुपर्छ।"
    res = translator.translate_text(text, decoder_type="beam_search")
    assert "translation" in res
    assert isinstance(res["translation"], str)
    assert res["decoder_type"] == "beam_search"

def test_translate_audio_pipeline(translator, sample_audio):
    res = translator.translate_audio(sample_audio, diagnostic=True)
    assert "transcription" in res
    assert "translation" in res
    assert "audio_duration_sec" in res
    assert "asr_latency_sec" in res
    assert "translation_latency_sec" in res
    assert "total_latency_sec" in res
    assert "real_time_factor" in res
    assert "diagnostic" in res
    assert res["audio_duration_sec"] > 0.9

def test_translate_audio_beam_search(translator, sample_audio):
    res = translator.translate_audio(
        sample_audio,
        asr_decoder_type="beam_search",
        translation_decoder_type="beam_search"
    )
    assert "transcription" in res
    assert "translation" in res
    assert res["asr_decoder_used"] == "beam_search"
    assert res["translation_decoder_used"] == "beam_search"

def test_translate_silence_audio(translator, silence_audio):
    res = translator.translate_audio(silence_audio)
    assert res["transcription"] == ""
    assert res["translation"] == ""
    assert res["status"] == "silence_or_empty_audio"

def test_translate_long_audio_chunking(translator, long_sample_audio):
    res = translator.translate_long_audio(
        long_sample_audio,
        chunk_duration_sec=5.0,
        overlap_sec=1.0
    )
    assert "num_chunks" in res
    assert res["num_chunks"] >= 2
    assert "chunk_transcriptions" in res
    assert res["audio_duration_sec"] >= 11.9

def test_model_profiler(translator, sample_audio):
    profile = profile_system_parameters(translator)
    assert "asr_model" in profile
    assert profile["asr_model"]["total_params"] > 1_000_000
    assert "translation_model" in profile
    assert profile["translation_model"]["total_params"] > 1_000_000
    assert profile["combined_system"]["total_params"] > 5_000_000

    bench = benchmark_inference_latency(translator, [sample_audio], num_runs=2, warmup_runs=1)
    assert "total_latency_sec" in bench
    assert bench["total_latency_sec"]["mean"] > 0

def test_error_propagation_analysis():
    diag_correct = NepaliVoiceTranslator.analyze_error_propagation(
        ground_truth_nepali="हामी सबै मिलेर काम गर्नुपर्छ।",
        predicted_nepali="हामी सबै मिलेर काम गर्नुपर्छ।",
        ground_truth_english="we must all work together .",
        predicted_english="we must all work together ."
    )
    assert diag_correct["classification"] == "Correct"

    diag_asr_error = NepaliVoiceTranslator.analyze_error_propagation(
        ground_truth_nepali="हामी सबै मिलेर काम गर्नुपर्छ।",
        predicted_nepali="काठमाडौं नेपाल",
        ground_truth_english="we must all work together .",
        predicted_english="kathmandu nepal",
        standalone_english_from_gt="we must all work together ."
    )
    assert diag_asr_error["classification"] == "ASR error"

def test_invalid_audio_path(translator):
    with pytest.raises(FileNotFoundError):
        translator.translate_audio("non_existent_file_path_12345.wav")
