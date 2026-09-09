import os
import io
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def wav_bytes():
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    waveform = 0.4 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    buf.seek(0)
    return buf.read()

def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["asr_model_loaded"] is True
    assert data["translation_model_loaded"] is True

def test_api_serve_ui(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Nepali Voice Translator" in response.text

def test_api_translate_valid_wav(client, wav_bytes):
    files = {"file": ("test_sample.wav", wav_bytes, "audio/wav")}
    response = client.post("/translate", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "transcription" in data
    assert "translation" in data
    assert "audio_duration_sec" in data
    assert "latency_sec" in data
    assert "real_time_factor" in data
    assert data["audio_duration_sec"] > 0.8

def test_api_translate_empty_file(client):
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/translate", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_api_translate_unsupported_extension(client):
    files = {"file": ("doc.txt", b"Hello World", "text/plain")}
    response = client.post("/translate", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()

def test_api_translate_corrupted_audio(client):
    files = {"file": ("corrupt.wav", b"INVALID_WAV_HEADER_DATA_ABCXYZ", "audio/wav")}
    response = client.post("/translate", files=files)
    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"].lower()
