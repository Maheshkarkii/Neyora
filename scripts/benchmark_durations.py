import os
import sys
import torch
import numpy as np
import soundfile as sf
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.voice_translator import NepaliVoiceTranslator

def main():
    trans = NepaliVoiceTranslator(device="cpu")
    durations = [0.5, 1.0, 2.5, 5.0, 10.0, 15.0]
    print("\n=== LATENCY & RTF vs AUDIO DURATION ===")
    for d in durations:
        sr = 16000
        t = np.linspace(0, d, int(sr * d), endpoint=False, dtype=np.float32)
        wave = 0.3 * np.sin(2 * np.pi * 440 * t)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wave, sr)
            p = f.name
        res = trans.translate_audio(p)
        os.remove(p)
        print(f"Duration: {d:>4.1f}s | ASR: {res['asr_latency_sec']:.4f}s | NMT: {res['translation_latency_sec']:.4f}s | Total: {res['total_latency_sec']:.4f}s | RTF: {res['real_time_factor']:.4f}")

    print("\n=== FAILURE CASE & ROBUSTNESS TESTING ===")
    # 1. Pure Silence
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, np.zeros(16000, dtype=np.float32), 16000)
        p = f.name
    res = trans.translate_audio(p)
    os.remove(p)
    print(f"1. Silence (1.0s) -> ASR: '{res['transcription']}', NMT: '{res['translation']}' (Confidence: {res['asr_confidence']})")

    # 2. Gaussian Noise
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, np.random.normal(0, 0.1, 16000).astype(np.float32), 16000)
        p = f.name
    res = trans.translate_audio(p)
    os.remove(p)
    print(f"2. White Noise (1.0s) -> ASR: '{res['transcription']}', NMT: '{res['translation']}' (Confidence: {res['asr_confidence']})")

    # 3. Very Short Audio (0.2s)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, np.random.normal(0, 0.05, 3200).astype(np.float32), 16000)
        p = f.name
    res = trans.translate_audio(p)
    os.remove(p)
    print(f"3. Very Short (0.2s) -> ASR: '{res['transcription']}', NMT: '{res['translation']}' (Confidence: {res['asr_confidence']})")

    # 4. Out-of-vocabulary / unseen text direct translation test
    oov_nepali = "कम्प्युटर सफ्टवेयर प्रोग्रामिङ"
    oov_res = trans.translate_text(oov_nepali)
    print(f"4. OOV Nepali Text -> '{oov_nepali}' -> NMT: '{oov_res['translation']}'")

if __name__ == "__main__":
    main()
