import os
import sys

# Ensure UTF-8 output streams on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.data.preprocessing import AudioPreprocessor
from src.data.vocabulary import ASRVocabulary, TranslationVocabulary
from src.data.tokenizer import ASRTokenizer, TranslationTokenizer

logger = get_logger("inspect_data")

def inspect():
    logger.info("==========================================================")
    logger.info("           NEPALI VOICE TRANSLATOR - DATA INSPECTION      ")
    logger.info("==========================================================")

    cfg = load_config("configs/config.yaml")

    # 1. ASR Inspection
    asr_train_csv = os.path.join(cfg["asr"]["metadata_dir"], "asr_train.csv")
    asr_vocab_json = os.path.join(cfg["asr"]["processed_data_dir"], "vocab.json")

    if os.path.exists(asr_train_csv) and os.path.exists(asr_vocab_json):
        df_asr = pd.read_csv(asr_train_csv)
        sample_row = df_asr.iloc[0]
        audio_path = sample_row["audio_path"]
        transcript = sample_row["transcription"]
        speaker = sample_row["speaker_id"]

        preprocessor = AudioPreprocessor(
            target_sample_rate=cfg["asr"]["target_sample_rate"],
            n_mels=cfg["asr"]["features"]["n_mels"],
            n_fft=cfg["asr"]["features"]["n_fft"],
            hop_length=cfg["asr"]["features"]["hop_length"],
            win_length=cfg["asr"]["features"]["win_length"]
        )
        waveform, sr = preprocessor.load_audio(audio_path)
        mel_spec = preprocessor.extract_mel_spectrogram(waveform)

        vocab = ASRVocabulary.load(asr_vocab_json)
        tokenizer = ASRTokenizer(vocab)
        encoded_tokens = tokenizer.encode(transcript)
        decoded_text = tokenizer.decode(encoded_tokens)

        print("\n--- [1] ASR SAMPLE INSPECTION ---")
        print(f"Audio Path:           {audio_path}")
        print(f"Speaker ID:           {speaker}")
        print(f"Sampling Rate:        {sr} Hz")
        print(f"Waveform Shape:       {list(waveform.shape)} ({waveform.dtype})")
        print(f"Waveform Min / Max:   {waveform.min().item():.3f} / {waveform.max().item():.3f}")
        print(f"Duration:             {sample_row['duration']:.2f} s")
        print(f"Mel-Spectrogram Shape:{list(mel_spec.shape)} [n_mels, time_frames]")
        print(f"Original Text:        '{transcript}'")
        print(f"Encoded Token IDs:    {encoded_tokens[:12]}... (len={len(encoded_tokens)})")
        print(f"Decoded Text:         '{decoded_text}'")
        print(f"ASR Vocab Size:       {len(vocab)} characters")

    # 2. Translation Inspection
    trans_train_csv = os.path.join(cfg["translation"]["metadata_dir"], "translation_train.csv")
    src_vocab_json = os.path.join(cfg["translation"]["processed_data_dir"], "src_vocab.json")
    tgt_vocab_json = os.path.join(cfg["translation"]["processed_data_dir"], "tgt_vocab.json")

    if os.path.exists(trans_train_csv) and os.path.exists(src_vocab_json) and os.path.exists(tgt_vocab_json):
        df_trans = pd.read_csv(trans_train_csv)
        sample_pair = df_trans.iloc[0]
        ne_text = sample_pair["nepali_text"]
        en_text = sample_pair["english_text"]

        src_vocab = TranslationVocabulary.load(src_vocab_json)
        tgt_vocab = TranslationVocabulary.load(tgt_vocab_json)

        src_tokenizer = TranslationTokenizer(src_vocab, is_nepali=True)
        tgt_tokenizer = TranslationTokenizer(tgt_vocab, is_nepali=False)

        src_encoded = src_tokenizer.encode(ne_text, add_special_tokens=True)
        tgt_encoded = tgt_tokenizer.encode(en_text, add_special_tokens=True)

        print("\n--- [2] TRANSLATION SAMPLE INSPECTION ---")
        print(f"Source (Nepali):      '{ne_text}'")
        print(f"Target (English):     '{en_text}'")
        print(f"Source Token IDs:     {src_encoded} (with <SOS>=2, <EOS>=3)")
        print(f"Target Token IDs:     {tgt_encoded} (with <SOS>=2, <EOS>=3)")
        print(f"Decoded Source:       '{src_tokenizer.decode(src_encoded)}'")
        print(f"Decoded Target:       '{tgt_tokenizer.decode(tgt_encoded)}'")
        print(f"Nepali Vocab Size:    {len(src_vocab)} tokens")
        print(f"English Vocab Size:   {len(tgt_vocab)} tokens")

        src_lengths = [len(t.split()) for t in df_trans["nepali_text"]]
        tgt_lengths = [len(t.split()) for t in df_trans["english_text"]]
        print(f"\n--- [3] LENGTH DISTRIBUTIONS (TRAIN SET) ---")
        print(f"Nepali Token Lengths: Min={min(src_lengths)}, Max={max(src_lengths)}, Mean={np.mean(src_lengths):.2f}")
        print(f"English Token Lengths:Min={min(tgt_lengths)}, Max={max(tgt_lengths)}, Mean={np.mean(tgt_lengths):.2f}")
        print("==========================================================\n")

if __name__ == "__main__":
    inspect()
