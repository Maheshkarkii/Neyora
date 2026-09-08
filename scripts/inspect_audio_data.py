import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
import matplotlib.pyplot as plt
from src.data.preprocessing import AudioPreprocessor, NepaliTextCleaner
from src.data.vocabulary import ASRVocabulary
from src.data.tokenizer import ASRTokenizer
from src.data.asr_dataset import ASRDataset

def main():
    preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)
    vocab = ASRVocabulary.load("data/processed/asr/vocab.json")
    tokenizer = ASRTokenizer(vocab)

    ds = ASRDataset("data/metadata/asr_train.csv", preprocessor, tokenizer)
    print(f"Inspecting ASR dataset of size: {len(ds)}")

    for i in range(min(3, len(ds))):
        sample = ds[i]
        print(f"\n[--- Sample {i} ---]")
        print(f"Audio Path:  {sample['audio_path']}")
        print(f"Speaker ID:  {sample['speaker_id']}")
        print(f"Duration:    {sample['duration']:.2f} seconds")
        print(f"Text:        {sample['text']}")
        print(f"Mel-Spec Shape: {sample['mel_spec'].shape}")
        print(f"Token Sequence:    {sample['tokens'].tolist()}")

    os.makedirs("reports", exist_ok=True)
    fig = plt.figure(figsize=(10, 4))
    spec = ds[0]['mel_spec'].detach().cpu().numpy()
    plt.imshow(spec, origin='lower', aspect='auto', cmap='viridis')
    plt.title(f"Log-Mel Spectrogram: {ds[0]['text']}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Filter Banks (80)")
    plt.tight_layout()
    plt.savefig("reports/sample_mel_spectrogram.png", dpi=150)
    print("\nSaved spectrogram plot to reports/sample_mel_spectrogram.png")

if __name__ == '__main__':
    main()
