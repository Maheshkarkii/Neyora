import os
import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.subword_tokenizer import BPETokenizer

def main():
    train_csv = "data/metadata/translation_train.csv"
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found.")
        sys.exit(1)

    print("=" * 70)
    print("      🧩  TRAINING BPE SUBWORD TOKENIZERS (PHASE 8)  🧩")
    print("=" * 70)

    df_train = pd.read_csv(train_csv)
    nepali_train_texts = df_train["nepali_text"].dropna().tolist()
    english_train_texts = df_train["english_text"].dropna().tolist()

    print(f"Training Nepali BPE on {len(nepali_train_texts)} train sentences (No val/test leakage)...")
    nepali_tokenizer = BPETokenizer(is_nepali=True)
    nepali_tokenizer.train(nepali_train_texts, num_merges=120, min_freq=2)
    nepali_path = "data/processed/nmt/subword_nepali.json"
    nepali_tokenizer.save(nepali_path)
    print(f"  • Nepali Vocab Size: {len(nepali_tokenizer)} subwords -> Saved to {nepali_path}")

    print(f"\nTraining English BPE on {len(english_train_texts)} train sentences...")
    english_tokenizer = BPETokenizer(is_nepali=False)
    english_tokenizer.train(english_train_texts, num_merges=100, min_freq=2)
    english_path = "data/processed/nmt/subword_english.json"
    english_tokenizer.save(english_path)
    print(f"  • English Vocab Size: {len(english_tokenizer)} subwords -> Saved to {english_path}")

    # Reversibility & OOV test
    print("\n--- Verifying Subword Encoding / Decoding Reversibility ---")
    sample_nep = "हामी सबै मिलेर काम गर्नुपर्छ।"
    enc_nep = nepali_tokenizer.encode(sample_nep)
    dec_nep = nepali_tokenizer.decode(enc_nep)
    tok_nep = nepali_tokenizer.tokenize(sample_nep)
    print(f"Input Nepali:    '{sample_nep}'")
    print(f"Subword Tokens:  {tok_nep}")
    print(f"Decoded Nepali:  '{dec_nep}'")
    assert sample_nep.replace(" ", "") == dec_nep.replace(" ", "")

    sample_eng = "we must all work together ."
    enc_eng = english_tokenizer.encode(sample_eng)
    dec_eng = english_tokenizer.decode(enc_eng)
    tok_eng = english_tokenizer.tokenize(sample_eng)
    print(f"\nInput English:   '{sample_eng}'")
    print(f"Subword Tokens:  {tok_eng}")
    print(f"Decoded English: '{dec_eng}'")
    assert sample_eng.replace(" ", "") == dec_eng.replace(" ", "")

    print("\n" + "=" * 70)
    print("✅ Subword Tokenizer Training & Reversibility Test Succeeded!")
    print("=" * 70)

if __name__ == "__main__":
    main()
