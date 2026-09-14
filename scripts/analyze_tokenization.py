import os
import sys
import pandas as pd
import json
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.data.subword_tokenizer import BPETokenizer

def main():
    test_csv = "data/metadata/translation_test.csv"
    val_csv = "data/metadata/translation_val.csv"
    df = pd.concat([pd.read_csv(val_csv), pd.read_csv(test_csv)])

    print("=" * 70)
    print("   📊  TOKENIZATION SCHEME COMPARISON: CHAR vs WORD vs SUBWORD  📊")
    print("=" * 70)

    # 1. Word-level tokenizers
    src_word_vocab = TranslationVocabulary.load("data/processed/nmt/src_vocab.json")
    tgt_word_vocab = TranslationVocabulary.load("data/processed/nmt/tgt_vocab.json")
    word_src_tok = TranslationTokenizer(src_word_vocab, is_nepali=True)
    word_tgt_tok = TranslationTokenizer(tgt_word_vocab, is_nepali=False)

    # 2. Subword tokenizers
    sub_src_tok = BPETokenizer.load("data/processed/nmt/subword_nepali.json")
    sub_tgt_tok = BPETokenizer.load("data/processed/nmt/subword_english.json")

    # Metrics containers
    word_src_lens, word_tgt_lens, word_unk_counts, total_word_tokens = [], [], 0, 0
    sub_src_lens, sub_tgt_lens, sub_unk_counts, total_sub_tokens = [], [], 0, 0
    char_src_lens, char_tgt_lens = [], []

    for _, row in df.iterrows():
        nep = str(row["nepali_text"])
        eng = str(row["english_text"])

        # Character
        char_src_lens.append(len(list(nep.replace(" ", ""))))
        char_tgt_lens.append(len(list(eng.replace(" ", ""))))

        # Word
        w_src = word_src_tok.encode(nep, add_special_tokens=False)
        w_tgt = word_tgt_tok.encode(eng, add_special_tokens=False)
        word_src_lens.append(len(w_src))
        word_tgt_lens.append(len(w_tgt))
        word_unk_counts += sum(1 for t in w_src if t == word_src_tok.vocab.unk_idx)
        total_word_tokens += len(w_src)

        # Subword
        s_src = sub_src_tok.encode(nep, add_special_tokens=False)
        s_tgt = sub_tgt_tok.encode(eng, add_special_tokens=False)
        sub_src_lens.append(len(s_src))
        sub_tgt_lens.append(len(s_tgt))
        sub_unk_counts += sum(1 for t in s_src if t == sub_src_tok.unk_idx)
        total_sub_tokens += len(s_src)

    table = [
        {
            "Tokenizer Scheme": "Character-Level",
            "Avg Source Len": round(float(np.mean(char_src_lens)), 2),
            "Max Source Len": int(np.max(char_src_lens)),
            "Avg Target Len": round(float(np.mean(char_tgt_lens)), 2),
            "Max Target Len": int(np.max(char_tgt_lens)),
            "UNK Rate (%)": "0.0%"
        },
        {
            "Tokenizer Scheme": "Word-Level (Phase 2-7)",
            "Avg Source Len": round(float(np.mean(word_src_lens)), 2),
            "Max Source Len": int(np.max(word_src_lens)),
            "Avg Target Len": round(float(np.mean(word_tgt_lens)), 2),
            "Max Target Len": int(np.max(word_tgt_lens)),
            "UNK Rate (%)": f"{(word_unk_counts / max(1, total_word_tokens)) * 100.0:.2f}%"
        },
        {
            "Tokenizer Scheme": "Subword BPE (Phase 8)",
            "Avg Source Len": round(float(np.mean(sub_src_lens)), 2),
            "Max Source Len": int(np.max(sub_src_lens)),
            "Avg Target Len": round(float(np.mean(sub_tgt_lens)), 2),
            "Max Target Len": int(np.max(sub_tgt_lens)),
            "UNK Rate (%)": f"{(sub_unk_counts / max(1, total_sub_tokens)) * 100.0:.2f}%"
        }
    ]

    df_out = pd.DataFrame(table)
    print(df_out.to_string(index=False))
    print("=" * 70)

    os.makedirs("results", exist_ok=True)
    df_out.to_json("results/tokenization_analysis.json", orient="records", indent=2)
    print("Tokenization analysis saved to 'results/tokenization_analysis.json'.")

if __name__ == "__main__":
    main()
