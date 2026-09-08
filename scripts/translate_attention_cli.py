import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.training.train_attention import build_seq2seq_attention_model
from src.utils.checkpoint import load_checkpoint
from src.inference.translate_attention import translate_with_attention
from src.evaluation.attention_visualization import plot_attention_matrix, print_text_attention_alignment

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "checkpoints/best_lstm_attention.pt"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Please run 'python scripts/train_attention.py' first.")
        return

    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    src_vocab = TranslationVocabulary(checkpoint["src_vocab"])
    tgt_vocab = TranslationVocabulary(checkpoint["tgt_vocab"])

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    model = build_seq2seq_attention_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=128,
        hid_dim=256,
        num_layers=2,
        dropout=0.0,
        pad_idx=src_vocab.pad_idx,
        device=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if len(sys.argv) > 1:
        inp = " ".join(sys.argv[1:])
        translation, attn_mat, s_toks, t_toks = translate_with_attention(model, inp, src_tok, tgt_tok, max_len=30, device=device)
        print(f"Nepali:  {inp}")
        print(f"English: {translation}")
        print_text_attention_alignment(attn_mat, s_toks, t_toks)
        plot_attention_matrix(attn_mat, s_toks, t_toks, save_path="reports/cli_attention_heatmap.png")
        print("Heatmap saved to reports/cli_attention_heatmap.png")
    else:
        print("Nepali Voice Translator - Attention CLI (Type 'exit' to quit)")
        while True:
            try:
                inp = input("Enter Nepali sentence > ").strip()
                if inp.lower() in ("exit", "quit", "q"):
                    break
                if not inp:
                    continue
                translation, attn_mat, s_toks, t_toks = translate_with_attention(model, inp, src_tok, tgt_tok, max_len=30, device=device)
                print(f"-> English: {translation}")
                print_text_attention_alignment(attn_mat, s_toks, t_toks)
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
