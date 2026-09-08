import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.training.train_translation import build_seq2seq_model
from src.utils.checkpoint import load_checkpoint
from src.inference.translate import translate_sentence
from src.utils.config_loader import load_config

def main():
    cfg = load_config("configs/translation.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg["training"]["checkpoint_dir"], cfg["training"]["checkpoint_filename"])
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}. Please run 'python scripts/train_translation.py' first.")
        return

    checkpoint = load_checkpoint(ckpt_path, map_location=device)
    src_vocab = TranslationVocabulary(checkpoint["src_vocab"])
    tgt_vocab = TranslationVocabulary(checkpoint["tgt_vocab"])

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    model = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=cfg["model"]["embedding_dim"],
        hid_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=0.0,
        pad_idx=src_vocab.pad_idx,
        device=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if len(sys.argv) > 1:
        nepali_input = " ".join(sys.argv[1:])
        translation = translate_sentence(model, nepali_input, src_tok, tgt_tok, max_len=30, device=device)
        print(f"Nepali:  {nepali_input}")
        print(f"English: {translation}")
    else:
        print("Nepali Voice Translator - Translation CLI (Type 'exit' to quit)")
        while True:
            try:
                inp = input("Enter Nepali sentence > ").strip()
                if inp.lower() in ("exit", "quit", "q"):
                    break
                if not inp:
                    continue
                out = translate_sentence(model, inp, src_tok, tgt_tok, max_len=30, device=device)
                print(f"-> English: {out}\n")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
