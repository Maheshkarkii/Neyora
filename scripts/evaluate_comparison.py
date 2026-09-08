import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.data.collate import get_translation_dataloaders
from src.training.train_translation import build_seq2seq_model, evaluate_epoch
from src.training.train_attention import build_seq2seq_attention_model, evaluate_attention_epoch
from src.inference.translate import translate_sentence, evaluate_corpus_bleu
from src.inference.translate_attention import translate_with_attention, evaluate_attention_corpus_bleu
from src.utils.checkpoint import load_checkpoint
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate_comparison")

def compare_models():
    logger.info("==========================================================")
    logger.info("   PHASE 3 BENCHMARK: BASELINE LSTM vs LSTM + ATTENTION   ")
    logger.info("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_vocab = TranslationVocabulary.load("data/processed/translation/src_vocab.json")
    tgt_vocab = TranslationVocabulary.load("data/processed/translation/tgt_vocab.json")

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    train_loader, val_loader, test_loader = get_translation_dataloaders(
        train_csv="data/metadata/translation_train.csv",
        val_csv="data/metadata/translation_val.csv",
        test_csv="data/metadata/translation_test.csv",
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=8
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    # 1. Load Baseline Model A
    ckpt_a = load_checkpoint("checkpoints/best_translation_model.pt", map_location=device)
    model_a = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=128,
        hid_dim=256,
        num_layers=2,
        pad_idx=src_vocab.pad_idx,
        device=device
    )
    model_a.load_state_dict(ckpt_a["model_state_dict"])
    val_loss_a = evaluate_epoch(model_a, val_loader, criterion, device)
    test_loss_a = evaluate_epoch(model_a, test_loader, criterion, device)
    val_bleu_a = evaluate_corpus_bleu(model_a, val_loader, src_tok, tgt_tok, device)
    test_bleu_a = evaluate_corpus_bleu(model_a, test_loader, src_tok, tgt_tok, device)

    # 2. Load Attention Model B
    ckpt_b = load_checkpoint("checkpoints/best_lstm_attention.pt", map_location=device)
    model_b = build_seq2seq_attention_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=128,
        hid_dim=256,
        num_layers=2,
        pad_idx=src_vocab.pad_idx,
        device=device
    )
    model_b.load_state_dict(ckpt_b["model_state_dict"])
    val_loss_b = evaluate_attention_epoch(model_b, val_loader, criterion, device)
    test_loss_b = evaluate_attention_epoch(model_b, test_loader, criterion, device)
    val_bleu_b = evaluate_attention_corpus_bleu(model_b, val_loader, src_tok, tgt_tok, device)
    test_bleu_b = evaluate_attention_corpus_bleu(model_b, test_loader, src_tok, tgt_tok, device)

    print("\n" + "="*78)
    print(f"{'Model Architecture':<28} | {'Val Loss':<10} | {'Test Loss':<10} | {'Val BLEU':<10} | {'Test BLEU':<10}")
    print("="*78)
    print(f"{'1. Baseline LSTM Seq2Seq':<28} | {val_loss_a:<10.4f} | {test_loss_a:<10.4f} | {val_bleu_a:<10.2f} | {test_bleu_a:<10.2f}")
    print(f"{'2. LSTM + Bahdanau Attention':<28} | {val_loss_b:<10.4f} | {test_loss_b:<10.4f} | {val_bleu_b:<10.2f} | {test_bleu_b:<10.2f}")
    print("="*78)

    # Qualitative comparison examples
    test_prompts = [
        {"ne": "नमस्ते, तपाईँलाई कस्तो छ?", "gt": "hello , how are you ?"},
        {"ne": "काठमाडौँ नेपालको राजधानी हो।", "gt": "kathmandu is the capital of nepal ."},
        {"ne": "सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो।", "gt": "mount everest is the highest mountain in the world ."},
        {"ne": "म कलेज जान्छु।", "gt": "i go to college ."}
    ]

    print("\n--- QUALITATIVE TRANSLATION COMPARISON ---")
    for item in test_prompts:
        ne = item["ne"]
        gt = item["gt"]
        pred_a = translate_sentence(model_a, ne, src_tok, tgt_tok, max_len=20, device=device)
        pred_b, _, _, _ = translate_with_attention(model_b, ne, src_tok, tgt_tok, max_len=20, device=device)
        print(f"Nepali Input:      '{ne}'")
        print(f"Ground Truth:      '{gt}'")
        print(f"Baseline LSTM:     '{pred_a}'")
        print(f"LSTM + Attention:  '{pred_b}'\n")
    print("==========================================================\n")

if __name__ == "__main__":
    compare_models()
