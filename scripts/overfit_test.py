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
from src.training.train_translation import build_seq2seq_model, train_epoch
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed
from src.utils.config_loader import load_config
from src.inference.translate import translate_sentence

logger = get_logger("overfit_test")

def run_overfit_test():
    logger.info("==========================================================")
    logger.info("     PHASE 2 - OVERFITTING SANITY CHECK (DEBUGGING TEST)  ")
    logger.info("==========================================================")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    cfg = load_config("configs/translation.yaml")

    src_vocab = TranslationVocabulary.load(cfg["data"]["src_vocab_path"])
    tgt_vocab = TranslationVocabulary.load(cfg["data"]["tgt_vocab_path"])

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    train_loader, _, _ = get_translation_dataloaders(
        train_csv=cfg["data"]["train_csv"],
        val_csv=cfg["data"]["val_csv"],
        test_csv=cfg["data"]["test_csv"],
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=8
    )

    model = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=128,
        hid_dim=256,
        num_layers=2,
        dropout=0.0, # Zero dropout to allow fast memorization
        pad_idx=src_vocab.pad_idx,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    logger.info("Training on small training subset for 40 epochs...")

    initial_loss = None
    final_loss = None

    for epoch in range(1, 41):
        loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=1.0,
            teacher_forcing_ratio=0.8,
            device=device
        )
        if epoch == 1:
            initial_loss = loss
        if epoch % 10 == 0 or epoch == 40:
            logger.info(f"Epoch {epoch:02d}/40 | Training Loss: {loss:.4f}")
        final_loss = loss

    print("\n--- OVERFIT TEST RESULTS ---")
    print(f"Initial Loss (Epoch 1): {initial_loss:.4f}")
    print(f"Final Loss   (Epoch 40): {final_loss:.4f}")
    assert final_loss < initial_loss * 0.35, f"Loss did not decrease sufficiently! ({initial_loss:.4f} -> {final_loss:.4f})"
    print("SUCCESS: Loss dropped significantly! Architecture and backprop are verified.\n")

    # Test sample inference on memorized pairs
    test_sample = "नमस्ते, तपाईँलाई कस्तो छ?"
    translation = translate_sentence(model, test_sample, src_tok, tgt_tok, device=device)
    print(f"Sample Input:     '{test_sample}'")
    print(f"Model Prediction: '{translation}'")
    print("==========================================================\n")

if __name__ == "__main__":
    run_overfit_test()
