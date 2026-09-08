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
from src.training.train_translation import build_seq2seq_model, train_epoch, evaluate_epoch
from src.training.evaluate_translation import run_full_evaluation
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed
from src.utils.config_loader import load_config
from src.utils.checkpoint import save_checkpoint
from src.inference.translate import translate_sentence

logger = get_logger("train_translation_script")

def main():
    logger.info("==========================================================")
    logger.info("   PHASE 2 - NEPALI TO ENGLISH LSTM SEQ2SEQ TRAINING      ")
    logger.info("==========================================================")

    cfg = load_config("configs/translation.yaml")
    set_seed(42)

    # Device selection
    if cfg["training"]["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["training"]["device"])
    logger.info(f"Using compute device: {device}")

    # Load vocabularies
    src_vocab = TranslationVocabulary.load(cfg["data"]["src_vocab_path"])
    tgt_vocab = TranslationVocabulary.load(cfg["data"]["tgt_vocab_path"])
    logger.info(f"Vocabularies Loaded: Nepali Source={len(src_vocab)}, English Target={len(tgt_vocab)}")

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    train_loader, val_loader, test_loader = get_translation_dataloaders(
        train_csv=cfg["data"]["train_csv"],
        val_csv=cfg["data"]["val_csv"],
        test_csv=cfg["data"]["test_csv"],
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=cfg["training"]["batch_size"],
        pin_memory=(device.type == "cuda")
    )

    # Build model
    model = build_seq2seq_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        emb_dim=cfg["model"]["embedding_dim"],
        hid_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        pad_idx=src_vocab.pad_idx,
        device=device
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total Trainable Parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    best_val_loss = float("inf")
    epochs = cfg["training"]["epochs"]
    checkpoint_dir = cfg["training"]["checkpoint_dir"]
    checkpoint_filename = cfg["training"]["checkpoint_filename"]

    print("\n" + "-"*65)
    print(f"{'Epoch':^8} | {'Train Loss':^14} | {'Val Loss':^14} | {'Status':^18}")
    print("-"*65)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=cfg["training"]["gradient_clip_max_norm"],
            teacher_forcing_ratio=cfg["training"]["teacher_forcing_ratio"],
            device=device
        )

        val_loss = evaluate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            status = "* Best Model Saved"
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": cfg,
                    "src_vocab": src_vocab.token2idx,
                    "tgt_vocab": tgt_vocab.token2idx
                },
                checkpoint_dir=checkpoint_dir,
                filename=checkpoint_filename
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"{epoch:^8} | {train_loss:^14.4f} | {val_loss:^14.4f} | {status:^18}")

    print("-"*65)
    logger.info("Training complete. Running final evaluation...")

    # Load best checkpoint for final evaluation
    from src.utils.checkpoint import load_checkpoint
    best_ckpt = load_checkpoint(os.path.join(checkpoint_dir, checkpoint_filename), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    metrics = run_full_evaluation(
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        criterion=criterion,
        device=device
    )

    print("\n" + "="*70)
    print("                    FINAL EVALUATION SUMMARY                       ")
    print("="*70)
    print(f"Best Validation Loss: {metrics['val_loss']:.4f}")
    print(f"Validation BLEU:      {metrics['val_bleu']:.2f}")
    print(f"Test Loss:            {metrics['test_loss']:.4f}")
    print(f"Test BLEU:            {metrics['test_bleu']:.2f}")
    print("="*70)

    # Sample translations
    sample_prompts = [
        "नमस्ते, तपाईँलाई कस्तो छ?",
        "काठमाडौँ नेपालको राजधानी हो।",
        "सगरमाथा संसारको सबैभन्दा अग्लो हिमाल हो।",
        "म कलेज जान्छु।",
        "हाम्रो टोलीले खेल जित्यो।"
    ]
    print("\n--- SAMPLE TRANSLATIONS (GREEDY INFERENCE) ---")
    for prompt in sample_prompts:
        translation = translate_sentence(model, prompt, src_tok, tgt_tok, max_len=25, device=device)
        print(f"Nepali:  '{prompt}'")
        print(f"English: '{translation}'\n")
    print("==========================================================\n")

if __name__ == "__main__":
    main()
