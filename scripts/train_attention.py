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
from src.training.train_attention import build_seq2seq_attention_model, train_attention_epoch, evaluate_attention_epoch
from src.inference.translate_attention import evaluate_attention_corpus_bleu, translate_with_attention
from src.evaluation.attention_visualization import plot_attention_matrix, print_text_attention_alignment
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed
from src.utils.config_loader import load_config
from src.utils.checkpoint import save_checkpoint

logger = get_logger("train_attention_script")

def main():
    logger.info("==========================================================")
    logger.info("  PHASE 3 - LSTM SEQ2SEQ WITH BAHDANAU ATTENTION TRAINING ")
    logger.info("==========================================================")

    cfg = load_config("experiments/lstm_attention.yaml")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    src_vocab = TranslationVocabulary.load("data/processed/translation/src_vocab.json")
    tgt_vocab = TranslationVocabulary.load("data/processed/translation/tgt_vocab.json")
    logger.info(f"Vocabularies: Nepali Source={len(src_vocab)}, English Target={len(tgt_vocab)}")

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    train_loader, val_loader, test_loader = get_translation_dataloaders(
        train_csv="data/metadata/translation_train.csv",
        val_csv="data/metadata/translation_val.csv",
        test_csv="data/metadata/translation_test.csv",
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=cfg["training"]["batch_size"],
        pin_memory=(device.type == "cuda")
    )

    model = build_seq2seq_attention_model(
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
    logger.info(f"Total Trainable Parameters (LSTM + Attention): {num_params:,}")

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
        train_loss, grads = train_attention_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=cfg["training"]["gradient_clip_max_norm"],
            teacher_forcing_ratio=cfg["training"]["teacher_forcing_ratio"],
            device=device
        )

        val_loss = evaluate_attention_epoch(
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
    logger.info("Training complete. Running final evaluation & generating attention heatmap...")

    # Load best checkpoint
    from src.utils.checkpoint import load_checkpoint
    best_ckpt = load_checkpoint(os.path.join(checkpoint_dir, checkpoint_filename), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_loss = evaluate_attention_epoch(model, val_loader, criterion, device)
    test_loss = evaluate_attention_epoch(model, test_loader, criterion, device)
    val_bleu = evaluate_attention_corpus_bleu(model, val_loader, src_tok, tgt_tok, device)
    test_bleu = evaluate_attention_corpus_bleu(model, test_loader, src_tok, tgt_tok, device)

    print("\n" + "="*70)
    print("              LSTM + ATTENTION FINAL EVALUATION SUMMARY            ")
    print("="*70)
    print(f"Best Validation Loss: {val_loss:.4f}")
    print(f"Validation BLEU:      {val_bleu:.2f}")
    print(f"Test Loss:            {test_loss:.4f}")
    print(f"Test BLEU:            {test_bleu:.2f}")
    print("="*70)

    # Attention visualization on key sentence
    demo_sentence = "नमस्ते, तपाईँलाई कस्तो छ?"
    translation, attn_matrix, s_tokens, t_tokens = translate_with_attention(
        model, demo_sentence, src_tok, tgt_tok, max_len=20, device=device
    )
    print(f"\nDemo Translation: '{demo_sentence}' -> '{translation}'")
    print_text_attention_alignment(attn_matrix, s_tokens, t_tokens)
    plot_attention_matrix(
        attn_matrix, s_tokens, t_tokens,
        save_path="reports/attention_heatmap.png",
        title=f"Attention Map: '{demo_sentence}'"
    )
    logger.info("Saved Attention Heatmap to reports/attention_heatmap.png")

if __name__ == "__main__":
    main()
