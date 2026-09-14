import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer import Transformer
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.data.collate import get_translation_dataloaders
from src.training.train_transformer import build_transformer_model, train_transformer_epoch, evaluate_transformer
from src.inference.translate_transformer import translate_transformer_greedy
from src.evaluation.metrics import calculate_sentence_bleu
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger

logger = get_logger("train_transformer")

def calculate_corpus_bleu(model, dataloader, src_tokenizer, tgt_tokenizer, device) -> float:
    model.eval()
    scores = []
    with torch.inference_mode():
        for batch in dataloader:
            _, _, _, _, raw_srcs, raw_tgts = batch
            for src_text, tgt_text in zip(raw_srcs, raw_tgts):
                pred, _, _, _ = translate_transformer_greedy(
                    model=model,
                    sentence=src_text,
                    src_tokenizer=src_tokenizer,
                    tgt_tokenizer=tgt_tokenizer,
                    device=device
                )
                ref_tokens = tgt_text.strip().lower().split()
                hyp_tokens = pred.strip().lower().split()
                scores.append(calculate_sentence_bleu(ref_tokens, hyp_tokens))
    return float(sum(scores) / max(1, len(scores))) * 100.0

def main():
    parser = argparse.ArgumentParser(description="Train Transformer from Scratch for Nepali-English NMT")
    parser.add_argument("--train_csv", type=str, default="data/metadata/translation_train.csv")
    parser.add_argument("--val_csv", type=str, default="data/metadata/translation_val.csv")
    parser.add_argument("--test_csv", type=str, default="data/metadata/translation_test.csv")
    parser.add_argument("--src_vocab", type=str, default="data/processed/nmt/src_vocab.json")
    parser.add_argument("--tgt_vocab", type=str, default="data/processed/nmt/tgt_vocab.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--ffn_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_transformer.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("       🚀  TRAINING TRANSFORMER FROM SCRATCH (PHASE 7)  🚀")
    print("=" * 70)
    print(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | d_model: {args.d_model}")

    src_vocab = TranslationVocabulary.load(args.src_vocab)
    tgt_vocab = TranslationVocabulary.load(args.tgt_vocab)

    src_tok = TranslationTokenizer(src_vocab, is_nepali=True)
    tgt_tok = TranslationTokenizer(tgt_vocab, is_nepali=False)

    train_loader, val_loader, test_loader = get_translation_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        src_tokenizer=src_tok,
        tgt_tokenizer=tgt_tok,
        batch_size=args.batch_size
    )

    model = build_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        pad_idx=src_vocab.pad_idx,
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_transformer_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_transformer(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        epoch_sec = time.time() - t0

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val_bleu = calculate_corpus_bleu(model, val_loader, src_tok, tgt_tok, device)
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:5.2f}% | Time: {epoch_sec:.2f}s")
        else:
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_sec:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": {
                    "model": {
                        "d_model": args.d_model,
                        "num_heads": args.num_heads,
                        "num_encoder_layers": args.num_encoder_layers,
                        "num_decoder_layers": args.num_decoder_layers,
                        "ffn_dim": args.ffn_dim,
                        "dropout": args.dropout
                    }
                },
                "src_vocab": src_vocab.token2idx,
                "tgt_vocab": tgt_vocab.token2idx,
                "model_type": "transformer_scratch_word_v1"
            }
            torch.save(checkpoint, args.checkpoint_path)
            logger.info(f"Saved new best checkpoint to {args.checkpoint_path} (Val Loss: {val_loss:.4f})")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s. Evaluating on Test Set...")

    # Load best checkpoint
    best_ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_loss = evaluate_transformer(model, test_loader, criterion, device)
    test_bleu = calculate_corpus_bleu(model, test_loader, src_tok, tgt_tok, device)

    print("\n" + "=" * 70)
    print("                     🏆  FINAL TRANSFORMER TEST RESULTS  🏆")
    print("=" * 70)
    print(f"  • Best Validation Loss : {best_val_loss:.4f}")
    print(f"  • Test Loss            : {test_loss:.4f}")
    print(f"  • Test BLEU Score      : {test_bleu:.2f}%")
    print(f"  • Checkpoint Saved     : {args.checkpoint_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
