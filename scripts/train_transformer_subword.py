import os
import sys
import json
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer import Transformer
from src.data.subword_tokenizer import BPETokenizer
from src.evaluation.metrics import calculate_sentence_bleu
from src.utils.logger import get_logger

logger = get_logger("train_transformer_subword")

class SubwordTranslationDataset(Dataset):
    def __init__(self, csv_path: str, src_tok: BPETokenizer, tgt_tok: BPETokenizer):
        self.df = pd.read_csv(csv_path)
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        nep = str(row["nepali_text"])
        eng = str(row["english_text"])
        src_ids = self.src_tok.encode(nep, add_special_tokens=True)
        tgt_ids = self.tgt_tok.encode(eng, add_special_tokens=True)
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "raw_src": nep,
            "raw_tgt": eng
        }

def subword_collate_fn(batch):
    src_list = [item["src_ids"] for item in batch]
    tgt_list = [item["tgt_ids"] for item in batch]
    raw_srcs = [item["raw_src"] for item in batch]
    raw_tgts = [item["raw_tgt"] for item in batch]

    src_lens = [len(s) for s in src_list]
    tgt_lens = [len(t) for t in tgt_list]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_src = torch.zeros((len(batch), max_src), dtype=torch.long)
    padded_tgt = torch.zeros((len(batch), max_tgt), dtype=torch.long)

    for i in range(len(batch)):
        padded_src[i, :src_lens[i]] = src_list[i]
        padded_tgt[i, :tgt_lens[i]] = tgt_list[i]

    return padded_src, padded_tgt, raw_srcs, raw_tgts

def evaluate_subword_bleu(model, dataloader, src_tok, tgt_tok, device) -> float:
    model.eval()
    scores = []
    sos_idx = tgt_tok.sos_idx
    eos_idx = tgt_tok.eos_idx

    with torch.inference_mode():
        for padded_src, _, raw_srcs, raw_tgts in dataloader:
            padded_src = padded_src.to(device)
            for i in range(len(raw_srcs)):
                src_row = padded_src[i:i+1] # [1, S]
                src_mask = model.make_src_mask(src_row)
                memory, _ = model.encoder(src_row, src_mask=src_mask)

                tgt_indices = [sos_idx]
                for _ in range(35):
                    tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long, device=device).unsqueeze(0)
                    tgt_mask = model.make_tgt_mask(tgt_tensor)
                    dec_out, _, _ = model.decoder(tgt_tensor, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
                    next_tok = model.output_projection(dec_out)[:, -1, :].argmax(dim=-1).item()
                    if next_tok == eos_idx:
                        break
                    tgt_indices.append(next_tok)

                pred_str = tgt_tok.decode([t for t in tgt_indices if t not in (sos_idx, eos_idx)])
                ref_tokens = raw_tgts[i].strip().lower().split()
                hyp_tokens = pred_str.strip().lower().split()
                scores.append(calculate_sentence_bleu(ref_tokens, hyp_tokens))

    return float(sum(scores) / max(1, len(scores))) * 100.0

def main():
    parser = argparse.ArgumentParser(description="Train Subword Transformer (Phase 8)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_transformer_subword.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("      🚀  TRAINING SUBWORD TRANSFORMER (PHASE 8)  🚀")
    print("=" * 70)

    src_tok = BPETokenizer.load("data/processed/nmt/subword_nepali.json")
    tgt_tok = BPETokenizer.load("data/processed/nmt/subword_english.json")

    train_ds = SubwordTranslationDataset("data/metadata/translation_train.csv", src_tok, tgt_tok)
    val_ds = SubwordTranslationDataset("data/metadata/translation_val.csv", src_tok, tgt_tok)
    test_ds = SubwordTranslationDataset("data/metadata/translation_test.csv", src_tok, tgt_tok)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=subword_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=subword_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=subword_collate_fn)

    model = Transformer(
        src_vocab_size=len(src_tok),
        tgt_vocab_size=len(tgt_tok),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        ffn_dim=256,
        dropout=0.1,
        pad_idx=src_tok.pad_idx,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=src_tok.pad_idx)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for padded_src, padded_tgt, _, _ in train_loader:
            padded_src = padded_src.to(device)
            padded_tgt = padded_tgt.to(device)
            dec_in = padded_tgt[:, :-1]
            labels = padded_tgt[:, 1:]

            optimizer.zero_grad()
            logits, _ = model(padded_src, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / max(1, len(train_loader))

        # Evaluate validation
        model.eval()
        val_loss_tot = 0.0
        with torch.no_grad():
            for padded_src, padded_tgt, _, _ in val_loader:
                padded_src = padded_src.to(device)
                padded_tgt = padded_tgt.to(device)
                dec_in = padded_tgt[:, :-1]
                labels = padded_tgt[:, 1:]
                logits, _ = model(padded_src, dec_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                val_loss_tot += loss.item()
        val_loss = val_loss_tot / max(1, len(val_loader))
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            val_bleu = evaluate_subword_bleu(model, val_loader, src_tok, tgt_tok, device)
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val BLEU: {val_bleu:5.2f}%")

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
                        "num_encoder_layers": args.num_layers,
                        "num_decoder_layers": args.num_layers,
                        "ffn_dim": 256,
                        "dropout": 0.1
                    }
                },
                "src_vocab": src_tok.token2idx,
                "tgt_vocab": tgt_tok.token2idx,
                "model_type": "transformer_subword_bpe_v1"
            }
            torch.save(checkpoint, args.checkpoint_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s. Evaluating Subword Transformer on Test Set...")

    best_ckpt = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_bleu = evaluate_subword_bleu(model, test_loader, src_tok, tgt_tok, device)

    print("\n" + "=" * 70)
    print("              🏆  SUBWORD TRANSFORMER TEST RESULTS  🏆")
    print("=" * 70)
    print(f"  • Best Validation Loss : {best_val_loss:.4f}")
    print(f"  • Test BLEU Score      : {test_bleu:.2f}%")
    print(f"  • Checkpoint Saved     : {args.checkpoint_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
