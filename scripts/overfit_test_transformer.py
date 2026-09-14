import os
import sys
import tempfile
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.transformer import Transformer
from src.data.vocabulary import TranslationVocabulary
from src.data.tokenizer import TranslationTokenizer
from src.data.translation_dataset import TranslationDataset
from src.data.collate import TranslationCollateFn
from src.inference.translate_transformer import translate_transformer_greedy

def main():
    print("=" * 60)
    print("    🧪  TRANSFORMER ARCHITECTURE OVERFIT TEST (TINY SET)  🧪")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    toy_data = [
        {"nepali_text": "नमस्ते", "english_text": "hello ."},
        {"nepali_text": "म घर जान्छु ।", "english_text": "i go home ."},
        {"nepali_text": "तपाईंलाई कस्तो छ ?", "english_text": "how are you ?"},
        {"nepali_text": "धन्यवाद साथी ।", "english_text": "thank you friend ."},
        {"nepali_text": "यो धेरै राम्रो छ ।", "english_text": "this is very good ."}
    ]

    df = pd.DataFrame(toy_data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        df.to_csv(f.name, index=False)
        temp_csv = f.name

    try:
        src_vocab = TranslationVocabulary()
        src_vocab.build_from_texts([x["nepali_text"] for x in toy_data])
        tgt_vocab = TranslationVocabulary()
        tgt_vocab.build_from_texts([x["english_text"] for x in toy_data])

        src_tokenizer = TranslationTokenizer(src_vocab, is_nepali=True)
        tgt_tokenizer = TranslationTokenizer(tgt_vocab, is_nepali=False)

        collate_fn = TranslationCollateFn(src_pad_id=src_vocab.pad_idx, tgt_pad_id=tgt_vocab.pad_idx)
        dataset = TranslationDataset(
            metadata_csv=temp_csv,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)

        # Lightweight Transformer model
        model = Transformer(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            d_model=64,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            ffn_dim=128,
            dropout=0.0,
            pad_idx=src_vocab.pad_idx,
            device=device
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)

        print(f"Training on {len(toy_data)} samples for 120 epochs...")
        for epoch in range(1, 121):
            model.train()
            for batch in loader:
                src_b, _, tgt_b, _, _, _ = batch
                src_b, tgt_b = src_b.to(device), tgt_b.to(device)
                dec_in = tgt_b[:, :-1]
                labels = tgt_b[:, 1:]

                optimizer.zero_grad()
                logits, _ = model(src_b, dec_in)
                loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss.backward()
                optimizer.step()

            if epoch % 30 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

        print("\nVerifying Autoregressive Greedy Decoding on Trained Samples:")
        for item in toy_data:
            pred, _, _, _ = translate_transformer_greedy(
                model=model,
                sentence=item["nepali_text"],
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device
            )
            print(f"Nepali: '{item['nepali_text']}' -> Target: '{item['english_text']}' | Pred: '{pred}'")

        print("\n" + "=" * 60)
        print("✅ Transformer Overfit Test Completed Successfully!")
        print("=" * 60)
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

if __name__ == "__main__":
    main()
