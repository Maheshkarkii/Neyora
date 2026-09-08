import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from src.data.preprocessing import AudioPreprocessor
from src.data.vocabulary import ASRVocabulary
from src.data.tokenizer import ASRTokenizer
from src.data.asr_dataset import ASRDataset
from src.data.collate import ASRCollateFn
from src.models.asr_model import NepaliASR
from src.decoding.ctc_decoder import CTCDecoder
from src.evaluation.asr_metrics import compute_corpus_cer, compute_corpus_wer

def main():
    print("=== STARTING ASR OVERFIT SANITY CHECK ===")
    device = torch.device("cpu")

    preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)
    vocab = ASRVocabulary.load("data/processed/asr/vocab.json")
    tokenizer = ASRTokenizer(vocab)
    decoder = CTCDecoder(tokenizer, blank_idx=0, pad_idx=1)

    full_ds = ASRDataset("data/metadata/asr_train.csv", preprocessor, tokenizer)
    overfit_ds = Subset(full_ds, range(10))
    collate_fn = ASRCollateFn(pad_id=vocab.pad_idx)
    loader = DataLoader(overfit_ds, batch_size=10, shuffle=False, collate_fn=collate_fn)

    model = NepaliASR(
        vocab_size=len(vocab),
        n_mels=80,
        cnn_out_channels=64,
        rnn_hidden_size=256,
        rnn_num_layers=2,
        rnn_dropout=0.0,
        bidirectional=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Beginning 160 epochs on 10 samples...\n")

    for epoch in range(1, 161):
        model.train()
        total_loss = 0.0
        for specs, targets, in_lens, tgt_lens, texts in loader:
            specs = specs.to(device)
            targets = targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)

            optimizer.zero_grad()
            log_probs, sub_lens = model(specs, in_lens)
            log_probs_t = log_probs.transpose(0, 1)

            loss = criterion(log_probs_t, targets, sub_lens, tgt_lens)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch == 160 or epoch == 1:
            model.eval()
            with torch.no_grad():
                log_probs, sub_lens = model(specs, in_lens)
                hyps = decoder.decode_greedy(log_probs, sub_lens)
                cer = compute_corpus_cer(texts, hyps)
                wer = compute_corpus_wer(texts, hyps)
            print(f"Epoch {epoch:03d} | Loss: {total_loss:.4f} | CER: {cer*100:.2f}% | WER: {wer*100:.2f}%")

    print("\n--- Sample Decoding Predictions ---")
    for i, (ref, hyp) in enumerate(zip(texts[:3], hyps[:3])):
        print(f"[Sample {i}]")
        print(f"  Target:  {ref}")
        print(f"  Predict: {hyp}")

    assert total_loss < 1.0, f"Loss did not drop sufficiently: {total_loss}"
    print("\n[+] OVERFIT TEST PASSED: CTC loss dropped successfully and model memorized speech!")

if __name__ == '__main__':
    main()
