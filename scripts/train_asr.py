import os
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data.preprocessing import AudioPreprocessor
from src.data.vocabulary import ASRVocabulary
from src.data.tokenizer import ASRTokenizer
from src.data.asr_dataset import ASRDataset
from src.data.collate import ASRCollateFn
from src.models.asr_model import NepaliASR
from src.decoding.ctc_decoder import CTCDecoder
from src.training.train_asr import ASRTrainer
from src.evaluation.asr_metrics import compute_corpus_cer, compute_corpus_wer
from src.utils.logger import get_logger

logger = get_logger('train_asr_script')

def main():
    print('=== NEPALI ASR TRAINING (PHASE 4) ===')
    device = torch.device('cpu')

    preprocessor = AudioPreprocessor(target_sample_rate=16000, n_mels=80)
    vocab = ASRVocabulary.load('data/processed/asr/vocab.json')
    tokenizer = ASRTokenizer(vocab)
    decoder = CTCDecoder(tokenizer, blank_idx=0, pad_idx=1)

    train_ds = ASRDataset('data/metadata/asr_train.csv', preprocessor, tokenizer, augment=False)
    val_ds = ASRDataset('data/metadata/asr_val.csv', preprocessor, tokenizer, augment=False)
    test_ds = ASRDataset('data/metadata/asr_test.csv', preprocessor, tokenizer, augment=False)

    collate_fn = ASRCollateFn(pad_id=vocab.pad_idx)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    model = NepaliASR(
        vocab_size=len(vocab),
        n_mels=80,
        cnn_out_channels=64,
        rnn_hidden_size=256,
        rnn_num_layers=2,
        rnn_dropout=0.2,
        bidirectional=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    trainer = ASRTrainer(model, decoder, optimizer, device, blank_idx=0, clip_grad_norm=1.0, scheduler=scheduler)

    num_epochs = 35
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    save_path = 'checkpoints/best_nepali_asr.pt'

    print(f'Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Training on {len(train_ds)} samples for {num_epochs} epochs...\n')

    for epoch in range(1, num_epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_cer, val_wer, refs, hyps = trainer.evaluate(val_loader)

        print(
            f'Epoch {epoch:02d}/{num_epochs:02d} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Val Loss: {val_loss:.4f} | '
            f'Val CER: {val_cer*100:.2f}% | '
            f'Val WER: {val_wer*100:.2f}%'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_cer': val_cer,
                'val_wer': val_wer,
                'vocab_size': len(vocab),
            }, save_path)

    print(f"\nSaved best ASR Checkpoint to {save_path}")

    # Evaluate on Test Set
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_cer, test_wer, test_refs, test_hyps = trainer.evaluate(test_loader)

    print("\n=== FINAL TEST SET RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test CER:  {test_cer*100:.2f}%")
    print(f"Test WER:  {test_wer*100:.2f}%")

    print("\n--- Sample Test Transcriptions ---")
    for i, (ref, hyp) in enumerate(zip(test_refs[:5], test_hyps[:5])):
        print(f"[Sample {i}]")
        print(f"  Reference:  {ref}")
        print(f"  Prediction: {hyp}")

if __name__ == '__main__':
    main()
