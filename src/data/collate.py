import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Tuple
from src.data.asr_dataset import ASRDataset
from src.data.translation_dataset import TranslationDataset
from src.data.preprocessing import AudioPreprocessor
from src.data.tokenizer import ASRTokenizer, TranslationTokenizer
from src.utils.reproducibility import seed_worker

class ASRCollateFn:
    """
    Collate function for ASR.
    
    Paddings & Tensor Shapes:
    - Mel-spectrograms: Padded along time axis with 0.0
      Output Shape: [B, n_mels=80, max_time], dtype torch.float32
    - Target Tokens: Padded to max token length in batch with pad_id (1)
      Output Shape: [B, max_target_tokens], dtype torch.long
    - input_lengths: 1D LongTensor of shape [B]
    - target_lengths: 1D LongTensor of shape [B]
    """
    def __init__(self, pad_id: int = 1):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        specs = [item["mel_spec"] for item in batch]
        targets = [item["tokens"] for item in batch]
        texts = [item["text"] for item in batch]

        batch_size = len(batch)
        n_mels = specs[0].shape[0]

        input_lengths = torch.tensor([s.shape[1] for s in specs], dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

        max_time = int(input_lengths.max().item())
        max_target_len = max(1, int(target_lengths.max().item()))

        # Explicit shapes: [B, n_mels, max_time]
        padded_specs = torch.zeros(batch_size, n_mels, max_time, dtype=torch.float32)
        padded_targets = torch.full((batch_size, max_target_len), fill_value=self.pad_id, dtype=torch.long)

        for i in range(batch_size):
            t_len = specs[i].shape[1]
            padded_specs[i, :, :t_len] = specs[i]
            l_len = len(targets[i])
            if l_len > 0:
                padded_targets[i, :l_len] = targets[i]

        return padded_specs, padded_targets, input_lengths, target_lengths, texts


class TranslationCollateFn:
    """
    Collate function for Neural Machine Translation.
    
    Paddings & Tensor Shapes:
    - Source sequences: Padded to max source tokens in batch with src_pad_id (0)
      Output Shape: [B, max_src_len], dtype torch.long
    - Target sequences: Padded to max target tokens in batch with tgt_pad_id (0)
      Output Shape: [B, max_tgt_len], dtype torch.long
    - src_lengths: 1D LongTensor of shape [B]
    - tgt_lengths: 1D LongTensor of shape [B]
    """
    def __init__(self, src_pad_id: int = 0, tgt_pad_id: int = 0):
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[str]]:
        src_list = [item["src_ids"] for item in batch]
        tgt_list = [item["tgt_ids"] for item in batch]
        raw_srcs = [item["raw_src"] for item in batch]
        raw_tgts = [item["raw_tgt"] for item in batch]

        batch_size = len(batch)
        src_lengths = torch.tensor([len(s) for s in src_list], dtype=torch.long)
        tgt_lengths = torch.tensor([len(t) for t in tgt_list], dtype=torch.long)

        max_src_len = int(src_lengths.max().item())
        max_tgt_len = int(tgt_lengths.max().item())

        src_padded = torch.full((batch_size, max_src_len), fill_value=self.src_pad_id, dtype=torch.long)
        tgt_padded = torch.full((batch_size, max_tgt_len), fill_value=self.tgt_pad_id, dtype=torch.long)

        for i in range(batch_size):
            src_padded[i, :src_lengths[i]] = src_list[i]
            tgt_padded[i, :tgt_lengths[i]] = tgt_list[i]

        return src_padded, src_lengths, tgt_padded, tgt_lengths, raw_srcs, raw_tgts


def get_asr_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    preprocessor: AudioPreprocessor,
    tokenizer: ASRTokenizer,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train, val, and test DataLoaders for ASR."""
    train_ds = ASRDataset(train_csv, preprocessor, tokenizer, augment=False)
    val_ds = ASRDataset(val_csv, preprocessor, tokenizer, augment=False)
    test_ds = ASRDataset(test_csv, preprocessor, tokenizer, augment=False)

    collate_fn = ASRCollateFn(pad_id=tokenizer.vocab.pad_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    return train_loader, val_loader, test_loader


def get_translation_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train, val, and test DataLoaders for NMT."""
    train_ds = TranslationDataset(train_csv, src_tokenizer, tgt_tokenizer)
    val_ds = TranslationDataset(val_csv, src_tokenizer, tgt_tokenizer)
    test_ds = TranslationDataset(test_csv, src_tokenizer, tgt_tokenizer)

    collate_fn = TranslationCollateFn(
        src_pad_id=src_tokenizer.vocab.pad_idx,
        tgt_pad_id=tgt_tokenizer.vocab.pad_idx
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker
    )
    return train_loader, val_loader, test_loader
