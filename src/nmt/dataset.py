import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from src.nmt.tokenizer import NMTTokenizer
from src.common.utils import load_json_manifest

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for Nepali-to-English Neural Machine Translation.

    === SPECIFICATION ===
    Inputs:
      - json_path: Path to dataset JSON containing 'src' (Nepali) and 'tgt' (English) fields.
      - src_tokenizer: NMTTokenizer for source language (Nepali).
      - tgt_tokenizer: NMTTokenizer for target language (English).

    Outputs (__getitem__):
      - Dictionary containing:
          * 'src_ids': LongTensor of shape [T_src], token IDs with <sos> and <eos>
          * 'tgt_ids': LongTensor of shape [T_tgt], token IDs with <sos> and <eos>
          * 'raw_src': str, raw Nepali sentence
          * 'raw_tgt': str, raw English sentence
    """
    def __init__(
        self,
        json_path: str,
        src_tokenizer: NMTTokenizer,
        tgt_tokenizer: NMTTokenizer
    ):
        self.records = load_json_manifest(json_path)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        src_text = record["src"]
        tgt_text = record["tgt"]

        src_token_ids = self.src_tokenizer.encode(src_text, add_special_tokens=True)
        tgt_token_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)

        return {
            "src_ids": torch.tensor(src_token_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_token_ids, dtype=torch.long),
            "raw_src": src_text,
            "raw_tgt": tgt_text
        }


class TranslationCollateFn:
    """
    Dynamic Collate Function for NMT DataLoader.

    === PADDING BEHAVIOR ===
    - Source Tensors: Padded to max source length in batch with pad_id (0).
      Final shape: [Batch_Size, T_src_max], dtype torch.long.
    - Target Tensors: Padded to max target length in batch with pad_id (0).
      Final shape: [Batch_Size, T_tgt_max], dtype torch.long.
    - Source Lengths: LongTensor of shape [Batch_Size], containing unpadded lengths.
    - Target Lengths: LongTensor of shape [Batch_Size], containing unpadded lengths.

    Outputs:
      Tuple: (src_padded, src_lengths, tgt_padded, tgt_lengths, raw_sources, raw_targets)
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


def get_nmt_dataloader(
    json_path: str,
    src_tokenizer: NMTTokenizer,
    tgt_tokenizer: NMTTokenizer,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Creates a ready-to-use PyTorch DataLoader for NMT."""
    dataset = TranslationDataset(json_path, src_tokenizer, tgt_tokenizer)
    collate_fn = TranslationCollateFn(
        src_pad_id=src_tokenizer.pad_id,
        tgt_pad_id=tgt_tokenizer.pad_id
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
