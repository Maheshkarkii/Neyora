import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any
from src.data.tokenizer import TranslationTokenizer

class TranslationDataset(Dataset):
    """
    PyTorch Dataset for Nepali -> English Neural Machine Translation.
    
    Item Output:
    - 'src_ids'  : Tensor of shape [T_src], token IDs containing <SOS> and <EOS>, dtype torch.long
    - 'tgt_ids'  : Tensor of shape [T_tgt], token IDs containing <SOS> and <EOS>, dtype torch.long
    - 'raw_src'  : str, Nepali sentence
    - 'raw_tgt'  : str, English sentence
    """
    def __init__(
        self,
        metadata_csv: str,
        src_tokenizer: TranslationTokenizer,
        tgt_tokenizer: TranslationTokenizer
    ):
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
        self.df = pd.read_csv(metadata_csv)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        nepali_text = str(row["nepali_text"])
        english_text = str(row["english_text"])

        src_ids = self.src_tokenizer.encode(nepali_text, add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(english_text, add_special_tokens=True)

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
            "raw_src": nepali_text,
            "raw_tgt": english_text
        }
