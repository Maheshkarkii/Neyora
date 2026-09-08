import json
import os
from typing import List, Dict, Optional
from src.asr.text_cleaner import DevanagariTextCleaner

class ASRTokenizer:
    """
    Character/Grapheme Tokenizer for Devanagari ASR (CTC Architecture).
    Includes special tokens:
      <blank>: Index 0 (Mandatory for CTC loss blank token)
      <pad>  : Index 1 (For batch padding)
      <unk>  : Index 2 (Unknown character fallback)
      |      : Index 3 (Word delimiter / space token)
    """
    BLANK_TOKEN = "<blank>"
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SPACE_TOKEN = "|"

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.cleaner = DevanagariTextCleaner()
        if vocab:
            self.char2idx = vocab
            self.idx2char = {idx: char for char, idx in vocab.items()}
        else:
            self.char2idx = {
                self.BLANK_TOKEN: 0,
                self.PAD_TOKEN: 1,
                self.UNK_TOKEN: 2,
                self.SPACE_TOKEN: 3,
            }
            self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    @property
    def blank_id(self) -> int:
        return self.char2idx[self.BLANK_TOKEN]

    @property
    def pad_id(self) -> int:
        return self.char2idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.char2idx[self.UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.char2idx)

    def build_vocab_from_texts(self, texts: List[str]) -> None:
        for text in texts:
            cleaned = self.cleaner.normalize(text)
            for char in cleaned:
                if char == " ":
                    continue
                if char not in self.char2idx:
                    idx = len(self.char2idx)
                    self.char2idx[char] = idx
                    self.idx2char[idx] = char

    def encode(self, text: str) -> List[int]:
        cleaned = self.cleaner.normalize(text)
        token_ids = []
        for char in cleaned:
            if char == " ":
                token_ids.append(self.char2idx[self.SPACE_TOKEN])
            elif char in self.char2idx:
                token_ids.append(self.char2idx[char])
            else:
                token_ids.append(self.unk_id)
        return token_ids

    def decode(self, token_ids: List[int],  remove_special: bool = True) -> str:
        chars = []
        for tid in token_ids:
            if tid not in self.idx2char:
                continue
            char = self.idx2char[tid]
            if remove_special and char in (self.BLANK_TOKEN, self.PAD_TOKEN):
                continue
            if char == self.SPACE_TOKEN:
                chars.append(" ")
            elif char == self.UNK_TOKEN:
                chars.append("<?>")
            else:
                chars.append(char)
        return "".join(chars).strip()

    def save_vocab(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, file_path: str) -> "ASRTokenizer":
        with open(file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
