import json
import os
from collections import Counter
from typing import List, Dict, Optional, Set

class ASRVocabulary:
    """
    Devanagari character vocabulary for CTC-based Automatic Speech Recognition.
    
    Special Tokens:
    - <blank>: 0 (CTC Blank token)
    - <pad>  : 1 (Sequence padding token)
    - <unk>  : 2 (Unknown token)
    - |      : 3 (Word boundary / space separator)
    """
    BLANK = "<blank>"
    PAD = "<pad>"
    UNK = "<unk>"
    SPACE = "|"

    def __init__(self, char2idx: Optional[Dict[str, int]] = None):
        if char2idx:
            self.char2idx = char2idx
            self.idx2char = {idx: char for char, idx in char2idx.items()}
        else:
            self.char2idx = {
                self.BLANK: 0,
                self.PAD: 1,
                self.UNK: 2,
                self.SPACE: 3
            }
            self.idx2char = {0: self.BLANK, 1: self.PAD, 2: self.UNK, 3: self.SPACE}

    @property
    def blank_idx(self) -> int:
        return self.char2idx[self.BLANK]

    @property
    def pad_idx(self) -> int:
        return self.char2idx[self.PAD]

    @property
    def unk_idx(self) -> int:
        return self.char2idx[self.UNK]

    @property
    def space_idx(self) -> int:
        return self.char2idx[self.SPACE]

    def __len__(self) -> int:
        return len(self.char2idx)

    def build_from_texts(self, texts: List[str]) -> None:
        """Extracts unique Devanagari characters from texts and populates the vocabulary."""
        unique_chars: Set[str] = set()
        for text in texts:
            for char in text:
                if char not in (" ", "\t", "\n"):
                    unique_chars.add(char)

        # Sort uniquely for deterministic index assignment
        for char in sorted(list(unique_chars)):
            if char not in self.char2idx:
                idx = len(self.char2idx)
                self.char2idx[char] = idx
                self.idx2char[idx] = char

    def save(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.char2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, json_path: str) -> "ASRVocabulary":
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Vocabulary file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            char2idx = json.load(f)
        return cls(char2idx=char2idx)


class TranslationVocabulary:
    """
    Subword / Word Vocabulary for Neural Machine Translation.
    
    Special Tokens:
    - <PAD>: 0 (Batch padding)
    - <UNK>: 1 (Unknown word fallback)
    - <SOS>: 2 (Start-of-Sequence token)
    - <EOS>: 3 (End-of-Sequence token)
    """
    PAD = "<PAD>"
    UNK = "<UNK>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    def __init__(self, token2idx: Optional[Dict[str, int]] = None):
        if token2idx:
            self.token2idx = token2idx
            self.idx2token = {int(idx): tok for tok, idx in token2idx.items()}
        else:
            self.token2idx = {
                self.PAD: 0,
                self.UNK: 1,
                self.SOS: 2,
                self.EOS: 3
            }
            self.idx2token = {0: self.PAD, 1: self.UNK, 2: self.SOS, 3: self.EOS}

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.SOS]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS]

    def __len__(self) -> int:
        return len(self.token2idx)

    def build_from_texts(self, texts: List[str], min_freq: int = 1, max_size: int = 8000) -> None:
        """Builds vocabulary based on frequency thresholds and maximum size."""
        counter = Counter()
        for text in texts:
            tokens = text.strip().split()
            counter.update(tokens)

        sorted_tokens = [tok for tok, count in counter.most_common() if count >= min_freq]
        available_slots = max_size - len(self.token2idx)

        for tok in sorted_tokens[:available_slots]:
            if tok not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[tok] = idx
                self.idx2token[idx] = tok

    def save(self, json_path: str) -> None:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.token2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, json_path: str) -> "TranslationVocabulary":
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Vocabulary file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            token2idx = json.load(f)
        return cls(token2idx=token2idx)
