import json
import os
from collections import Counter
from typing import List, Dict, Optional, Union

class NMTTokenizer:
    """
    Vocabulary and Tokenizer for Neural Machine Translation.
    Handles special tokens:
      <pad>: 0 (Padding)
      <unk>: 1 (Unknown)
      <sos>: 2 (Start of Sequence)
      <eos>: 3 (End of Sequence)
    """
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab:
            self.word2idx = vocab
            self.idx2word = {int(idx): word for word, idx in vocab.items()}
        else:
            self.word2idx = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.SOS_TOKEN: 2,
                self.EOS_TOKEN: 3
            }
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.word2idx[self.SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.word2idx[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def build_vocab(self, texts: List[str], min_freq: int = 1, max_vocab_size: int = 8000) -> None:
        """Builds vocabulary from a list of pre-tokenized/cleaned sentences."""
        counter = Counter()
        for text in texts:
            tokens = text.strip().split()
            counter.update(tokens)

        sorted_words = [word for word, count in counter.most_common() if count >= min_freq]
        
        available_slots = max_vocab_size - len(self.word2idx)
        for word in sorted_words[:available_slots]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes space-delimited text into integer token IDs."""
        tokens = text.strip().split()
        token_ids = [self.word2idx.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            return [self.sos_id] + token_ids + [self.eos_id]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back into a readable string."""
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.pad_id, self.sos_id, self.eos_id):
                continue
            if tid == self.unk_id:
                words.append("<unk>")
            else:
                words.append(self.idx2word.get(tid, "<unk>"))
        return " ".join(words).strip()

    def save_vocab(self, file_path: str) -> None:
        """Saves vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, file_path: str) -> "NMTTokenizer":
        """Loads tokenizer from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
