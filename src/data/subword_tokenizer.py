import os
import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Subword Tokenizer built from scratch in PyTorch/Python.
    
    Supports:
    - Devanagari Unicode script and English Latin text
    - Configurable vocabulary size and merge iterations
    - Special tokens: <pad>, <unk>, <sos>, <eos>
    - Deterministic subword encoding with '@@' continuation markers
    - Seamless reversible detokenization
    """
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        is_nepali: bool = False
    ):
        self.is_nepali = is_nepali
        if vocab:
            self.token2idx = vocab
            self.idx2token = {idx: tok for tok, idx in vocab.items()}
        else:
            self.token2idx = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.SOS_TOKEN: 2,
                self.EOS_TOKEN: 3
            }
            self.idx2token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN, 2: self.SOS_TOKEN, 3: self.EOS_TOKEN}
        
        self.merges = merges or []
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.token2idx[self.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    def __len__(self) -> int:
        return len(self.token2idx)

    def train(self, texts: List[str], num_merges: int = 150, min_freq: int = 2) -> None:
        """
        Learns BPE subword merges from training texts (no data leakage from val/test).
        """
        # 1. Initialize character vocabulary with word boundary indicator
        word_freqs = Counter()
        for text in texts:
            words = text.strip().split()
            for w in words:
                if w:
                    # Represent word as characters with end-of-word tag
                    chars = tuple(list(w)[:-1] + [list(w)[-1] + "</w>"])
                    word_freqs[chars] += 1

        # Collect base characters
        base_chars: Set[str] = set()
        for word in word_freqs.keys():
            for symbol in word:
                base_chars.add(symbol)

        for c in sorted(list(base_chars)):
            if c not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[c] = idx
                self.idx2token[idx] = c

        # 2. Iteratively merge most frequent adjacent pairs
        self.merges = []
        for i in range(num_merges):
            pair_counts = defaultdict(int)
            for word, freq in word_freqs.items():
                for j in range(len(word) - 1):
                    pair = (word[j], word[j + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            best_pair = max(pair_counts.keys(), key=lambda p: pair_counts[p])
            if pair_counts[best_pair] < min_freq:
                break

            self.merges.append(best_pair)
            merged_symbol = "".join(best_pair)

            if merged_symbol not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[merged_symbol] = idx
                self.idx2token[idx] = merged_symbol

            # Update vocabulary words with merged symbol
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = []
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best_pair:
                        new_word.append(merged_symbol)
                        j += 2
                    else:
                        new_word.append(word[j])
                        j += 1
                new_word_freqs[tuple(new_word)] = freq
            word_freqs = new_word_freqs

        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def _tokenize_word(self, word: str) -> List[str]:
        if not word:
            return []
        symbols = list(word)[:-1] + [list(word)[-1] + "</w>"]
        if len(symbols) == 1:
            return symbols

        while True:
            # Find all possible pairs
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # Find pair with lowest rank (earliest merge)
            known_pairs = [p for p in pairs if p in self.bpe_ranks]
            if not known_pairs:
                break
            best_pair = min(known_pairs, key=lambda p: self.bpe_ranks[p])

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append("".join(best_pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break

        return symbols

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into subword token strings."""
        if not text:
            return []
        words = text.strip().split()
        subwords = []
        for w in words:
            subwords.extend(self._tokenize_word(w))
        return subwords

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes text into a list of integer token IDs."""
        tokens = self.tokenize(text)
        ids = [self.token2idx.get(t, self.unk_idx) for t in tokens]
        if add_special_tokens:
            ids = [self.sos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back into text."""
        tokens = []
        special_ids = {self.pad_idx, self.sos_idx, self.eos_idx} if skip_special_tokens else set()
        for idx in token_ids:
            if idx in special_ids:
                continue
            tok = self.idx2token.get(idx, self.UNK_TOKEN)
            tokens.append(tok)

        # Reconstruct words from </w> delimiters
        joined = "".join(tokens).replace("</w>", " ")
        # Clean extra spaces
        return re.sub(r"\s+", " ", joined).strip()

    def save(self, file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        data = {
            "token2idx": self.token2idx,
            "merges": self.merges,
            "is_nepali": self.is_nepali
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "BPETokenizer":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Tokenizer file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merges = [tuple(p) for p in data.get("merges", [])]
        return cls(
            vocab=data.get("token2idx", {}),
            merges=merges,
            is_nepali=data.get("is_nepali", False)
        )
