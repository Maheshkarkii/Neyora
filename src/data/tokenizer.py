from typing import List, Optional
from src.data.vocabulary import ASRVocabulary, TranslationVocabulary
from src.data.preprocessing import NepaliTextCleaner, EnglishTextCleaner

class ASRTokenizer:
    """
    Tokenizer for ASR mapping Devanagari characters to integer IDs and back.
    Encodes whitespace to word delimiter '|' (ID 3).
    """
    def __init__(self, vocab: ASRVocabulary):
        self.vocab = vocab
        self.cleaner = NepaliTextCleaner()

    def encode(self, text: str) -> List[int]:
        """Encodes text to a sequence of character token IDs."""
        cleaned = self.cleaner.normalize(text)
        token_ids: List[int] = []
        for char in cleaned:
            if char == " ":
                token_ids.append(self.vocab.space_idx)
            elif char in self.vocab.char2idx:
                token_ids.append(self.vocab.char2idx[char])
            else:
                token_ids.append(self.vocab.unk_idx)
        return token_ids

    def decode(self, token_ids: List[int], remove_special: bool = True) -> str:
        """Decodes integer token IDs back to a reconstructed Devanagari string."""
        chars: List[str] = []
        for tid in token_ids:
            if tid not in self.vocab.idx2char:
                continue
            char = self.vocab.idx2char[tid]
            if remove_special and char in (self.vocab.BLANK, self.vocab.PAD):
                continue
            if char == self.vocab.SPACE:
                chars.append(" ")
            elif char == self.vocab.UNK:
                chars.append("<?>")
            else:
                chars.append(char)
        return "".join(chars).strip()


class TranslationTokenizer:
    """
    Tokenizer for Machine Translation (Nepali source & English target).
    Adds <SOS> and <EOS> tokens during sequence encoding.
    """
    def __init__(self, vocab: TranslationVocabulary, is_nepali: bool = False):
        self.vocab = vocab
        self.cleaner = NepaliTextCleaner() if is_nepali else EnglishTextCleaner()

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes space-delimited text into integer token IDs."""
        cleaned = self.cleaner.normalize(text)
        tokens = cleaned.split()
        token_ids = [self.vocab.token2idx.get(tok, self.vocab.unk_idx) for tok in tokens]
        if add_special_tokens:
            return [self.vocab.sos_idx] + token_ids + [self.vocab.eos_idx]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes integer token IDs back to a readable sentence string."""
        words: List[str] = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.vocab.pad_idx, self.vocab.sos_idx, self.vocab.eos_idx):
                continue
            if tid == self.vocab.unk_idx:
                words.append("<UNK>")
            else:
                words.append(self.vocab.idx2token.get(tid, "<UNK>"))
        return " ".join(words).strip()
