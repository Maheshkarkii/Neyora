import unicodedata
import re
from typing import Tuple, Optional

class ParallelTextCleaner:
    """
    Bilingual Text Cleaner for Nepali (Source) and English (Target).
    Ensures Unicode normalization, punctuation harmonization, and strict pair validity.
    """
    def __init__(
        self,
        min_src_len: int = 1,
        max_src_len: int = 50,
        min_tgt_len: int = 1,
        max_tgt_len: int = 50,
        max_len_ratio: float = 2.5
    ):
        self.min_src_len = min_src_len
        self.max_src_len = max_src_len
        self.min_tgt_len = min_tgt_len
        self.max_tgt_len = max_tgt_len
        self.max_len_ratio = max_len_ratio

    def clean_nepali(self, text: Optional[str]) -> str:
        """Cleans Nepali Devanagari text preserving Unicode integrity and danda."""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text.strip())
        text = text.replace("||", "।").replace("?", " ?").replace("!", " !")
        text = re.sub(r"[\s\t\n]+", " ", text).strip()
        return text

    def clean_english(self, text: Optional[str]) -> str:
        """Cleans English target text: lowercases, separates punctuation, normalizes quotes."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKC", text.strip().lower())
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        text = re.sub(r"([.,!?;:\"()])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def validate_pair(self, src: str, tgt: str) -> Tuple[bool, str]:
        """
        Validates whether a (Nepali, English) sentence pair is clean and suitable for NMT.
        Checks length bounds and word-count ratios.
        """
        if not src or not tgt:
            return False, "Empty sentence"
        
        src_tokens = src.split()
        tgt_tokens = tgt.split()
        
        if len(src_tokens) < self.min_src_len or len(src_tokens) > self.max_src_len:
            return False, f"Source length out of bounds ({len(src_tokens)})"
            
        if len(tgt_tokens) < self.min_tgt_len or len(tgt_tokens) > self.max_tgt_len:
            return False, f"Target length out of bounds ({len(tgt_tokens)})"
            
        ratio = max(len(src_tokens), len(tgt_tokens)) / max(1, min(len(src_tokens), len(tgt_tokens)))
        if ratio > self.max_len_ratio:
            return False, f"Token length ratio too large ({ratio:.2f} > {self.max_len_ratio})"
            
        return True, "Valid"
