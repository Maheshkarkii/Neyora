import unicodedata
import re
from typing import Optional

class DevanagariTextCleaner:
    """
    Text cleaner for Devanagari script designed specifically for ASR.
    Handles Unicode NFC normalization, character whitelisting, punctuation standardization,
    and whitespace collapsing.
    """
    def __init__(self):
        self.devanagari_pattern = re.compile(r"[\u0900-\u097F]")

    def normalize(self, text: Optional[str]) -> str:
        if not text:
            return ""
        # Unicode NFC Normalization
        text = unicodedata.normalize("NFC", text.strip())
        # Standardize Danda (। -> |) and punctuation
        text = text.replace("||", "।").replace("!", "।").replace("?", "।").replace(".", "।")
        # Remove non-Devanagari characters (except whitespace and danda)
        text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
        # Collapse spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_valid_nepali(self, text: str) -> bool:
        if not text or len(text.strip()) == 0:
            return False
        dev_chars = len(self.devanagari_pattern.findall(text))
        return dev_chars > 0 and (dev_chars / max(1, len(text.replace(" ", "")))) > 0.8
