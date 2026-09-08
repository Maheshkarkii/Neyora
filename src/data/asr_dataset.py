import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional, Union
from src.data.preprocessing import AudioPreprocessor
from src.data.tokenizer import ASRTokenizer
from src.utils.reproducibility import seed_worker

class ASRDataset(Dataset):
    """
    PyTorch Dataset for Nepali Speech Recognition.
    
    Dynamic Loading Policy:
    Loads audio dynamically from disk on each `__getitem__` call rather than caching
    all waveforms in RAM, ensuring low memory footprint during scaling.
    
    Item Output:
    - 'mel_spec'   : Tensor of shape [n_mels, time], dtype torch.float32
    - 'tokens'     : Tensor of shape [num_tokens], dtype torch.long
    - 'text'       : str, normalized transcription
    - 'duration'   : float, audio duration in seconds
    - 'speaker_id' : str, speaker identifier
    - 'audio_path' : str, path to audio file
    """
    def __init__(
        self,
        metadata_csv: str,
        preprocessor: AudioPreprocessor,
        tokenizer: ASRTokenizer,
        augment: bool = False
    ):
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata file not found: {metadata_csv}")
        self.df = pd.read_csv(metadata_csv)
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        audio_path = str(row["audio_path"])
        transcription = str(row["transcription"])
        speaker_id = str(row.get("speaker_id", "unknown"))
        duration = float(row.get("duration", 0.0))

        # Dynamic audio load and Log-Mel extraction
        waveform, _ = self.preprocessor.load_audio(audio_path)
        mel_spec = self.preprocessor.extract_mel_spectrogram(waveform, augment=self.augment) # [n_mels, time]

        # Tokenize transcription
        token_ids = self.tokenizer.encode(transcription)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        return {
            "mel_spec": mel_spec,
            "tokens": tokens,
            "text": transcription,
            "duration": duration,
            "speaker_id": speaker_id,
            "audio_path": audio_path
        }
