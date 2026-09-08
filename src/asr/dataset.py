import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
from src.asr.audio_transforms import AudioPreprocessor
from src.asr.tokenizer import ASRTokenizer
from src.common.utils import load_json_manifest

class NepaliASRDataset(Dataset):
    """
    PyTorch Dataset for Nepali Automatic Speech Recognition.

    === SPECIFICATION ===
    Inputs:
      - manifest_path: Path to JSON manifest file containing audio_filepath, text, duration.
      - preprocessor: AudioPreprocessor instance for feature extraction.
      - tokenizer: ASRTokenizer instance for text encoding.
    
    Outputs (__getitem__):
      - Dictionary containing:
          * 'mel_spec': FloatTensor of shape [n_mels=80, T_frames], dtype torch.float32
          * 'tokens': LongTensor of shape [L_tokens], dtype torch.long
          * 'text': str, cleaned Nepali transcript
          * 'duration': float, duration in seconds
          * 'audio_path': str, path to wav file
    """
    def __init__(
        self,
        manifest_path: str,
        preprocessor: AudioPreprocessor,
        tokenizer: ASRTokenizer,
        augment: bool = False
    ):
        self.records = load_json_manifest(manifest_path)
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        audio_path = record["audio_filepath"]
        text = record["text"]
        
        waveform, _ = self.preprocessor.load_audio(audio_path)
        mel_spec = self.preprocessor.extract_features(waveform, augment=self.augment) # [80, T_frames]
        
        token_ids = self.tokenizer.encode(text)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        
        return {
            "mel_spec": mel_spec,
            "tokens": tokens,
            "text": text,
            "duration": record.get("duration", 0.0),
            "audio_path": audio_path
        }


class ASRCollateFn:
    """
    Dynamic Collate and Padding Function for ASR DataLoader.

    === PADDING BEHAVIOR ===
    - Mel Spectrograms: Padded along time axis (dim 1) to max frames in batch with 0.0.
      Final shape: [Batch_Size, n_mels=80, T_max_frames], dtype torch.float32.
    - Token IDs: Padded along length to max target length in batch with pad_id (1).
      Final shape: [Batch_Size, L_max_tokens], dtype torch.long.
    - Input Lengths: 1D LongTensor of shape [Batch_Size], containing unpadded frame count T_i.
    - Target Lengths: 1D LongTensor of shape [Batch_Size], containing unpadded token count L_i.

    Outputs:
      Tuple: (padded_specs, padded_targets, input_lengths, target_lengths, raw_texts)
    """
    def __init__(self, pad_id: int = 1):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        specs = [item["mel_spec"] for item in batch]
        targets = [item["tokens"] for item in batch]
        texts = [item["text"] for item in batch]

        batch_size = len(batch)
        n_mels = specs[0].shape[0]
        
        input_lengths = torch.tensor([s.shape[1] for s in specs], dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        
        max_time = int(input_lengths.max().item())
        max_target_len = max(1, int(target_lengths.max().item()))

        padded_specs = torch.zeros(batch_size, n_mels, max_time, dtype=torch.float32)
        padded_targets = torch.full((batch_size, max_target_len), fill_value=self.pad_id, dtype=torch.long)

        for i in range(batch_size):
            t_len = specs[i].shape[1]
            padded_specs[i, :, :t_len] = specs[i]
            l_len = len(targets[i])
            if l_len > 0:
                padded_targets[i, :l_len] = targets[i]

        return padded_specs, padded_targets, input_lengths, target_lengths, texts


def get_asr_dataloader(
    manifest_path: str,
    preprocessor: AudioPreprocessor,
    tokenizer: ASRTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    augment: bool = False
) -> DataLoader:
    """Creates a ready-to-use PyTorch DataLoader for ASR."""
    dataset = NepaliASRDataset(manifest_path, preprocessor, tokenizer, augment=augment)
    collate_fn = ASRCollateFn(pad_id=tokenizer.pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
