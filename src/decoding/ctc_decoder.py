import torch
from typing import List, Union
from src.data.tokenizer import ASRTokenizer

class CTCDecoder:
    def __init__(self, tokenizer: ASRTokenizer, blank_idx: int = 0, pad_idx: int = 1):
        self.tokenizer = tokenizer
        self.blank_idx = blank_idx
        self.pad_idx = pad_idx

    def decode_indices(self, indices: List[int]) -> List[int]:
        collapsed = []
        prev_idx = None
        for idx in indices:
            if idx != prev_idx:
                if idx not in (self.blank_idx, self.pad_idx):
                    collapsed.append(idx)
                prev_idx = idx
        return collapsed

    def decode_greedy(self, log_probs: torch.Tensor, input_lengths: Union[torch.Tensor, List[int]]) -> List[str]:
        if log_probs.ndim == 3 and log_probs.shape[1] == len(input_lengths) and log_probs.shape[0] != len(input_lengths):
            log_probs = log_probs.transpose(0, 1)
        batch_size = log_probs.size(0)
        argmax_preds = torch.argmax(log_probs, dim=-1).detach().cpu().numpy()
        transcriptions = []
        for i in range(batch_size):
            length = int(input_lengths[i])
            raw_seq = argmax_preds[i, :length].tolist()
            filtered_ids = self.decode_indices(raw_seq)
            text = self.tokenizer.decode(filtered_ids, remove_special=True)
            transcriptions.append(text)
        return transcriptions
