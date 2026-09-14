import torch
import math
from collections import defaultdict
from typing import List, Union, Dict, Tuple, Optional
from src.data.tokenizer import ASRTokenizer

class CTCDecoder:
    """
    CTC Decoder implementing:
    1. Fast Greedy Best-Path Decoding
    2. CTC Prefix Beam Search Decoding
    """
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

    def decode_beam_search(
        self,
        log_probs: torch.Tensor,
        input_lengths: Union[torch.Tensor, List[int]],
        beam_width: int = 5
    ) -> List[str]:
        """
        CTC Prefix Beam Search Decoder.
        Args:
            log_probs: Tensor of shape [B, T, V] or [T, B, V]
            input_lengths: Lengths of input frames per batch item
            beam_width: Number of active candidate beams to maintain
        Returns:
            List of decoded transcription strings
        """
        if log_probs.ndim == 3 and log_probs.shape[1] == len(input_lengths) and log_probs.shape[0] != len(input_lengths):
            log_probs = log_probs.transpose(0, 1)

        batch_size = log_probs.size(0)
        vocab_size = log_probs.size(-1)
        # Convert log_probs to probabilities for numerical stability in beam search or keep in log space
        probs = torch.exp(log_probs).detach().cpu().numpy()
        transcriptions: List[str] = []

        for b in range(batch_size):
            T = int(input_lengths[b])
            # Beams store: prefix_tuple -> (p_blank, p_non_blank)
            # Empty prefix has p_blank = 1.0, p_non_blank = 0.0
            beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (1.0, 0.0)}

            for t in range(T):
                next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = defaultdict(lambda: (0.0, 0.0))
                p_frame = probs[b, t] # [V]

                # Top-k characters at this step to speed up beam search
                top_indices = (-p_frame).argsort()[:max(beam_width * 2, 10)]

                for prefix, (p_b, p_nb) in beams.items():
                    p_total = p_b + p_nb
                    if p_total <= 0.0:
                        continue

                    # 1. Transition with blank token
                    p_blank = p_frame[self.blank_idx]
                    curr_b, curr_nb = next_beams[prefix]
                    next_beams[prefix] = (curr_b + p_total * p_blank, curr_nb)

                    # 2. Transition with non-blank characters
                    for c in top_indices:
                        if c in (self.blank_idx, self.pad_idx):
                            continue
                        p_c = p_frame[c]
                        if p_c <= 1e-8:
                            continue

                        end_char = prefix[-1] if prefix else None
                        if c == end_char:
                            # Repeated char: same prefix gets p_nb * p_c (no collapse without blank)
                            curr_b, curr_nb = next_beams[prefix]
                            next_beams[prefix] = (curr_b, curr_nb + p_nb * p_c)

                            # New prefix with extra char gets p_b * p_c
                            new_prefix = prefix + (int(c),)
                            n_b, n_nb = next_beams[new_prefix]
                            next_beams[new_prefix] = (n_b, n_nb + p_b * p_c)
                        else:
                            new_prefix = prefix + (int(c),)
                            n_b, n_nb = next_beams[new_prefix]
                            next_beams[new_prefix] = (n_b, n_nb + p_total * p_c)

                # Prune to top beam_width beams based on total probability
                sorted_beams = sorted(
                    next_beams.items(),
                    key=lambda item: item[1][0] + item[1][1],
                    reverse=True
                )
                beams = dict(sorted_beams[:beam_width])

            if beams:
                best_prefix = max(beams.keys(), key=lambda p: beams[p][0] + beams[p][1])
                text = self.tokenizer.decode(list(best_prefix), remove_special=True)
            else:
                text = ""
            transcriptions.append(text)

        return transcriptions
