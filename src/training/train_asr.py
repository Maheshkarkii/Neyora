import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from src.models.asr_model import NepaliASR
from src.decoding.ctc_decoder import CTCDecoder
from src.evaluation.asr_metrics import compute_corpus_cer, compute_corpus_wer
from src.utils.logger import get_logger

logger = get_logger("ASR_Trainer")

class ASRTrainer:
    def __init__(
        self,
        model: NepaliASR,
        decoder: CTCDecoder,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        blank_idx: int = 0,
        clip_grad_norm: float = 1.0,
        scheduler: Any = None
    ):
        self.model = model
        self.decoder = decoder
        self.optimizer = optimizer
        self.device = device
        self.blank_idx = blank_idx
        self.clip_grad_norm = clip_grad_norm
        self.scheduler = scheduler
        self.criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for batch_idx, (specs, targets, in_lens, tgt_lens, texts) in enumerate(dataloader):
            specs = specs.to(self.device)
            targets = targets.to(self.device)
            in_lens = in_lens.to(self.device)
            tgt_lens = tgt_lens.to(self.device)

            self.optimizer.zero_grad()
            log_probs, sub_lens = self.model(specs, in_lens)

            log_probs_t = log_probs.transpose(0, 1)
            loss = self.criterion(log_probs_t, targets, sub_lens, tgt_lens)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss encountered at batch {batch_idx}. Skipping.")
                continue

            loss.backward()

            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

            self.optimizer.step()
            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(1, len(dataloader))

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float, float, List[str], List[str]]:
        self.model.eval()
        total_loss = 0.0
        all_refs = []
        all_hyps = []

        for specs, targets, in_lens, tgt_lens, texts in dataloader:
            specs = specs.to(self.device)
            targets = targets.to(self.device)
            in_lens = in_lens.to(self.device)
            tgt_lens = tgt_lens.to(self.device)

            log_probs, sub_lens = self.model(specs, in_lens)
            log_probs_t = log_probs.transpose(0, 1)
            loss = self.criterion(log_probs_t, targets, sub_lens, tgt_lens)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()

            hyps = self.decoder.decode_greedy(log_probs, sub_lens)
            all_refs.extend(texts)
            all_hyps.extend(hyps)

        cer = compute_corpus_cer(all_refs, all_hyps)
        wer = compute_corpus_wer(all_refs, all_hyps)
        avg_loss = total_loss / max(1, len(dataloader))
        return avg_loss, cer, wer, all_refs, all_hyps
