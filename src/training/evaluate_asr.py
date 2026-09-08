import os
import torch
from typing import Dict, Any, Tuple
from src.models.asr_model import NepaliASR
from src.decoding.ctc_decoder import CTCDecoder
from src.evaluation.asr_metrics import compute_corpus_cer, compute_corpus_wer
from src.utils.logger import get_logger

logger = get_logger("ASR_Evaluator")

Ptorch.no_grad()
def evaluate_asr_model(
    model: NepaliASR,
    decoder: CTCDecoder,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float, list, list]:
    model.eval()
    total_loss = 0.0
    all_refs = []
    all_hyps = []

    for specs, targets, in_lens, tgt_lens, texts in dataloader:
        specs = specs.to(device)
        targets = targets.to(device)
        in_lens = in_lens.to(device)
        tgt_lens = tgt_lens.to(device)

        log_probs, sub_lens = model(specs, in_lens)
        log_probs_t = log_probs.transpose(0, 1)
        loss = criterion(log_probs_t, targets, sub_lens, tgt_lens)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()

        hyps = decoder.decode_greedy(log_probs, sub_lens)
        all_refs.extend(texts)
        all_hyps.extend(hyps)

    cer = compute_corpus_cer(all_refs, all_hyps)
    wer = compute_corpus_wer(all_refs, all_hyps)
    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss, cer, wer, all_refs, all_hyps
