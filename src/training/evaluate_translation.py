import os
import torch
import torch.nn as nn
from typing import Dict, Any
from src.models.seq2seq import Seq2Seq
from src.data.tokenizer import TranslationTokenizer
from src.training.train_translation import evaluate_epoch
from src.inference.translate import evaluate_corpus_bleu
from src.utils.logger import get_logger

logger = get_logger("evaluate_translation")

def run_full_evaluation(
    model: Seq2Seq,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Runs loss and BLEU evaluation on validation and test splits."""
    val_loss = evaluate_epoch(model, val_loader, criterion, device)
    test_loss = evaluate_epoch(model, test_loader, criterion, device)

    val_bleu = evaluate_corpus_bleu(model, val_loader, src_tokenizer, tgt_tokenizer, device)
    test_bleu = evaluate_corpus_bleu(model, test_loader, src_tokenizer, tgt_tokenizer, device)

    metrics = {
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val_bleu": val_bleu,
        "test_bleu": test_bleu
    }

    logger.info(f"Evaluation Metrics: Val Loss={val_loss:.4f}, Val BLEU={val_bleu:.2f} | Test Loss={test_loss:.4f}, Test BLEU={test_bleu:.2f}")
    return metrics
