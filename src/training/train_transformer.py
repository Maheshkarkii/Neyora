import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List
from src.models.transformer import Transformer
from src.data.tokenizer import TranslationTokenizer
from src.evaluation.metrics import calculate_sentence_bleu
from src.utils.logger import get_logger

logger = get_logger("train_transformer")

def build_transformer_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 256,
    num_heads: int = 8,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    ffn_dim: int = 512,
    dropout: float = 0.1,
    pad_idx: int = 0,
    device: torch.device = torch.device("cpu")
) -> Transformer:
    """Instantiates the Transformer architecture."""
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
        pad_idx=pad_idx,
        device=device
    ).to(device)
    return model

def train_transformer_epoch(
    model: Transformer,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    gradient_clip: float = 1.0
) -> float:
    """Runs a single training epoch with shifted target teacher forcing."""
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        src_batch, _, tgt_batch, _, _, _ = batch
        src_batch = src_batch.to(device) # [B, S]
        tgt_batch = tgt_batch.to(device) # [B, T]

        # Target shifting for teacher forcing
        # decoder_input: [B, T-1] (excludes last token / EOS)
        # target_labels: [B, T-1] (excludes first token / SOS)
        decoder_input = tgt_batch[:, :-1]
        target_labels = tgt_batch[:, 1:]

        optimizer.zero_grad()
        logits, _ = model(src_batch, decoder_input) # logits: [B, T-1, V_tgt]

        # Flatten for CrossEntropyLoss
        loss = criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            target_labels.contiguous().view(-1)
        )

        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / max(1, len(dataloader))

def evaluate_transformer(
    model: Transformer,
    dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device
) -> float:
    """Evaluates validation/test loss for the Transformer."""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src_batch, _, tgt_batch, _, _, _ = batch
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            decoder_input = tgt_batch[:, :-1]
            target_labels = tgt_batch[:, 1:]

            logits, _ = model(src_batch, decoder_input)
            loss = criterion(
                logits.contiguous().view(-1, logits.size(-1)),
                target_labels.contiguous().view(-1)
            )
            epoch_loss += loss.item()

    return epoch_loss / max(1, len(dataloader))
