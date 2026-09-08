import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq
from src.data.tokenizer import TranslationTokenizer
from src.data.collate import get_translation_dataloaders
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed
from src.utils.config_loader import load_config
from src.utils.checkpoint import save_checkpoint
from src.inference.translate import evaluate_corpus_bleu

logger = get_logger("train_translation")

def train_epoch(
    model: Seq2Seq,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    teacher_forcing_ratio: float,
    device: torch.device
) -> float:
    """
    Runs one training epoch.

    Why Gradient Clipping is Useful:
    LSTMs backpropagate through time (BPTT). Over long sequence steps, repeated matrix multiplications
    can cause gradients to explode (scaling exponentially), destabilizing weights or resulting in NaN losses.
    Clipping scales gradients back when their total Euclidean norm exceeds `clip` (e.g. 1.0).
    """
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        src, _, tgt, _, _, _ = batch
        src = src.to(device) # [B, S]
        tgt = tgt.to(device) # [B, T]

        optimizer.zero_grad()

        # Forward pass: output is [B, T, output_dim]
        output = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)

        output_dim = output.shape[-1]

        # Reshape for CrossEntropyLoss:
        # We ignore timestep 0 (<SOS>) of target in loss calculation
        # output: [B * (T-1), output_dim]
        # target: [B * (T-1)]
        output_flattened = output[:, 1:, :].reshape(-1, output_dim)
        tgt_flattened = tgt[:, 1:].reshape(-1)

        loss = criterion(output_flattened, tgt_flattened)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_epoch(
    model: Seq2Seq,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluates model on validation/test set without teacher forcing.
    """
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src, _, tgt, _, _, _ = batch
            src = src.to(device)
            tgt = tgt.to(device)

            # In evaluation, teacher forcing ratio is strictly 0.0
            output = model(src, tgt, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output_flattened = output[:, 1:, :].reshape(-1, output_dim)
            tgt_flattened = tgt[:, 1:].reshape(-1)

            loss = criterion(output_flattened, tgt_flattened)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def build_seq2seq_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    emb_dim: int = 128,
    hid_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
    pad_idx: int = 0,
    device: torch.device = torch.device("cpu")
) -> Seq2Seq:
    """Instantiates Encoder, Decoder, and Seq2Seq."""
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    model = Seq2Seq(encoder, decoder, device=device).to(device)
    return model
