import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.models.encoder import Encoder
from src.models.decoder_attention import AttentiveDecoder
from src.models.seq2seq_attention import Seq2SeqAttention
from src.utils.logger import get_logger

logger = get_logger("train_attention")

def check_gradients(model: nn.Module) -> Dict[str, float]:
    """Inspects gradients across all parameter groups to detect zero, exploding, or NaN gradients."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            norm = param.grad.norm().item()
            grad_norms[name] = norm
            if torch.isnan(param.grad).any():
                logger.error(f"NaN gradient detected in {name}!")
    return grad_norms

def train_attention_epoch(
    model: Seq2SeqAttention,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    teacher_forcing_ratio: float,
    device: torch.device
) -> Tuple[float, Dict[str, float]]:
    model.train()
    epoch_loss = 0.0
    last_grad_norms = {}

    for batch in dataloader:
        src, _, tgt, _, _, _ = batch
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()

        output, _ = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
        output_dim = output.shape[-1]

        output_flattened = output[:, 1:, :].reshape(-1, output_dim)
        tgt_flattened = tgt[:, 1:].reshape(-1)

        loss = criterion(output_flattened, tgt_flattened)
        loss.backward()

        last_grad_norms = check_gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader), last_grad_norms

def evaluate_attention_epoch(
    model: Seq2SeqAttention,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src, _, tgt, _, _, _ = batch
            src = src.to(device)
            tgt = tgt.to(device)

            output, _ = model(src, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]

            output_flattened = output[:, 1:, :].reshape(-1, output_dim)
            tgt_flattened = tgt[:, 1:].reshape(-1)

            loss = criterion(output_flattened, tgt_flattened)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def build_seq2seq_attention_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    emb_dim: int = 128,
    hid_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
    pad_idx: int = 0,
    device: torch.device = torch.device("cpu")
) -> Seq2SeqAttention:
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    decoder = AttentiveDecoder(
        output_dim=tgt_vocab_size,
        emb_dim=emb_dim,
        enc_hid_dim=hid_dim,
        dec_hid_dim=hid_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx
    )
    model = Seq2SeqAttention(encoder, decoder, pad_idx=pad_idx, device=device).to(device)
    return model
